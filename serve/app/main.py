"""FastAPI service for PatchCore anomaly detection on MVTec AD categories.

Environment variables
---------------------
MODEL_DIR     directory with one sub-folder per category (meta.json + model_*.onnx)  [/models]
ENGINE_CACHE  where TensorRT engines are built/cached                                 [/cache]
BACKEND       trt | ort-cuda | ort-cpu                                                [trt]
PRECISION     fp16 | fp32                                                             [fp16]
CATEGORIES    comma-separated list, or "all"                                          [all]
MAX_BATCH     max batch size baked into TensorRT engines                              [1]
MAX_UPLOAD_MB reject larger uploads                                                    [20]
"""
from __future__ import annotations

import base64
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from PIL import UnidentifiedImageError

from .inference import CategoryModel, preprocess, render_overlay

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("patchcore")

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
ENGINE_CACHE = Path(os.getenv("ENGINE_CACHE", "/cache"))
BACKEND = os.getenv("BACKEND", "trt")
PRECISION = os.getenv("PRECISION", "fp16")
CATEGORIES = os.getenv("CATEGORIES", "all")
MAX_BATCH = int(os.getenv("MAX_BATCH", "1"))
MAX_UPLOAD = int(float(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024)

MODELS: dict[str, CategoryModel] = {}


def _discover() -> list[str]:
    found = sorted(p.name for p in MODEL_DIR.iterdir() if (p / "meta.json").exists())
    if CATEGORIES.strip().lower() == "all":
        return found
    wanted = [c.strip() for c in CATEGORIES.split(",") if c.strip()]
    missing = set(wanted) - set(found)
    if missing:
        raise RuntimeError(f"Categories not found in {MODEL_DIR}: {sorted(missing)}")
    return wanted


@asynccontextmanager
async def lifespan(_: FastAPI):
    t0 = time.perf_counter()
    for cat in _discover():
        log.info("Loading %s (backend=%s, precision=%s)", cat, BACKEND, PRECISION)
        MODELS[cat] = CategoryModel(MODEL_DIR / cat, BACKEND, PRECISION, ENGINE_CACHE, MAX_BATCH)
    if not MODELS:
        raise RuntimeError(f"No models found in {MODEL_DIR}")
    log.info("Loaded %d categories in %.1fs", len(MODELS), time.perf_counter() - t0)
    yield
    MODELS.clear()


app = FastAPI(title="MVTec AD PatchCore API", version="1.0.0", lifespan=lifespan)


def _get(category: str) -> CategoryModel:
    m = MODELS.get(category)
    if m is None:
        raise HTTPException(404, f"Unknown category '{category}'. Available: {sorted(MODELS)}")
    return m


async def _read(file: UploadFile) -> bytes:
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty upload")
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, f"Upload larger than {MAX_UPLOAD // (1024 * 1024)} MB")
    return data


async def _infer(m: CategoryModel, data: bytes):
    t0 = time.perf_counter()
    try:
        img = await run_in_threadpool(preprocess, data, m.resize, m.crop)
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(400, f"Could not decode image: {e}") from e
    t1 = time.perf_counter()
    amap, score = await run_in_threadpool(m.predict, img)
    t2 = time.perf_counter()
    return img, amap, score, {"preprocess_ms": (t1 - t0) * 1e3, "inference_ms": (t2 - t1) * 1e3}


@app.get("/health")
def health():
    return {"status": "ok", "backend": BACKEND, "precision": PRECISION,
            "categories": sorted(MODELS),
            "runtime": next(iter(MODELS.values())).runner.description if MODELS else None}


@app.get("/categories")
def categories():
    return {c: {"image_threshold": m.image_threshold, "pixel_threshold": m.pixel_threshold,
                "input_size": m.crop, "metrics": m.meta.get("metrics", {})}
            for c, m in MODELS.items()}


@app.post("/predict/{category}")
async def predict(category: str, file: UploadFile = File(...),
                  return_heatmap: bool = Query(False, description="include a base64 PNG overlay")):
    t_start = time.perf_counter()
    m = _get(category)
    data = await _read(file)
    img, amap, score, timing = await _infer(m, data)
    resp = {
        "category": category,
        "score": score,
        "threshold": m.image_threshold,
        "normalized_score": score / m.image_threshold,
        "is_anomalous": bool(score > m.image_threshold),
        "defect_area_fraction": float(np.mean(amap > m.pixel_threshold)),
    }
    if return_heatmap:
        png = await run_in_threadpool(render_overlay, img, amap, m.pixel_threshold)
        resp["heatmap_png_base64"] = base64.b64encode(png).decode()
    timing["total_ms"] = (time.perf_counter() - t_start) * 1e3
    resp["timing_ms"] = {k: round(v, 3) for k, v in timing.items()}
    return resp


@app.post("/predict/{category}/heatmap", response_class=Response,
          responses={200: {"content": {"image/png": {}}}})
async def predict_heatmap(category: str, file: UploadFile = File(...)):
    m = _get(category)
    data = await _read(file)
    img, amap, score, _ = await _infer(m, data)
    png = await run_in_threadpool(render_overlay, img, amap, m.pixel_threshold)
    return Response(png, media_type="image/png", headers={
        "X-Anomaly-Score": f"{score:.6f}",
        "X-Is-Anomalous": str(score > m.image_threshold).lower()})
