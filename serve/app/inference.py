"""Inference backends for the PatchCore anomaly-detection API.

Every exported model has the same contract (see the notebook, Step 6):
    input  "image"       uint8  [B, 224, 224, 3]  RGB, NHWC, already resized+cropped
    output "anomaly_map" float32 [B, 1, 224, 224]
    output "score"       float32 [B]
Normalisation happens inside the graph, so the server only decodes, resizes and crops.
"""
from __future__ import annotations

import io
import json
import logging
import re
import threading
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger("patchcore")


# --------------------------------------------------------------------------- #
# Pre-processing: MUST match `load_image` in the notebook exactly.
# --------------------------------------------------------------------------- #
def preprocess(data: bytes, resize: int, crop: int) -> np.ndarray:
    """Encoded image bytes -> uint8 [crop, crop, 3] RGB."""
    with Image.open(io.BytesIO(data)) as im:
        im = im.convert("RGB").resize((resize, resize), Image.Resampling.BILINEAR)
    off = (resize - crop) // 2
    im = im.crop((off, off, off + crop, off + crop))
    return np.ascontiguousarray(np.asarray(im, dtype=np.uint8))


# --------------------------------------------------------------------------- #
# ONNX Runtime backend (CUDA or CPU)
# --------------------------------------------------------------------------- #
class ORTRunner:
    def __init__(self, onnx_path: Path, device: str = "cuda"):
        import onnxruntime as ort

        if device == "cuda" and hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:  # noqa: BLE001 - best effort, falls back to system libs
                pass
        providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                     if device == "cuda" else ["CPUExecutionProvider"])
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(onnx_path), so, providers=providers)
        active = self.sess.get_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in active:
            raise RuntimeError(
                f"CUDAExecutionProvider not active (got {active}). Check that the installed "
                "onnxruntime-gpu build matches the container's CUDA/cuDNN versions.")
        self.description = f"onnxruntime-{device} ({active[0]})"

    def __call__(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        amap, score = self.sess.run(["anomaly_map", "score"], {"image": batch})
        return amap, score


# --------------------------------------------------------------------------- #
# TensorRT backend (TensorRT 10.x and 11.x, strongly-typed networks)
# --------------------------------------------------------------------------- #
def build_engine(onnx_path: Path, engine_path: Path, max_batch: int = 1,
                 opt_batch: int = 1, workspace_gb: float = 2.0) -> Path:
    """Build a serialized TensorRT engine from an ONNX file.

    Precision is decided by the ONNX graph itself (strong typing): the fp16 ONNX
    contains an fp16 backbone and an fp32 nearest-neighbour head. This works on
    TensorRT 10 (STRONGLY_TYPED flag) and TensorRT 11 (strong typing is the only mode).
    """
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if int(trt.__version__.split(".")[0]) < 11:
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{errs}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    inp = network.get_input(0)
    _, h, w, c = inp.shape
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, (1, h, w, c), (min(opt_batch, max_batch), h, w, c),
                      (max_batch, h, w, c))
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(serialized))
    return engine_path


class TRTRunner:
    """Runs a TensorRT engine with CuPy-managed device buffers (allocated once)."""

    def __init__(self, engine_path: Path, max_batch: int = 1):
        import cupy as cp
        import tensorrt as trt

        self.cp, self.trt = cp, trt
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"Could not deserialize {engine_path}")
        self.context = self.engine.create_execution_context()
        self.stream = cp.cuda.Stream(non_blocking=True)
        self.lock = threading.Lock()  # one execution context -> serialise calls
        self.max_batch = max_batch

        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.in_name = next(n for n in names
                            if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT)
        self.out_names = [n for n in names if n != self.in_name]

        # Allocate every buffer at max batch once; smaller batches use a prefix.
        in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        full_in = (max_batch,) + in_shape[1:]
        self.context.set_input_shape(self.in_name, full_in)
        self.buffers = {self.in_name: cp.empty(full_in, dtype=np.uint8)}
        for n in self.out_names:
            shape = tuple(self.context.get_tensor_shape(n))
            dtype = trt.nptype(self.engine.get_tensor_dtype(n))
            self.buffers[n] = cp.empty(shape, dtype=dtype)
        self.description = f"tensorrt-{trt.__version__}"

    def __call__(self, batch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        b = batch.shape[0]
        if b > self.max_batch:
            raise ValueError(f"batch {b} > engine max batch {self.max_batch}")
        with self.lock, self.stream:
            d_in = self.buffers[self.in_name]
            d_in[:b].set(np.ascontiguousarray(batch), stream=self.stream)
            self.context.set_input_shape(self.in_name, batch.shape)
            for name, buf in self.buffers.items():
                self.context.set_tensor_address(name, buf.data.ptr)
            if not self.context.execute_async_v3(self.stream.ptr):
                raise RuntimeError("TensorRT execution failed")
            out = {n: self.buffers[n][:b].get(stream=self.stream) for n in self.out_names}
            self.stream.synchronize()
        return out["anomaly_map"], out["score"]


def gpu_tag() -> str:
    import cupy as cp
    name = cp.cuda.runtime.getDeviceProperties(0)["name"]
    name = name.decode() if isinstance(name, bytes) else str(name)
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()


# --------------------------------------------------------------------------- #
# One loaded category
# --------------------------------------------------------------------------- #
class CategoryModel:
    def __init__(self, model_dir: Path, backend: str, precision: str,
                 cache_dir: Path, max_batch: int = 1):
        self.meta = json.loads((model_dir / "meta.json").read_text())
        self.category = self.meta["category"]
        self.resize = int(self.meta["preprocess"]["resize"])
        self.crop = int(self.meta["preprocess"]["crop"])
        self.image_threshold = float(self.meta["thresholds"]["image"])
        self.pixel_threshold = float(self.meta["thresholds"]["pixel"])
        onnx_path = model_dir / f"model_{precision}.onnx"
        if not onnx_path.exists():
            raise FileNotFoundError(onnx_path)

        if backend == "trt":
            import tensorrt as trt
            engine = cache_dir / (f"{self.category}_{precision}_b{max_batch}_"
                                  f"trt{trt.__version__}_{gpu_tag()}.engine")
            if not engine.exists():
                log.info("Building TensorRT engine for %s (%s) -> %s", self.category, precision, engine)
                build_engine(onnx_path, engine, max_batch=max_batch)
            self.runner = TRTRunner(engine, max_batch=max_batch)
        elif backend in ("ort", "ort-cuda"):
            self.runner = ORTRunner(onnx_path, device="cuda")
        elif backend == "ort-cpu":
            self.runner = ORTRunner(onnx_path, device="cpu")
        else:
            raise ValueError(f"Unknown BACKEND={backend!r}")
        # warm-up (first call triggers lazy init / autotuning)
        dummy = np.zeros((1, self.crop, self.crop, 3), np.uint8)
        for _ in range(3):
            self.runner(dummy)

    def predict(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        amap, score = self.runner(img[None])
        return np.asarray(amap[0, 0], dtype=np.float32), float(score[0])


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
def _jet(v: np.ndarray) -> np.ndarray:
    """Matplotlib-like 'jet' colormap without the matplotlib dependency. v in [0,1]."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], -1) * 255).astype(np.uint8)


def render_overlay(img: np.ndarray, amap: np.ndarray, pixel_threshold: float,
                   alpha: float = 0.45) -> bytes:
    """Heat-map overlay; the colour scale saturates at 2x the pixel threshold."""
    v = np.clip(amap / (2.0 * pixel_threshold + 1e-12), 0.0, 1.0)
    over = (1 - alpha) * img.astype(np.float32) + alpha * _jet(v).astype(np.float32)
    buf = io.BytesIO()
    Image.fromarray(over.clip(0, 255).astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()
