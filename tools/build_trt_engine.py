"""Build a TensorRT engine from an ONNX model in a clean process.

Run it in its own process (the notebook does this via subprocess) so that:
  * no PyTorch / ONNX Runtime CUDA DLLs are loaded next to TensorRT's (a common crash source on Windows),
  * a native crash inside TensorRT cannot kill the Jupyter kernel.

Usage:
    python tools/build_trt_engine.py --onnx artifacts/models/bottle/model_fp16.onnx \
        --engine artifacts/engines/bottle_fp16.engine --max-batch 8 --gpu 1 [--verbose]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True, type=Path)
    ap.add_argument("--engine", required=True, type=Path)
    ap.add_argument("--max-batch", type=int, default=8)
    ap.add_argument("--opt-batch", type=int, default=1)
    ap.add_argument("--workspace-gb", type=float, default=2.0)
    ap.add_argument("--gpu", default=None, help="nvidia-smi index; default: inherit CUDA_VISIBLE_DEVICES")
    ap.add_argument("--verbose", action="store_true", help="TensorRT VERBOSE logging (for debugging)")
    args = ap.parse_args()

    # GPU selection must happen before TensorRT initialises CUDA.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"[build] python {sys.version.split()[0]} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
          flush=True)
    import tensorrt as trt
    print(f"[build] TensorRT {trt.__version__}", flush=True)

    logger = trt.Logger(trt.Logger.VERBOSE if args.verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if int(trt.__version__.split(".")[0]) < 11:          # TRT 10: opt in to strong typing
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    print(f"[build] parsing {args.onnx}", flush=True)
    if not parser.parse_from_file(str(args.onnx)):
        for i in range(parser.num_errors):
            print("[build] parser error:", parser.get_error(i), file=sys.stderr)
        return 2

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))
    inp = network.get_input(0)
    _, h, w, c = inp.shape
    profile = builder.create_optimization_profile()
    profile.set_shape(inp.name, (1, h, w, c), (min(args.opt_batch, args.max_batch), h, w, c),
                      (args.max_batch, h, w, c))
    config.add_optimization_profile(profile)

    print(f"[build] building (input {inp.name} {tuple(inp.shape)} {inp.dtype}, batch 1..{args.max_batch}) ...",
          flush=True)
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("[build] build_serialized_network returned None (see TensorRT messages above)", file=sys.stderr)
        return 3
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(bytes(serialized))
    dt = time.perf_counter() - t0

    # Sanity check: the engine deserialises and exposes the expected I/O tensors.
    engine = trt.Runtime(logger).deserialize_cuda_engine(args.engine.read_bytes())
    if engine is None:
        print("[build] engine was written but cannot be deserialised", file=sys.stderr)
        return 4
    io = [f"{engine.get_tensor_name(i)}:{engine.get_tensor_dtype(engine.get_tensor_name(i))}"
          for i in range(engine.num_io_tensors)]
    print(f"[build] OK in {dt:.0f}s -> {args.engine} ({args.engine.stat().st_size / 2**20:.0f} MB) | io: {io}",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
