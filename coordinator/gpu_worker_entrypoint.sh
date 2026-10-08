#!/bin/sh
set -eu

python3 - <<'PY'
import json

import paddle

if not paddle.device.is_compiled_with_cuda():
    raise SystemExit("GPU worker requires a CUDA-enabled Paddle build")

device_count = paddle.device.cuda.device_count()
if device_count < 1:
    raise SystemExit("GPU worker requires at least one visible CUDA device")

paddle.device.set_device("gpu:0")
probe = paddle.to_tensor([1.0, 2.0, 3.0]).square().sum().numpy().item()
if probe != 14.0:
    raise SystemExit(f"GPU tensor probe returned {probe!r}, expected 14.0")

print(
    json.dumps(
        {
            "event": "gpu_runtime_verified",
            "paddle_version": paddle.__version__,
            "device": paddle.device.get_device(),
            "visible_device_count": device_count,
            "tensor_probe": probe,
        },
        sort_keys=True,
    ),
    flush=True,
)
PY

exec "$@"
