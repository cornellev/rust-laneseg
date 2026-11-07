#!/usr/bin/env bash
# Jetson Thor (JetPack 7, CUDA 13) GPU runtime for lane-seg, without sudo:
# ONNX Runtime with the TensorRT + CUDA execution providers, TensorRT 10.13
# and the CUDA libraries they link, unpacked from pip wheels into ort-runtime/.
#
#   tools/setup_jetson_runtime.sh               # once (downloads ~3 GB) into ./ort-runtime
#   tools/setup_jetson_runtime.sh /opt/cev/ort-runtime   # on the car (see README)
#   source ort-runtime/env.sh                   # in every shell that runs clrnet-lane
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
dest="${1:-$root/ort-runtime}"
mkdir -p "$dest"

# Only the shared libraries are used, so the wheel's Python tag is irrelevant.
python3 -m pip install --upgrade --no-deps --target "$dest" \
    --python-version 3.12 --platform linux_aarch64 --only-binary=:all: \
    --index-url https://pypi.jetson-ai-lab.io/sbsa/cu130 onnxruntime-gpu==1.24.0
python3 -m pip install --upgrade --no-deps --target "$dest" \
    --extra-index-url https://pypi.nvidia.com \
    tensorrt-cu13-libs==10.13.3.9 nvidia-cublas nvidia-cuda-runtime \
    nvidia-cudnn-cu13 nvidia-cufft nvidia-cuda-nvrtc nvidia-nvjitlink

cat > "$dest/env.sh" <<ENV
export ORT_DYLIB_PATH="$dest/onnxruntime/capi/libonnxruntime.so.1.24.0"
export LD_LIBRARY_PATH="$dest/onnxruntime/capi:$dest/tensorrt_libs:\$(printf '%s:' "$dest"/nvidia/*/lib)\${LD_LIBRARY_PATH:-}"
ENV
echo "done: source $dest/env.sh"
