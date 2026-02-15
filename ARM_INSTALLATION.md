# ARM Platform Installation Guide

This guide covers installation on ARM-based systems including Raspberry Pi, Apple Silicon (M1/M2/M3), and other ARM devices.

## Quick Fix for "Illegal instruction" Error

If you're seeing:
```
Illegal instruction (core dumped)
```

This means PyTorch/OpenVINO were installed with x86_64 binaries. **Fix immediately:**

```bash
# 1. Remove incompatible packages
pip uninstall -y torch torchvision openvino nncf

# 2. Reinstall for your platform (see sections below)
```

Then follow the platform-specific instructions below.

---

## Problem

The standard `pip install torch` and `pip install openvino` commands install x86_64 wheels that are compiled with AVX/SSE CPU instructions. These instructions do not exist on ARM processors, leading to:

```
Illegal instruction (core dumped)
```

## Solutions by Platform

### 🍎 Apple Silicon (M1/M2/M3/M4) — macOS

**Status**: ✅ Fully Supported

Apple Silicon has excellent support for both PyTorch and OpenVINO. Simply use the standard installation:

```bash
# Create conda environment
conda env create -f environment.yml
conda activate fashion-bench

# Or install manually:
conda create -n fashion-bench python=3.11
conda activate fashion-bench
pip install torch torchvision openvino nncf onnx onnxscript
```

PyTorch provides native ARM64 wheels optimized for Apple Silicon, and OpenVINO supports ARM64 on macOS.

---

### 🥧 Raspberry Pi (ARMv7/ARMv8) — Linux

**Status**: ⚠️ Experimental

Raspberry Pi requires special handling for both PyTorch and OpenVINO.

#### Option 1: PyTorch CPU-only (ARMv8 / 64-bit OS)

For 64-bit Raspberry Pi OS or Ubuntu on Pi 4/5:

```bash
# Remove incompatible packages if already installed
pip uninstall -y torch torchvision openvino

# Install PyTorch for ARM64
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# OpenVINO: Use pre-built ARM wheels (if available)
pip install openvino==2025.4.1

# NNCF for quantization
pip install nncf==2.19.0
```

**Note**: OpenVINO ARM support is experimental. If you encounter issues, see Option 3 below.

#### Option 2: PyTorch from Source (ARMv7 / 32-bit OS)

For 32-bit Raspberry Pi OS, you'll need to build PyTorch from source:

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y python3-dev libopenblas-dev libjpeg-dev zlib1g-dev

# Build PyTorch (this takes 6-12 hours on a Pi 4)
pip install --no-cache-dir torch torchvision --no-binary :all:
```

This is **very slow** on Raspberry Pi and may require swap space expansion.

#### Option 3: Use ONNX Runtime Instead (Recommended for Pi)

For inference-only workloads, ONNX Runtime is better optimized for ARM:

```bash
pip install onnxruntime numpy
```

**Code changes required**: You'll need to modify `src/benchmark.py` to use ONNX Runtime instead of OpenVINO for inference.

#### Option 4: Train on x86, Benchmark on ARM

The best approach for Raspberry Pi:

1. **On x86/x64 machine** (PC, Mac, cloud VM):
   - Train the model: `python -m src.main train`
   - Optimize to ONNX: `python -m src.main optimize`
   - Copy `models/openvino/fashion_mnist.onnx` to Raspberry Pi

2. **On Raspberry Pi**:
   - Install ONNX Runtime: `pip install onnxruntime`
   - Run inference using the ONNX model (requires code modifications)

---

### 🔧 Other ARM Linux (NVIDIA Jetson, AWS Graviton, etc.)

For ARM64 servers and edge devices:

```bash
# Most ARM64 Linux systems work with the standard PyTorch ARM wheel:
pip install torch torchvision

# OpenVINO may require building from source
git clone https://github.com/openvinotoolkit/openvino.git
cd openvino
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(nproc)
```

Refer to [OpenVINO Build Instructions](https://github.com/openvinotoolkit/openvino/wiki/BuildingForLinuxSystems) for detailed steps.

---

## Verifying Installation

After installation, verify PyTorch and OpenVINO load correctly:

```bash
python3 -c "import torch; print(f'PyTorch {torch.__version__} on {torch.__config__.show()}')"
python3 -c "import openvino as ov; print(f'OpenVINO {ov.__version__}')"
```

If you see "Illegal instruction", the packages are not ARM-compatible.

---

## Performance Expectations

| Platform | Training Speed | Inference Speed | Recommended Use |
|----------|---------------|----------------|-----------------|
| Apple M1 Pro | Fast | Very Fast | ✅ Full pipeline |
| Raspberry Pi 4 (8GB) | Very Slow | Moderate | ⚠️ Inference only |
| Raspberry Pi 5 | Slow | Moderate | ⚠️ Inference only |
| AWS Graviton3 | Moderate | Fast | ✅ Full pipeline |

---

## Troubleshooting

### "Illegal instruction (core dumped)"

- You're using x86_64 binaries on ARM → Follow the installation steps above for your platform

### "Cannot find -ltorch_cpu" when building from source

- Install dependencies: `sudo apt-get install libopenblas-dev libjpeg-dev zlib1g-dev`

### OpenVINO fails to import on ARM

- Use ONNX Runtime instead, or build OpenVINO from source
- Check [OpenVINO ARM support status](https://github.com/openvinotoolkit/openvino/issues)

### Training is too slow on Raspberry Pi

- Train on a PC/Mac/cloud VM, then copy the model files to Pi for inference
- Consider using a smaller model or fewer epochs

---

## Quick Reference

| Platform | PyTorch | OpenVINO | Status |
|----------|---------|----------|--------|
| macOS ARM64 (M1/M2/M3) | `pip install torch` | `pip install openvino` | ✅ Works |
| Linux ARM64 (64-bit) | `pip install torch` | Build from source | ⚠️ Experimental |
| Linux ARMv7 (32-bit Pi) | Build from source | Not supported | ❌ Use ONNX Runtime |
| Windows ARM64 | `pip install torch` | Not tested | ⚠️ Unknown |

---

For questions or issues, please open a GitHub issue with your:
- Platform: `uname -m` and `lsb_release -a`
- Python version: `python --version`
- Error message and full traceback
