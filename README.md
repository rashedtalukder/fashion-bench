# Fashion MNIST Performance Evaluator

Train a CNN on Fashion MNIST with PyTorch, export to OpenVINO IR, generate quantized variants (FP32, INT8, INT4), and benchmark inference across devices.

## Supported Platforms

| Platform | Status | Notes |
|----------|--------|-------|
| **Windows x64** | ✅ Supported | Standard installation |
| **macOS x64** | ✅ Supported | Intel Macs |
| **macOS ARM64** | ✅ Supported | M1/M2/M3/M4 with native ARM wheels |
| **Linux x64** | ✅ Supported | Intel/AMD processors |
| **Linux ARM64** | ⚠️ Experimental | Raspberry Pi, AWS Graviton — see [ARM_INSTALLATION.md](ARM_INSTALLATION.md) |
| **Linux ARMv7** | ❌ Limited | Requires building from source — see [ARM_INSTALLATION.md](ARM_INSTALLATION.md) |

**ARM Users**: If you get "Illegal instruction" errors, see [ARM_INSTALLATION.md](ARM_INSTALLATION.md) for platform-specific setup instructions.

## Setup

```bash
conda env create -f environment.yml
conda activate fashion-bench
```

### Verify Installation

After setup, verify all dependencies are working:

```bash
python check_install.py
```

This will check PyTorch, OpenVINO, and other dependencies for compatibility with your platform.

**Common Issues:**

- **"Illegal instruction" on Raspberry Pi/ARM**: See [ARM_INSTALLATION.md](ARM_INSTALLATION.md)
- **Import errors**: Recreate the environment with `conda env create -f environment.yml --force`
- **CUDA/GPU errors**: The project uses CPU by default; GPU support is optional

## Usage

Run the full pipeline (train → optimize → benchmark):

```bash
python -m src.main
```

### Individual stages

```bash
python -m src.main train        # Train the model only
python -m src.main optimize     # Export & quantize only
python -m src.main benchmark    # Benchmark only
```

### CLI options

| Flag | Description |
|------|-------------|
| `--epochs N` | Number of training epochs (default: 10) |
| `--devices CPU GPU` | OpenVINO devices to benchmark on (default: auto-detect) |
| `--force-train` | Force re-training even if a saved model already exists |

### Retraining

By default, training is **skipped** if a saved model already exists at `models/fashion_mnist_cnn.pth`. To retrain from scratch:

```bash
# Retrain and then run the full pipeline
python -m src.main --force-train

# Retrain only (no optimization or benchmarking)
python -m src.main train --force-train

# Retrain with a custom epoch count
python -m src.main train --force-train --epochs 20
```

## Project structure

```
src/                  Python application (model, optimization, benchmark)
models/               Trained PyTorch model (.pth)
models/openvino/      OpenVINO IRs organized by quantization (fp32/, int8/, int4/)
benchmark_logs/       JSON benchmark results
html/                 HTML dashboard that renders benchmark results
data/                 Fashion MNIST dataset (auto-downloaded)
```

## Quantization variants

- **FP32** — baseline OpenVINO IR converted from PyTorch
- **INT8** — NNCF post-training quantization
- **INT4** — NNCF weight compression

## Benchmarks

Each benchmark run measures:

- **Accuracy** — percent of correctly identified items in the test dataset
- **Average latency** — mean inference time per image
- **Total time** — wall-clock time to process the entire test set
