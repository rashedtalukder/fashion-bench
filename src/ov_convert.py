"""
Convert the trained FashionMNIST PyTorch model to OpenVINO Intermediate
Representation (IR) in both FP32 and INT8 (post-training quantized) formats.

Usage (from project root):
    python src/ov_convert.py
"""

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import openvino as ov
import nncf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app import create_model, get_training_data  # noqa: E402

MODEL_WEIGHTS = PROJECT_ROOT / "model.pth"
OUTPUT_DIR = PROJECT_ROOT / "models" / "openvino"


# ---------------------------------------------------------------------------
# FP32 conversion
# ---------------------------------------------------------------------------
def convert_to_fp32(model: torch.nn.Module) -> ov.Model:
    """Convert a PyTorch model to an OpenVINO FP32 IR and save it."""
    fp32_dir = OUTPUT_DIR / "fp32"
    fp32_dir.mkdir(parents=True, exist_ok=True)

    example_input = torch.randn(1, 1, 28, 28)
    ov_model = ov.convert_model(model, example_input=example_input)

    save_path = fp32_dir / "model.xml"
    ov.save_model(ov_model, str(save_path))
    print(f"[FP32] Saved to {save_path}")
    return ov_model


# ---------------------------------------------------------------------------
# INT8 quantization via NNCF post-training quantization
# ---------------------------------------------------------------------------
def _prepare_calibration_dataset(num_samples: int = 300) -> nncf.Dataset:
    """Build an NNCF calibration dataset from FashionMNIST training data."""
    train_data = get_training_data()
    subset = torch.utils.data.Subset(train_data, list(range(num_samples)))
    cal_loader = DataLoader(subset, batch_size=1, shuffle=False)

    def transform_fn(data_item):
        images, _ = data_item
        return images.numpy()

    return nncf.Dataset(cal_loader, transform_fn)


def quantize_to_int8(ov_model: ov.Model, num_calibration_samples: int = 300) -> ov.Model:
    """Quantize an OpenVINO FP32 model to INT8 and save it."""
    int8_dir = OUTPUT_DIR / "int8"
    int8_dir.mkdir(parents=True, exist_ok=True)

    print("[INT8] Preparing calibration dataset …")
    cal_dataset = _prepare_calibration_dataset(num_calibration_samples)

    print("[INT8] Running NNCF post-training quantization …")
    quantized_model = nncf.quantize(
        ov_model,
        cal_dataset,
        subset_size=num_calibration_samples,
    )

    save_path = int8_dir / "model.xml"
    ov.save_model(quantized_model, str(save_path))
    print(f"[INT8] Saved to {save_path}")
    return quantized_model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if not MODEL_WEIGHTS.exists():
        print(f"ERROR: Trained model not found at {MODEL_WEIGHTS}")
        print("Run  python src/app.py  first to train the model.")
        sys.exit(1)

    # Load the trained PyTorch model on CPU (device-agnostic export)
    model = create_model("cpu")
    model.load_state_dict(
        torch.load(str(MODEL_WEIGHTS), map_location="cpu", weights_only=True)
    )
    model.eval()
    print("Loaded trained PyTorch model\n")

    # ---- FP32 ----
    ov_model_fp32 = convert_to_fp32(model)

    # ---- INT8 ----
    quantize_to_int8(ov_model_fp32)

    print(f"\nAll models saved under: {OUTPUT_DIR}")
    print("  FP32 → models/openvino/fp32/model.xml")
    print("  INT8 → models/openvino/int8/model.xml")


if __name__ == "__main__":
    main()
