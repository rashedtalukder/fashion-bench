"""Export a trained PyTorch model to OpenVINO IR and create quantized variants."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Optional

import openvino as ov
import nncf
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import (
    INPUT_SHAPE,
    OPENVINO_DIR,
)
from src.model import FashionCNN, get_dataloaders, load_model


# ── Helpers ────────────────────────────────────────────────────────────────────


def _detect_device_tag() -> str:
    """Return a short tag describing the current hardware platform."""
    machine: str = platform.machine().lower()
    system: str = platform.system().lower()

    if "arm" in machine or "aarch64" in machine:
        if system == "darwin":
            return "arm_mac"
        return "armv7"  # Raspberry Pi or similar
    return "intel"  # x86/x64


def _ir_dir(quantisation: str) -> Path:
    """Return the directory for a given quantisation level's IR files."""
    d: Path = OPENVINO_DIR / quantisation
    d.mkdir(parents=True, exist_ok=True)
    return d


# ── ONNX export ───────────────────────────────────────────────────────────────


def export_onnx(model: FashionCNN, dst: Path) -> Path:
    """Export a PyTorch model to ONNX format.

    Returns the path to the ONNX file.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    dummy: torch.Tensor = torch.randn(*INPUT_SHAPE)
    torch.onnx.export(
        model,
        (dummy,),
        str(dst),
        input_names=["input"],
        output_names=["output"],
        opset_version=18,
        dynamo=False,
    )
    print(f"[export] ONNX model saved → {dst}")
    return dst


# ── OpenVINO IR conversion ────────────────────────────────────────────────────


def convert_to_openvino_ir(onnx_path: Path, output_dir: Path) -> Path:
    """Convert an ONNX model to OpenVINO IR (FP32).

    Returns the path to the .xml file.
    """
    ov_model: ov.Model = ov.convert_model(str(onnx_path))
    xml_path: Path = output_dir / "fashion_mnist.xml"
    ov.save_model(ov_model, str(xml_path))
    print(f"[convert] OpenVINO FP32 IR saved → {xml_path}")
    return xml_path


# ── Quantisation helpers ──────────────────────────────────────────────────────


def _calibration_data(loader: DataLoader, num_samples: int = 300) -> list[np.ndarray]:
    """Return a list of numpy arrays from the data loader for NNCF calibration."""
    samples: list[np.ndarray] = []
    for images, _ in loader:
        for img in images:
            if len(samples) >= num_samples:
                return samples
            samples.append(img.unsqueeze(0).numpy())
    return samples


def quantize_int8(fp32_xml: Path, output_dir: Path) -> Path:
    """Create an INT8-quantised IR using NNCF post-training quantization.

    Returns the path to the quantised .xml file.
    """
    core: ov.Core = ov.Core()
    ov_model: ov.Model = core.read_model(str(fp32_xml))

    _, test_loader = get_dataloaders(batch_size=1)

    calibration_dataset: nncf.Dataset = nncf.Dataset(
        _calibration_data(test_loader)
    )

    quantized_model: ov.Model = nncf.quantize(
        ov_model,
        calibration_dataset,
        model_type=nncf.ModelType.TRANSFORMER,
        preset=nncf.QuantizationPreset.PERFORMANCE,
    )

    xml_path: Path = output_dir / "fashion_mnist.xml"
    ov.save_model(quantized_model, str(xml_path))
    print(f"[quantize] INT8 IR saved → {xml_path}")
    return xml_path


def quantize_int4(fp32_xml: Path, output_dir: Path) -> Path:
    """Create an INT4-quantised IR using NNCF weight compression.

    NNCF's ``compress_weights`` supports INT4 via ``CompressWeightsMode.INT4_SYM``
    or ``INT4_ASYM``.  Returns the path to the quantised .xml file.
    """
    core: ov.Core = ov.Core()
    ov_model: ov.Model = core.read_model(str(fp32_xml))

    compressed_model: ov.Model = nncf.compress_weights(
        ov_model,
        mode=nncf.CompressWeightsMode.INT4_SYM,
    )

    xml_path: Path = output_dir / "fashion_mnist.xml"
    ov.save_model(compressed_model, str(xml_path))
    print(f"[quantize] INT4 IR saved → {xml_path}")
    return xml_path


# ── Public orchestrator ───────────────────────────────────────────────────────


def optimize_all(model_path: Optional[Path] = None) -> dict[str, Path]:
    """Run the full optimisation pipeline and return a dict of {variant: xml_path}.

    Steps:
      1. Load the trained PyTorch model
      2. Export to ONNX
      3. Convert ONNX → OpenVINO FP32 IR
      4. Quantise to INT8 and INT4
    """
    model: FashionCNN = load_model(model_path)

    # 1️⃣  ONNX
    onnx_path: Path = OPENVINO_DIR / "fashion_mnist.onnx"
    export_onnx(model, onnx_path)

    # 2️⃣  FP32 IR
    fp32_dir: Path = _ir_dir("fp32")
    fp32_xml: Path = convert_to_openvino_ir(onnx_path, fp32_dir)

    results: dict[str, Path] = {"fp32": fp32_xml}

    # 3️⃣  INT8 IR
    int8_dir: Path = _ir_dir("int8")
    int8_xml: Path = quantize_int8(fp32_xml, int8_dir)
    results["int8"] = int8_xml

    # 4️⃣  INT4 IR
    int4_dir: Path = _ir_dir("int4")
    int4_xml: Path = quantize_int4(fp32_xml, int4_dir)
    results["int4"] = int4_xml

    print("\n[optimize] All variants generated:")
    for variant, path in results.items():
        print(f"  {variant:>5}: {path}")
    return results
