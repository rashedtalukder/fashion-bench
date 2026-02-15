"""Benchmark OpenVINO IR models on the Fashion-MNIST test set and write JSON logs."""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import openvino as ov

from src.config import (
    BENCHMARK_LOGS_DIR,
    CLASS_NAMES,
    NUM_CLASSES,
    OPENVINO_DIR,
    QUANTIZATION_VARIANTS,
)
from src.model import get_dataloaders


# ── Types ──────────────────────────────────────────────────────────────────────

BenchmarkResult = dict[str, object]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _system_info() -> dict[str, str]:
    """Collect basic system information for the benchmark log."""
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "hostname": platform.node(),
    }


def _available_ov_devices() -> list[str]:
    """Return the list of devices the local OpenVINO runtime can target."""
    core: ov.Core = ov.Core()
    return core.available_devices


# ── Core benchmark routine ─────────────────────────────────────────────────────


def benchmark_model(
    xml_path: Path,
    device: str = "CPU",
    variant_name: str = "fp32",
) -> BenchmarkResult:
    """Run the full test-set through an OpenVINO model and collect metrics.

    Metrics collected
    -----------------
    * **accuracy** – percentage of correctly classified images.
    * **avg_latency_ms** – average per-image inference latency in milliseconds.
    * **total_time_s** – wall-clock time for the entire test-set evaluation.
    * **per_class_accuracy** – accuracy broken down by Fashion-MNIST class.

    Parameters
    ----------
    xml_path : Path
        Path to the OpenVINO ``.xml`` file.
    device : str
        OpenVINO device string, e.g. ``"CPU"`` or ``"GPU"``.
    variant_name : str
        Label for this quantisation variant (``"fp32"``, ``"int8"``, ``"int4"``).

    Returns
    -------
    BenchmarkResult
        A JSON-serialisable dict with all metrics and metadata.
    """
    core: ov.Core = ov.Core()
    compiled_model: ov.CompiledModel = core.compile_model(str(xml_path), device)
    infer_request: ov.InferRequest = compiled_model.create_infer_request()

    _, test_loader = get_dataloaders(batch_size=1)

    correct: int = 0
    total: int = 0
    latencies: list[float] = []
    per_class_correct: list[int] = [0] * NUM_CLASSES
    per_class_total: list[int] = [0] * NUM_CLASSES

    overall_start: float = time.perf_counter()

    for images, labels in test_loader:
        img_np: np.ndarray = images.numpy()
        label: int = int(labels.item())

        t0: float = time.perf_counter()
        infer_request.infer({0: img_np})
        t1: float = time.perf_counter()

        output: np.ndarray = infer_request.get_output_tensor(0).data
        predicted: int = int(np.argmax(output, axis=1)[0])

        latencies.append((t1 - t0) * 1000.0)  # ms
        total += 1
        per_class_total[label] += 1
        if predicted == label:
            correct += 1
            per_class_correct[label] += 1

    overall_end: float = time.perf_counter()

    accuracy: float = 100.0 * correct / total if total > 0 else 0.0
    avg_latency_ms: float = float(np.mean(latencies)) if latencies else 0.0
    total_time_s: float = overall_end - overall_start

    per_class_accuracy: dict[str, float] = {}
    for idx in range(NUM_CLASSES):
        cls_name: str = CLASS_NAMES[idx]
        if per_class_total[idx] > 0:
            per_class_accuracy[cls_name] = round(
                100.0 * per_class_correct[idx] / per_class_total[idx], 2
            )
        else:
            per_class_accuracy[cls_name] = 0.0

    result: BenchmarkResult = {
        "variant": variant_name,
        "device": device,
        "model_path": str(xml_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "system_info": _system_info(),
        "total_images": total,
        "accuracy_pct": round(accuracy, 2),
        "avg_latency_ms": round(avg_latency_ms, 4),
        "total_time_s": round(total_time_s, 3),
        "per_class_accuracy": per_class_accuracy,
    }

    return result


# ── Orchestrator ───────────────────────────────────────────────────────────────


def run_all_benchmarks(
    devices: Optional[list[str]] = None,
) -> list[BenchmarkResult]:
    """Benchmark every quantisation variant on every available device.

    Results are written to ``benchmark_logs/<variant>_<device>.json``.
    """
    if devices is None:
        devices = _available_ov_devices()
        # Filter to CPU / GPU only (skip e.g. GNA, VPU unless present)
        devices = [d for d in devices if d in ("CPU", "GPU")]
        if not devices:
            devices = ["CPU"]

    BENCHMARK_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[BenchmarkResult] = []

    for variant in QUANTIZATION_VARIANTS:
        xml_path: Path = OPENVINO_DIR / variant / "fashion_mnist.xml"
        if not xml_path.exists():
            print(f"[bench] Skipping {variant} — IR not found at {xml_path}")
            continue

        for device in devices:
            print(f"\n[bench] Running {variant} on {device} …")
            try:
                result: BenchmarkResult = benchmark_model(
                    xml_path, device=device, variant_name=variant
                )
            except RuntimeError as exc:
                print(f"[bench] {variant}/{device} failed: {exc}")
                continue

            # Pretty-print summary
            print(
                f"  accuracy:    {result['accuracy_pct']:.2f}%\n"
                f"  avg latency: {result['avg_latency_ms']:.4f} ms\n"
                f"  total time:  {result['total_time_s']:.3f} s"
            )

            # Write individual JSON log
            log_name: str = f"{variant}_{device.lower()}.json"
            log_path: Path = BENCHMARK_LOGS_DIR / log_name
            with open(log_path, "w", encoding="utf-8") as fh:
                json.dump(result, fh, indent=2)
            print(f"  log → {log_path}")

            all_results.append(result)

    # Also write a combined log
    combined_path: Path = BENCHMARK_LOGS_DIR / "all_benchmarks.json"
    with open(combined_path, "w", encoding="utf-8") as fh:
        json.dump(all_results, fh, indent=2)
    print(f"\n[bench] Combined log → {combined_path}")

    return all_results
