"""
Benchmark every OpenVINO model variant (FP32 / INT8) on every available
device (CPU, GPU, MULTI:CPU,GPU) and log latency + FPS results.

Usage (from project root):
    python src/ov_benchmark.py                     # 30 s per config (default)
    python src/ov_benchmark.py --duration 60        # 60 s per config
    python src/ov_benchmark.py --duration 10 --csv  # also write a CSV summary
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import openvino as ov

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "openvino"
LOG_DIR = PROJECT_ROOT / "benchmark_logs"

# Model configurations: (label, relative path to .xml inside MODELS_DIR)
MODEL_CONFIGS = [
    ("FP32", "fp32/model.xml"),
    ("INT8", "int8/model.xml"),
]

# Device configurations: (friendly name, OpenVINO device string)
DEVICE_CONFIGS = [
    ("CPU", "CPU"),
    ("GPU", "GPU"),
    ("CPU+GPU", "MULTI:CPU,GPU"),
]

# Input shape for FashionMNIST
INPUT_SHAPE = (1, 1, 28, 28)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _device_available(core: ov.Core, device_id: str) -> bool:
    """Return True if *device_id* (or all sub-devices for MULTI) is present."""
    available = core.available_devices
    if device_id.startswith("MULTI:"):
        return all(d in available for d in device_id.split(":")[1].split(","))
    return device_id in available


def benchmark_model(
    model_path: Path,
    device_id: str,
    duration_sec: int,
    core: ov.Core,
) -> dict | None:
    """Run synchronous inference for *duration_sec* and return statistics."""

    if not _device_available(core, device_id):
        return None

    # Read + compile
    model = core.read_model(str(model_path))
    compiled = core.compile_model(model, device_id)
    infer_req = compiled.create_infer_request()

    # Synthetic input
    input_data = np.random.randn(*INPUT_SHAPE).astype(np.float32)
    input_tensor = ov.Tensor(input_data)

    # Warm-up
    for _ in range(20):
        infer_req.infer({0: input_tensor})

    # Timed loop
    latencies: list[float] = []
    t_start = time.perf_counter()

    while (time.perf_counter() - t_start) < duration_sec:
        t0 = time.perf_counter()
        infer_req.infer({0: input_tensor})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    elapsed = time.perf_counter() - t_start
    n_frames = len(latencies)

    return {
        "total_frames": n_frames,
        "duration_sec": round(elapsed, 2),
        "fps": round(n_frames / elapsed, 2),
        "avg_latency_ms": round(float(np.mean(latencies)), 4),
        "min_latency_ms": round(float(np.min(latencies)), 4),
        "max_latency_ms": round(float(np.max(latencies)), 4),
        "median_latency_ms": round(float(np.median(latencies)), 4),
        "p95_latency_ms": round(float(np.percentile(latencies, 95)), 4),
        "p99_latency_ms": round(float(np.percentile(latencies, 99)), 4),
        "std_latency_ms": round(float(np.std(latencies)), 4),
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
HEADER_FMT = "{:<25s} {:>10s} {:>12s} {:>12s} {:>12s} {:>12s}"
ROW_FMT = "{:<25s} {:>10.2f} {:>12.4f} {:>12.4f} {:>12.4f} {:>12.4f}"


def _print_table(results: list[dict]):
    hdr = HEADER_FMT.format("Config", "FPS", "Avg (ms)", "Median (ms)", "P95 (ms)", "P99 (ms)")
    sep = "=" * len(hdr)
    print(f"\n{sep}")
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            ROW_FMT.format(
                r["config_name"],
                r["fps"],
                r["avg_latency_ms"],
                r["median_latency_ms"],
                r["p95_latency_ms"],
                r["p99_latency_ms"],
            )
        )
    print(sep)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def _save_json(results: list[dict], meta: dict, path: Path):
    payload = {**meta, "results": results}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _save_txt(results: list[dict], meta: dict, path: Path):
    with open(path, "w") as f:
        f.write("OpenVINO FashionMNIST Benchmark Results\n")
        f.write(f"Date:              {meta['timestamp']}\n")
        f.write(f"OpenVINO version:  {meta['openvino_version']}\n")
        f.write(f"Available devices: {meta['available_devices']}\n")
        f.write(f"Duration / test:   {meta['duration_per_test_sec']}s\n")
        f.write("=" * 80 + "\n\n")

        for r in results:
            f.write(f"Configuration: {r['config_name']}\n")
            f.write(f"  Precision:     {r['precision']}\n")
            f.write(f"  Device:        {r['device']} ({r['device_id']})\n")
            f.write(f"  Model Path:    {r['model_path']}\n")
            f.write(f"  Total Frames:  {r['total_frames']}\n")
            f.write(f"  Duration:      {r['duration_sec']}s\n")
            f.write(f"  FPS:           {r['fps']:.2f}\n")
            f.write(f"  Avg Latency:   {r['avg_latency_ms']:.4f} ms\n")
            f.write(f"  Min Latency:   {r['min_latency_ms']:.4f} ms\n")
            f.write(f"  Max Latency:   {r['max_latency_ms']:.4f} ms\n")
            f.write(f"  Median:        {r['median_latency_ms']:.4f} ms\n")
            f.write(f"  P95 Latency:   {r['p95_latency_ms']:.4f} ms\n")
            f.write(f"  P99 Latency:   {r['p99_latency_ms']:.4f} ms\n")
            f.write(f"  Std Dev:       {r['std_latency_ms']:.4f} ms\n\n")


def _save_csv(results: list[dict], path: Path):
    if not results:
        return
    fieldnames = [
        "config_name", "precision", "device", "device_id",
        "total_frames", "duration_sec", "fps",
        "avg_latency_ms", "min_latency_ms", "max_latency_ms",
        "median_latency_ms", "p95_latency_ms", "p99_latency_ms", "std_latency_ms",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark OpenVINO FashionMNIST models (FP32 / INT8, CPU / GPU)"
    )
    parser.add_argument(
        "--duration", type=int, default=30,
        help="Seconds to run each benchmark configuration (default: 30)",
    )
    parser.add_argument(
        "--csv", action="store_true",
        help="Also save a CSV summary alongside JSON and TXT logs",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    core = ov.Core()
    available = core.available_devices
    print(f"OpenVINO {ov.__version__}")
    print(f"Available devices: {available}")
    print(f"Benchmark duration per config: {args.duration}s\n")

    timestamp = datetime.now()
    ts_str = timestamp.strftime("%Y%m%d_%H%M%S")

    all_results: list[dict] = []

    for precision, rel_path in MODEL_CONFIGS:
        model_path = MODELS_DIR / rel_path
        if not model_path.exists():
            print(f"  [SKIP] {precision} model not found ({model_path})")
            print("         Run  python src/ov_convert.py  first.\n")
            continue

        for device_name, device_id in DEVICE_CONFIGS:
            config_name = f"{precision} / {device_name}"
            print(f"  [{config_name}]")

            if not _device_available(core, device_id):
                print(f"    Skipping — device '{device_id}' not available\n")
                continue

            result = benchmark_model(model_path, device_id, args.duration, core)

            if result is None:
                print("    No results (device unavailable)\n")
                continue

            result.update({
                "config_name": config_name,
                "precision": precision,
                "device": device_name,
                "device_id": device_id,
                "model_path": str(model_path),
            })
            all_results.append(result)

            print(
                f"    FPS: {result['fps']:.2f}  |  "
                f"Avg: {result['avg_latency_ms']:.4f} ms  |  "
                f"P95: {result['p95_latency_ms']:.4f} ms"
            )
            print()

    if not all_results:
        print("No benchmark results were collected.")
        sys.exit(1)

    # Summary table
    _print_table(all_results)

    # Metadata shared across log files
    meta = {
        "timestamp": timestamp.isoformat(),
        "openvino_version": ov.__version__,
        "available_devices": available,
        "duration_per_test_sec": args.duration,
    }

    # Save logs
    json_path = LOG_DIR / f"benchmark_{ts_str}.json"
    txt_path = LOG_DIR / f"benchmark_{ts_str}.txt"
    _save_json(all_results, meta, json_path)
    _save_txt(all_results, meta, txt_path)
    print(f"\nResults saved:")
    print(f"  JSON → {json_path}")
    print(f"  TXT  → {txt_path}")

    if args.csv:
        csv_path = LOG_DIR / f"benchmark_{ts_str}.csv"
        _save_csv(all_results, csv_path)
        print(f"  CSV  → {csv_path}")


if __name__ == "__main__":
    main()
