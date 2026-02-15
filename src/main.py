#!/usr/bin/env python3
"""Fashion MNIST Performance Evaluator — CLI entry point.

Usage
-----
    # Run the full pipeline: train → optimise → benchmark
    python -m src.main

    # Individual stages
    python -m src.main train
    python -m src.main optimize
    python -m src.main benchmark
    python -m src.main all          # same as no argument
"""

from __future__ import annotations

import argparse

from src.model import train_model, PYTORCH_MODEL_PATH
from src.optimize import optimize_all
from src.benchmark import run_all_benchmarks


def _parse_args() -> argparse.Namespace:
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="Fashion MNIST Performance Evaluator",
    )
    parser.add_argument(
        "stage",
        nargs="?",
        default="all",
        choices=["train", "optimize", "benchmark", "all"],
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--devices",
        nargs="*",
        default=None,
        help="OpenVINO devices to benchmark on, e.g. CPU GPU (default: auto-detect)",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        default=False,
        help="Force re-training even if a saved model already exists",
    )
    return parser.parse_args()


def main() -> None:
    args: argparse.Namespace = _parse_args()

    if args.stage in ("train", "all"):
        print("\n" + "=" * 60)
        print("  STAGE 1: Training")
        print("=" * 60)
        train_model(epochs=args.epochs, force=args.force_train)

    if args.stage in ("optimize", "all"):
        print("\n" + "=" * 60)
        print("  STAGE 2: OpenVINO Optimisation")
        print("=" * 60)
        if not PYTORCH_MODEL_PATH.exists():
            print("[main] No trained model found — training first…")
            train_model(epochs=args.epochs, force=True)
        optimize_all()

    if args.stage in ("benchmark", "all"):
        print("\n" + "=" * 60)
        print("  STAGE 3: Benchmarking")
        print("=" * 60)
        results = run_all_benchmarks(devices=args.devices)
        print(f"\n[main] Completed {len(results)} benchmark run(s).")

    print("\n✓ Done.")


if __name__ == "__main__":
    main()
