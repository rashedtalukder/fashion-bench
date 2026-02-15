"""Platform compatibility checks for Fashion MNIST Performance Evaluator."""

from __future__ import annotations

import platform
import sys


def check_platform_compatibility() -> None:
    """Verify platform compatibility and provide installation guidance if needed.
    
    Raises SystemExit if the platform is not properly configured.
    """
    machine: str = platform.machine().lower()
    system: str = platform.system()
    
    # ARM platforms require special handling
    is_arm: bool = any(arch in machine for arch in ["arm", "aarch64"])
    
    if not is_arm:
        # x86/x64 platforms should work with standard pip packages
        return
    
    # For ARM platforms, try to detect if packages are properly installed
    # Note: "Illegal instruction" often manifests as SIGILL, which may not be
    # catchable in Python. We do a best-effort check here.
    print(f"[platform] Detected ARM architecture: {system} {machine}")
    print("[platform] Checking PyTorch and OpenVINO compatibility...")
    
    # Try importing torch
    try:
        import torch
        print(f"[platform] ✓ PyTorch {torch.__version__} loaded successfully")
    except Exception as e:
        error_msg: str = str(e).lower()
        if any(x in error_msg for x in ["illegal", "cannot execute", "exec format"]):
            print(f"[platform] ✗ PyTorch import failed: {e}")
            _print_arm_installation_guide(system, machine)
            sys.exit(1)
        # Re-raise other errors (missing package, etc.)
        print(f"[platform] ⚠ PyTorch import error: {e}")
        raise
    
    # Try importing openvino
    try:
        import openvino
        print(f"[platform] ✓ OpenVINO {openvino.__version__} loaded successfully")
    except Exception as e:
        error_msg: str = str(e).lower()
        if any(x in error_msg for x in ["illegal", "cannot execute", "exec format"]):
            print(f"[platform] ✗ OpenVINO import failed: {e}")
            _print_arm_installation_guide(system, machine)
            sys.exit(1)
        # For ARM, OpenVINO might not be available - warn but continue
        if is_arm:
            print(f"[platform] ⚠ OpenVINO not available: {e}")
            print("[platform] You can still train models, but optimization/benchmarking won't work.")
            print("[platform] See ARM_INSTALLATION.md for alternatives.")
        else:
            raise
    
    print("[platform] ✓ All dependencies loaded successfully\n")


def _print_arm_installation_guide(system: str, machine: str) -> None:
    """Print installation instructions for ARM platforms."""
    print("\n" + "=" * 70)
    print("  ❌ ARM Platform Detected — Incompatible Dependencies")
    print("=" * 70)
    print(f"\nPlatform: {system} {machine}")
    print("\nThe default PyTorch/OpenVINO packages from pip are compiled for x86_64")
    print("and will not work on ARM processors (Raspberry Pi, etc.).")
    print("\n" + "-" * 70)
    print("  Installation Options for ARM:")
    print("-" * 70)
    
    if "linux" in system.lower() and "arm" in machine:
        # Raspberry Pi (ARMv7/ARMv8)
        print("\n1. Install PyTorch for ARM Linux (32-bit/64-bit):")
        print("   Remove the current torch installation:")
        print("   $ pip uninstall torch torchvision")
        print()
        print("   Install ARM-compatible PyTorch:")
        print("   # For ARMv7 (32-bit Raspberry Pi OS):")
        print("   $ pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        print()
        print("   # For ARMv8 (64-bit):")
        print("   $ pip install torch torchvision")
        print()
        print("2. For OpenVINO on ARM:")
        print("   OpenVINO 2025.4+ has experimental ARM support.")
        print("   You may need to build from source:")
        print("   https://github.com/openvinotoolkit/openvino/wiki/BuildingForLinuxSystems")
        print()
        print("3. Alternative: Use ONNX Runtime for inference instead of OpenVINO")
        print("   $ pip install onnxruntime")
        print("   (Requires code modifications to use ONNX Runtime instead of OpenVINO)")
    
    elif "darwin" in system.lower():
        # Apple Silicon
        print("\n✓ Apple Silicon (M1/M2/M3) is supported!")
        print("  Make sure you're using the correct Python environment:")
        print()
        print("  $ conda create -n fashion-bench python=3.11")
        print("  $ conda activate fashion-bench")
        print("  $ pip install torch torchvision openvino nncf")
        print()
        print("  PyTorch and OpenVINO have native ARM64 support on macOS.")
    
    print("\n" + "=" * 70)
    print("  For more details, see: README.md or docs/ARM_INSTALLATION.md")
    print("=" * 70 + "\n")
