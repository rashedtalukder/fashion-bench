#!/usr/bin/env python3
"""Pre-flight check for platform compatibility.

Run this before attempting to use the Fashion MNIST Performance Evaluator
to verify that all dependencies are properly installed for your platform.

Usage:
    python check_install.py
"""

from __future__ import annotations

import platform
import subprocess
import sys


def print_header(text: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def print_status(component: str, status: str, details: str = "") -> None:
    """Print component status with colored indicator."""
    symbols = {"✓": "✓", "✗": "✗", "⚠": "⚠"}
    symbol = symbols.get(status, status)
    print(f"{symbol} {component:20s} {details}")


def main() -> int:
    """Run platform compatibility checks and return exit code."""
    print_header("Fashion MNIST Benchmarker — Installation Check")
    
    # System info
    machine = platform.machine()
    system = platform.system()
    py_version = platform.python_version()
    
    print(f"\nPlatform: {system} {machine}")
    print(f"Python:   {py_version}")
    
    is_arm = any(arch in machine.lower() for arch in ["arm", "aarch64"])
    
    if is_arm:
        print("\n⚠️  ARM platform detected!")
        print("   Standard PyPI packages may not work.")
        print("   See ARM_INSTALLATION.md for guidance.")
    
    print_header("Checking Dependencies")
    
    all_ok = True
    
    # Check Python version
    version_tuple = tuple(map(int, py_version.split(".")[:2]))
    if version_tuple >= (3, 9):
        print_status("Python version", "✓", f"{py_version} (>= 3.9)")
    else:
        print_status("Python version", "✗", f"{py_version} (need >= 3.9)")
        all_ok = False
    
    # Check PyTorch
    try:
        import torch
        print_status("PyTorch", "✓", f"version {torch.__version__}")
        
        # Try a simple tensor operation
        try:
            x = torch.randn(2, 2)
            _ = x.sum()
            print_status("PyTorch ops", "✓", "basic operations work")
        except Exception as e:
            print_status("PyTorch ops", "✗", f"error: {e}")
            all_ok = False
    except ImportError:
        print_status("PyTorch", "✗", "not installed")
        all_ok = False
    except Exception as e:
        print_status("PyTorch", "✗", f"import failed: {e}")
        all_ok = False
    
    # Check torchvision
    try:
        import torchvision
        print_status("torchvision", "✓", f"version {torchvision.__version__}")
    except ImportError:
        print_status("torchvision", "✗", "not installed")
        all_ok = False
    except Exception as e:
        print_status("torchvision", "✗", f"import failed: {e}")
        all_ok = False
    
    # Check OpenVINO
    try:
        import openvino as ov
        print_status("OpenVINO", "✓", f"version {ov.__version__}")
        
        # Try loading core
        try:
            core = ov.Core()
            devices = core.available_devices()
            print_status("OpenVINO devices", "✓", f"{', '.join(devices)}")
        except Exception as e:
            print_status("OpenVINO devices", "⚠", f"error: {e}")
    except ImportError:
        print_status("OpenVINO", "✗", "not installed")
        if is_arm:
            print("           ⚠️  OpenVINO has limited ARM support")
            print("           Consider using ONNX Runtime instead")
        all_ok = False
    except Exception as e:
        print_status("OpenVINO", "✗", f"import failed: {e}")
        all_ok = False
    
    # Check NNCF
    try:
        import nncf
        print_status("NNCF", "✓", f"version {nncf.__version__}")
    except ImportError:
        print_status("NNCF", "✗", "not installed")
        all_ok = False
    except Exception as e:
        print_status("NNCF", "⚠", f"import warning: {e}")
    
    # Check NumPy
    try:
        import numpy as np
        print_status("NumPy", "✓", f"version {np.__version__}")
    except ImportError:
        print_status("NumPy", "✗", "not installed")
        all_ok = False
    
    # Check ONNX
    try:
        import onnx
        print_status("ONNX", "✓", f"version {onnx.__version__}")
    except ImportError:
        print_status("ONNX", "⚠", "not installed (optional)")
    
    # Summary
    print_header("Summary")
    
    if all_ok:
        print("\n✅ All required dependencies are installed and working!")
        print("\nYou can now run:")
        print("    python -m src.main")
        return 0
    else:
        print("\n❌ Some dependencies are missing or not working properly.")
        print("\nTo fix:")
        if is_arm:
            print("    1. See ARM_INSTALLATION.md for platform-specific instructions")
            print("    2. For Raspberry Pi, consider using ONNX Runtime instead")
        else:
            print("    1. Install/recreate conda environment:")
            print("       conda env create -f environment.yml --force")
            print("    2. Activate the environment:")
            print("       conda activate fashion-bench")
            print("    3. Run this check again:")
            print("       python check_install.py")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
