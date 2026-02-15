"""Shared configuration constants for the Fashion MNIST Performance Evaluator."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = ROOT_DIR / "data"
MODELS_DIR: Path = ROOT_DIR / "models"
OPENVINO_DIR: Path = MODELS_DIR / "openvino"
BENCHMARK_LOGS_DIR: Path = ROOT_DIR / "benchmark_logs"
HTML_DIR: Path = ROOT_DIR / "html"
PYTORCH_MODEL_PATH: Path = MODELS_DIR / "fashion_mnist_cnn.pth"

# ── Training hyper-parameters ─────────────────────────────────────────────────
BATCH_SIZE: int = 64
LEARNING_RATE: float = 1e-3
EPOCHS: int = 10

# ── Fashion-MNIST class labels ────────────────────────────────────────────────
CLASS_NAMES: list[str] = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# ── Image properties ──────────────────────────────────────────────────────────
INPUT_SHAPE: tuple[int, int, int, int] = (1, 1, 28, 28)  # NCHW
NUM_CLASSES: int = 10

# ── Quantization variants ─────────────────────────────────────────────────────
QUANTIZATION_VARIANTS: list[str] = ["fp32", "int8", "int4"]
