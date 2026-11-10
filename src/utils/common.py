"""Shared helpers: project paths, config loading, logging, seeding and device selection."""
from __future__ import annotations

import copy
import logging
import os
import platform
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG: Path = PROJECT_ROOT / "config.yaml"

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str, level: str | int | None = None) -> logging.Logger:
    """Return a module logger with a consistent format.

    Args:
        name: Logger name (usually ``__name__``).
        level: Optional log level; defaults to ``LOG_LEVEL`` env var or INFO.
    """
    root = logging.getLogger()
    if not root.handlers:
        logging.basicConfig(format=_LOG_FORMAT, datefmt="%H:%M:%S")
    logger = logging.getLogger(name)
    logger.setLevel(level or os.environ.get("LOG_LEVEL", "INFO"))
    return logger


@lru_cache(maxsize=4)
def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load ``config.yaml`` (cached) and return a deep copy so callers may mutate it."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    return copy.deepcopy(_load_yaml(str(cfg_path.resolve())))


def resolve_path(p: str | Path) -> Path:
    """Resolve a config path relative to the project root (absolute paths pass through)."""
    path = Path(p)
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def ensure_dir(p: str | Path) -> Path:
    """Create a directory (and parents) if needed and return it as a Path."""
    path = resolve_path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and (if installed) torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass


def get_device() -> int | str:
    """Return ``0`` (first CUDA GPU) when available, else ``"cpu"``."""
    try:
        import torch

        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


def describe_device(device: int | str) -> str:
    """Human-readable description of a device id (GPU name + VRAM, or CPU)."""
    if device == "cpu":
        return f"CPU ({platform.processor() or platform.machine()})"
    import torch

    props = torch.cuda.get_device_properties(int(device))
    return f"CUDA:{device} {props.name} ({props.total_memory / 1024**3:.1f} GB VRAM)"


def safe_workers(requested: int) -> int:
    """Cap DataLoader workers on Windows (spawn-based multiprocessing is fragile/slow)."""
    if platform.system() == "Windows":
        return max(0, min(int(requested), 2))
    return max(0, int(requested))
