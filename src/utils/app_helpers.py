"""Framework-agnostic helpers used by the Streamlit / Gradio UIs (unit-tested)."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import pandas as pd

MAX_IMAGE_SIDE = 4096


def decode_image(data: bytes) -> np.ndarray:
    """Decode uploaded bytes to a BGR uint8 image; raises ``ValueError`` on bad input."""
    if not data:
        raise ValueError("Empty file")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("File is not a readable image (supported: JPG, PNG, BMP, WEBP)")
    return limit_size(img)


def limit_size(img: np.ndarray, max_side: int = MAX_IMAGE_SIDE) -> np.ndarray:
    """Downscale very large images so inference/enhancement stays responsive."""
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    f = max_side / s
    return cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_png(img_bgr: np.ndarray) -> bytes:
    """BGR image -> PNG bytes (for download buttons)."""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Could not encode image")
    return buf.tobytes()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def mean_confidence(df: pd.DataFrame) -> float | None:
    """Mean detection confidence, or None when there are no detections."""
    return None if df.empty else round(float(df["confidence"].mean()), 4)


def validate_coordinates(lat: float | None, lon: float | None) -> tuple[float | None, float | None]:
    """Return (lat, lon) if valid, raising ``ValueError`` on out-of-range values."""
    if lat is None and lon is None:
        return None, None
    if lat is None or lon is None:
        raise ValueError("Provide both latitude and longitude (or neither)")
    if not -90 <= float(lat) <= 90:
        raise ValueError("Latitude must be between -90 and 90")
    if not -180 <= float(lon) <= 180:
        raise ValueError("Longitude must be between -180 and 180")
    return float(lat), float(lon)


def validate_water_inputs(values: Mapping[str, Any], guidelines: Mapping[str, Mapping[str, Any]]) -> tuple[dict[str, float | None], list[str]]:
    """Coerce form values to floats and collect human-readable problems.

    Negative concentrations and pH outside 0-14 are errors; ``None`` means "not measured".
    """
    clean: dict[str, float | None] = {}
    errors: list[str] = []
    for k, v in values.items():
        label = guidelines.get(k, {}).get("label", k)
        if v is None or (isinstance(v, str) and not v.strip()):
            clean[k] = None
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            errors.append(f"{label}: '{v}' is not a number")
            continue
        if np.isnan(f) or np.isinf(f):
            clean[k] = None
            continue
        if k == "ph" and not 0 <= f <= 14:
            errors.append("pH must be between 0 and 14")
        elif f < 0:
            errors.append(f"{label} cannot be negative")
        clean[k] = f
    return clean, errors


def read_water_csv(data: bytes, required: list[str]) -> pd.DataFrame:
    """Parse an uploaded water CSV; at least one required column must be present."""
    try:
        df = pd.read_csv(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not parse CSV: {exc}") from exc
    if df.empty:
        raise ValueError("CSV has no rows")
    present = [c for c in required if c in df.columns]
    if not present:
        raise ValueError(f"CSV must contain at least one of: {', '.join(required)}")
    for c in present:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def load_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON report file if present (used by the Model Performance page)."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def side_by_side(before: np.ndarray, after: np.ndarray, gap: int = 6) -> np.ndarray:
    """Concatenate two BGR images horizontally (resizing ``after`` to ``before``'s height)."""
    h = before.shape[0]
    if after.shape[0] != h:
        after = cv2.resize(after, (int(after.shape[1] * h / after.shape[0]), h))
    spacer = np.full((h, gap, 3), 255, np.uint8)
    return np.hstack([before, spacer, after])
