"""Underwater image enhancement.

Implements:
* Dark Channel Prior (He et al., 2009) dehazing with guided-filter transmission refinement.
* CLAHE contrast enhancement on the L channel of LAB.
* Denoising (fast bilateral for real-time use, or OpenCV non-local means).
* Optional gray-world white balance to counter the blue/green colour cast.

All functions take and return ``uint8`` BGR images (OpenCV convention).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# Dark channel prior
# ----------------------------------------------------------------------------
def dark_channel(img: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """Dark channel: per-pixel min over colour channels followed by a min filter.

    Args:
        img: float image in [0, 1], shape (H, W, 3).
        patch_size: Size of the square min-filter window.
    """
    min_rgb = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_rgb, kernel)


def estimate_atmospheric_light(img: np.ndarray, dark: np.ndarray, top_percent: float = 0.001) -> np.ndarray:
    """Estimate the airlight A as the mean colour of the brightest dark-channel pixels."""
    h, w = dark.shape
    n = max(1, int(h * w * top_percent))
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    return img.reshape(-1, 3)[idx].mean(axis=0)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Fast guided filter (He et al., 2010) using box filters. ``guide`` is grayscale float."""
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, ksize)
    corr_ip = cv2.boxFilter(guide * src, cv2.CV_32F, ksize)
    corr_ii = cv2.boxFilter(guide * guide, cv2.CV_32F, ksize)
    var_i = corr_ii - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)
    return mean_a * guide + mean_b


def dehaze(
    img_bgr: np.ndarray,
    patch_size: int = 15,
    omega: float = 0.95,
    t0: float = 0.1,
    top_percent: float = 0.001,
    guided_radius: int = 40,
    guided_eps: float = 1e-3,
) -> np.ndarray:
    """Remove haze / backscatter using the Dark Channel Prior.

    Returns a uint8 BGR image of the same shape.
    """
    _validate(img_bgr)
    img = img_bgr.astype(np.float32) / 255.0
    dark = dark_channel(img, patch_size)
    airlight = np.maximum(estimate_atmospheric_light(img, dark, top_percent), 1e-3)
    transmission = 1.0 - omega * dark_channel(img / airlight, patch_size)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    radius = max(1, min(guided_radius, min(img.shape[:2]) // 4))
    transmission = guided_filter(gray, transmission.astype(np.float32), radius, guided_eps)
    transmission = np.clip(transmission, t0, 1.0)[..., None]
    recovered = (img - airlight) / transmission + airlight
    return (np.clip(recovered, 0.0, 1.0) * 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Contrast, denoise, colour
# ----------------------------------------------------------------------------
def apply_clahe(img_bgr: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalisation on the LAB lightness channel."""
    _validate(img_bgr)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid), int(tile_grid)))
    return cv2.cvtColor(cv2.merge((clahe.apply(l_chan), a_chan, b_chan)), cv2.COLOR_LAB2BGR)


def denoise(img_bgr: np.ndarray, h: int = 5, fast: bool = True) -> np.ndarray:
    """Denoise: bilateral filter when ``fast`` else non-local means (slower, stronger)."""
    _validate(img_bgr)
    if fast:
        return cv2.bilateralFilter(img_bgr, d=5, sigmaColor=10 * h, sigmaSpace=5)
    return cv2.fastNlMeansDenoisingColored(img_bgr, None, h, h, 7, 21)


def gray_world_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    """Gray-world assumption: scale each channel so their means become equal."""
    _validate(img_bgr)
    img = img_bgr.astype(np.float32)
    means = img.reshape(-1, 3).mean(axis=0)
    gain = means.mean() / np.maximum(means, 1e-6)
    return np.clip(img * gain, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------
@dataclass
class EnhanceOptions:
    """Which enhancement steps to run (order: white balance -> dehaze -> CLAHE -> denoise)."""

    dehaze: bool = True
    clahe: bool = True
    denoise: bool = False
    white_balance: bool = False
    params: dict[str, Any] = field(default_factory=dict)


def enhance(img_bgr: np.ndarray, options: EnhanceOptions | None = None, **overrides: Any) -> np.ndarray:
    """Run the configured enhancement pipeline.

    Args:
        img_bgr: uint8 BGR image.
        options: Step toggles plus the ``preprocessing`` config section in ``options.params``.
        **overrides: Shortcut toggles, e.g. ``enhance(img, dehaze=False)``.
    """
    _validate(img_bgr)
    opts = options or EnhanceOptions()
    for key, value in overrides.items():
        setattr(opts, key, value)
    p = opts.params or {}
    out = img_bgr
    if opts.white_balance:
        out = gray_world_white_balance(out)
    if opts.dehaze:
        out = dehaze(out, **p.get("dehaze", {}))
    if opts.clahe:
        out = apply_clahe(out, **p.get("clahe", {}))
    if opts.denoise:
        out = denoise(out, **p.get("denoise", {}))
    return out


def options_from_config(cfg: dict[str, Any], **toggles: bool) -> EnhanceOptions:
    """Build :class:`EnhanceOptions` from the ``preprocessing`` config section."""
    pre = cfg.get("preprocessing", {})
    opts = EnhanceOptions(white_balance=bool(pre.get("white_balance", False)), params=pre)
    for key, value in toggles.items():
        setattr(opts, key, value)
    return opts


def _validate(img: np.ndarray) -> None:
    if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Expected an HxWx3 BGR image, got {getattr(img, 'shape', type(img))}")
    if img.dtype != np.uint8:
        raise ValueError(f"Expected uint8 image, got {img.dtype}")
    if img.shape[0] < 2 or img.shape[1] < 2:
        raise ValueError("Image too small for enhancement")
