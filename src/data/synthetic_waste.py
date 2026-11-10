"""Procedural synthetic underwater-trash dataset (YOLO format).

Used as an offline fallback so the whole pipeline runs without internet or a
downloaded dataset. Each class has a distinctive shape/colour signature; images
are degraded with underwater effects (colour attenuation, haze, marine snow,
caustics, blur) so the enhancement module has something meaningful to do.

The synthetic data is only meant for smoke-testing and demos, NOT for producing
a model that generalises to real underwater footage.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

Rng = np.random.Generator
Patch = tuple[np.ndarray, np.ndarray]  # (bgr uint8, alpha uint8)


def _canvas(h: int, w: int) -> Patch:
    return np.zeros((h, w, 3), np.uint8), np.zeros((h, w), np.uint8)


def _col(rng: Rng, base: tuple[int, int, int], jitter: int = 25) -> tuple[int, int, int]:
    return tuple(int(np.clip(c + rng.integers(-jitter, jitter + 1), 0, 255)) for c in base)  # type: ignore[return-value]


def _bottle(rng: Rng, s: int, glass: bool = False) -> Patch:
    h, w = s, max(8, s // 3)
    img, a = _canvas(h, w)
    base = (40, 90, 30) if glass else (220, 200, 150)
    if glass and rng.random() < 0.5:
        base = (20, 60, 110)  # brown glass
    col = _col(rng, base)
    neck = w // 3
    cv2.rectangle(img, (0, h // 4), (w - 1, h - 1), col, -1)
    cv2.rectangle(a, (0, h // 4), (w - 1, h - 1), 255 if glass else 200, -1)
    cv2.rectangle(img, (neck, 0), (w - neck, h // 4), col, -1)
    cv2.rectangle(a, (neck, 0), (w - neck, h // 4), 255 if glass else 200, -1)
    cap = (30, 30, 200) if not glass else (40, 40, 40)
    cv2.rectangle(img, (neck, 0), (w - neck, max(2, h // 12)), _col(rng, cap), -1)
    cv2.rectangle(a, (neck, 0), (w - neck, max(2, h // 12)), 255, -1)
    cv2.line(img, (w // 4, h // 3), (w // 4, h - 4), (255, 255, 255), max(1, w // 10))  # highlight
    if not glass:
        cv2.rectangle(img, (0, h // 2), (w - 1, h // 2 + h // 8), _col(rng, (60, 160, 230)), -1)  # label
    return img, a


def _bag(rng: Rng, s: int) -> Patch:
    img, a = _canvas(s, s)
    n = 10
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    rad = rng.uniform(0.3, 0.5, n) * s
    pts = np.stack([s / 2 + rad * np.cos(ang), s / 2 + rad * np.sin(ang)], 1).astype(np.int32)
    cv2.fillPoly(img, [pts], _col(rng, (235, 235, 230), 15))
    cv2.fillPoly(a, [pts], 150)
    for _ in range(4):
        p1, p2 = rng.integers(s // 4, 3 * s // 4, 2), rng.integers(s // 4, 3 * s // 4, 2)
        cv2.line(img, tuple(int(v) for v in p1), tuple(int(v) for v in p2), (200, 200, 200), 1)
    return img, a


def _can(rng: Rng, s: int) -> Patch:
    h, w = s, max(8, int(s * 0.55))
    img, a = _canvas(h, w)
    col = _col(rng, [(40, 40, 200), (200, 200, 200), (40, 160, 40), (200, 80, 20)][rng.integers(4)])
    cv2.rectangle(img, (0, h // 10), (w - 1, h - h // 10), col, -1)
    cv2.rectangle(a, (0, h // 10), (w - 1, h - h // 10), 255, -1)
    for y in (h // 10, h - h // 10):
        cv2.ellipse(img, (w // 2, y), (w // 2, h // 10), 0, 0, 360, (190, 190, 190), -1)
        cv2.ellipse(a, (w // 2, y), (w // 2, h // 10), 0, 0, 360, 255, -1)
    grad = np.linspace(0.6, 1.3, w)[None, :, None]
    img = np.clip(img * grad, 0, 255).astype(np.uint8)
    return img, a


def _metal(rng: Rng, s: int) -> Patch:
    img, a = _canvas(s, s)
    n = 7
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    rad = rng.uniform(0.2, 0.5, n) * s
    pts = np.stack([s / 2 + rad * np.cos(ang), s / 2 + rad * np.sin(ang)], 1).astype(np.int32)
    cv2.fillPoly(img, [pts], _col(rng, (40, 70, 120)))  # rusty brown
    cv2.fillPoly(a, [pts], 255)
    noise = rng.integers(0, 40, (s, s, 1), dtype=np.uint8)
    img = cv2.add(img, np.repeat(noise, 3, 2))
    cv2.polylines(img, [pts], True, (150, 150, 150), 2)
    return img, a


def _net(rng: Rng, s: int) -> Patch:
    img, a = _canvas(s, s)
    col = _col(rng, [(0, 140, 255), (0, 220, 255), (60, 200, 60)][rng.integers(3)])
    step = max(4, s // 7)
    off = rng.integers(0, step)
    for i in range(off, s, step):
        cv2.line(img, (i, 0), (i + s // 6, s - 1), col, 2)
        cv2.line(a, (i, 0), (i + s // 6, s - 1), 255, 2)
        cv2.line(img, (0, i), (s - 1, i + s // 8), col, 2)
        cv2.line(a, (0, i), (s - 1, i + s // 8), 255, 2)
    return img, a


def _fishing_gear(rng: Rng, s: int) -> Patch:
    img, a = _canvas(s, s)
    col = _col(rng, (0, 0, 230))
    c1 = (s // 3, s // 3)
    cv2.circle(img, c1, s // 5, col, -1)
    cv2.circle(a, c1, s // 5, 255, -1)
    cv2.circle(img, c1, s // 5, (255, 255, 255), max(1, s // 25))
    cv2.ellipse(img, (2 * s // 3, 2 * s // 3), (s // 5, s // 5), 0, 0, 270, (180, 180, 180), max(2, s // 20))
    cv2.ellipse(a, (2 * s // 3, 2 * s // 3), (s // 5, s // 5), 0, 0, 270, 255, max(2, s // 20))
    cv2.line(img, c1, (2 * s // 3 + s // 5, 2 * s // 3), (220, 220, 220), 1)
    cv2.line(a, c1, (2 * s // 3 + s // 5, 2 * s // 3), 255, 1)
    return img, a


def _tire(rng: Rng, s: int) -> Patch:
    img, a = _canvas(s, s)
    c, r = (s // 2, s // 2), s // 2 - 1
    cv2.circle(img, c, r, _col(rng, (25, 25, 25), 10), -1)
    cv2.circle(a, c, r, 255, -1)
    cv2.circle(a, c, int(r * 0.5), 0, -1)
    cv2.circle(img, c, int(r * 0.5), (0, 0, 0), -1)
    for t in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        p1 = (int(c[0] + r * 0.8 * np.cos(t)), int(c[1] + r * 0.8 * np.sin(t)))
        p2 = (int(c[0] + r * np.cos(t)), int(c[1] + r * np.sin(t)))
        cv2.line(img, p1, p2, (70, 70, 70), 2)
    return img, a


def _cloth(rng: Rng, s: int) -> Patch:
    h, w = int(s * 0.7), s
    img, a = _canvas(h, w)
    c1, c2 = _col(rng, tuple(int(x) for x in rng.integers(40, 220, 3))), _col(rng, (240, 240, 240))
    for x in range(0, w, max(3, w // 10)):
        cv2.rectangle(img, (x, 0), (x + max(2, w // 20), h), c2, -1)
    img[np.all(img == 0, axis=2)] = c1
    ys = (h / 2 + (h / 2.3) * np.sin(np.linspace(0, 3 * np.pi, w))).astype(int)
    for x in range(w):
        top = max(0, int(abs(ys[x] - h / 2) * 0.3))
        a[top : h - top, x] = 230
    return img, a


def _paper(rng: Rng, s: int) -> Patch:
    h, w = int(s * 0.8), s
    img, a = _canvas(h, w)
    img[:] = _col(rng, (90, 150, 190))  # cardboard brown
    for y in range(0, h, 5):
        cv2.line(img, (0, y), (w - 1, y), (70, 120, 160), 1)
    a[:] = 255
    return img, a


def _wood(rng: Rng, s: int) -> Patch:
    h, w = max(8, s // 4), s
    img, a = _canvas(h, w)
    img[:] = _col(rng, (40, 80, 120))
    for y in range(1, h, 3):
        cv2.line(img, (0, y), (w - 1, y + int(rng.integers(-2, 3))), (25, 55, 85), 1)
    cv2.ellipse(a, (w // 2, h // 2), (w // 2, h // 2), 0, 0, 360, 255, -1)
    return img, a


def _cup(rng: Rng, s: int) -> Patch:
    h, w = s, int(s * 0.8)
    img, a = _canvas(h, w)
    pts = np.array([[0, 0], [w - 1, 0], [int(w * 0.8), h - 1], [int(w * 0.2), h - 1]], np.int32)
    col = _col(rng, [(245, 245, 245), (60, 60, 220), (220, 160, 60)][rng.integers(3)])
    cv2.fillPoly(img, [pts], col)
    cv2.fillPoly(a, [pts], 255)
    cv2.line(img, (0, 2), (w - 1, 2), (120, 120, 120), 3)
    return img, a


def _wrapper(rng: Rng, s: int) -> Patch:
    h, w = int(s * 0.5), s
    img, a = _canvas(h, w)
    img[:] = _col(rng, [(0, 215, 255), (255, 0, 255), (0, 255, 120)][rng.integers(3)], 30)
    for x in range(0, w, 6):
        cv2.line(img, (x, 0), (x + 3, h), (255, 255, 255), 1)
    zig = [(0, 0)] + [(x, int(rng.integers(0, h // 4))) for x in range(0, w, 6)] + [(w - 1, 0)]
    pts = np.array(zig + [(w - 1, h - 1)] + [(x, h - 1 - int(rng.integers(0, h // 4))) for x in range(w - 1, 0, -6)] + [(0, h - 1)], np.int32)
    cv2.fillPoly(a, [pts], 255)
    return img, a


def _pipe(rng: Rng, s: int) -> Patch:
    h, w = max(8, s // 5), s
    img, a = _canvas(h, w)
    base = _col(rng, (150, 150, 150), 20)
    for y in range(h):
        f = 0.5 + 0.8 * np.sin(np.pi * y / h)
        img[y, :] = np.clip(np.array(base) * f, 0, 255)
    cv2.rectangle(img, (w - max(4, w // 12), 0), (w - 1, h - 1), (90, 90, 90), -1)
    a[:] = 255
    return img, a


def _other(rng: Rng, s: int) -> Patch:
    img, a = _canvas(s, s)
    n = int(rng.integers(3, 6))
    pts = rng.integers(0, s, (n, 2)).astype(np.int32)
    hull = cv2.convexHull(pts)
    cv2.fillPoly(img, [hull], tuple(int(x) for x in rng.integers(30, 255, 3)))
    cv2.fillPoly(a, [hull], 255)
    cv2.circle(img, (s // 2, s // 2), max(2, s // 8), (0, 0, 0), -1)
    return img, a


#: class name -> drawing function. Must cover the unified class list in config.yaml.
DRAWERS: dict[str, Callable[[Rng, int], Patch]] = {
    "plastic_bottle": lambda r, s: _bottle(r, s, glass=False),
    "plastic": _bag,
    "can": _can,
    "glass": lambda r, s: _bottle(r, s, glass=True),
    "metal": _metal,
    "rope_net": _net,
    "fishing_gear": _fishing_gear,
    "rubber_tire": _tire,
    "cloth": _cloth,
    "paper_cardboard": _paper,
    "wood": _wood,
    "cup_container": _cup,
    "wrapper": _wrapper,
    "pipe": _pipe,
    "other_debris": _other,
}


def _background(rng: Rng, h: int, w: int) -> np.ndarray:
    top = np.array([rng.uniform(120, 200), rng.uniform(90, 150), rng.uniform(10, 40)])  # BGR blue-green
    bottom = np.array([rng.uniform(60, 110), rng.uniform(90, 130), rng.uniform(60, 100)])  # sandy-green
    t = np.linspace(0, 1, h)[:, None, None]
    bg = (top * (1 - t) + bottom * t) * np.ones((1, w, 1))
    yy, xx = np.mgrid[0:h, 0:w]
    caustics = np.sin(xx / rng.uniform(15, 30) + np.sin(yy / 20.0) * 2) * np.sin(yy / rng.uniform(15, 30))
    bg += (caustics[..., None] * 12) * (1 - t)
    bg += rng.normal(0, 6, (h, w, 1))
    return np.clip(bg, 0, 255).astype(np.uint8)


def _paste(dst: np.ndarray, patch: Patch, x: int, y: int) -> None:
    img, a = patch
    ph, pw = a.shape
    alpha = (a.astype(np.float32) / 255.0)[..., None]
    roi = dst[y : y + ph, x : x + pw].astype(np.float32)
    dst[y : y + ph, x : x + pw] = (img * alpha + roi * (1 - alpha)).astype(np.uint8)


def _rotate(patch: Patch, angle: float) -> Patch:
    img, a = patch
    h, w = a.shape
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos) + 1, int(h * cos + w * sin) + 1
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    img_r = cv2.warpAffine(img, m, (nw, nh), flags=cv2.INTER_LINEAR)
    a_r = cv2.warpAffine(a, m, (nw, nh), flags=cv2.INTER_LINEAR)
    ys, xs = np.nonzero(a_r > 20)
    if len(xs) == 0:
        return img, a
    return img_r[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1], a_r[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def _underwater(rng: Rng, img: np.ndarray) -> np.ndarray:
    out = img.astype(np.float32)
    out[..., 2] *= rng.uniform(0.45, 0.8)   # red attenuates fastest
    out[..., 1] *= rng.uniform(0.85, 1.0)
    haze = np.array([rng.uniform(150, 210), rng.uniform(120, 170), rng.uniform(30, 70)])
    out = out * (1 - (k := rng.uniform(0.15, 0.4))) + haze * k
    h, w = img.shape[:2]
    for _ in range(int(rng.integers(30, 120))):  # marine snow
        cv2.circle(out, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(1, 3)), (230, 230, 220), -1)
    out = cv2.GaussianBlur(out, (3, 3), rng.uniform(0.3, 1.2))
    out += rng.normal(0, 4, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def generate_image(rng: Rng, classes: list[str], width: int, height: int, max_objects: int) -> tuple[np.ndarray, list[tuple[int, float, float, float, float]]]:
    """Generate one synthetic image and its YOLO labels ``(cls, xc, yc, w, h)`` normalised."""
    img = _background(rng, height, width)
    boxes: list[tuple[int, int, int, int]] = []
    labels: list[tuple[int, float, float, float, float]] = []
    for _ in range(int(rng.integers(1, max_objects + 1))):
        cls_id = int(rng.integers(len(classes)))
        size = int(rng.integers(min(height, width) // 8, min(height, width) // 3))
        patch = _rotate(DRAWERS[classes[cls_id]](rng, size), float(rng.uniform(-60, 60)))
        ph, pw = patch[1].shape
        if ph >= height or pw >= width:
            continue
        for _attempt in range(10):
            x, y = int(rng.integers(0, width - pw)), int(rng.integers(0, height - ph))
            box = (x, y, x + pw, y + ph)
            if all(_iou(box, b) < 0.1 for b in boxes):
                break
        else:
            continue
        _paste(img, patch, x, y)
        boxes.append(box)
        labels.append((cls_id, (x + pw / 2) / width, (y + ph / 2) / height, pw / width, ph / height))
    return _underwater(rng, img), labels


def generate_dataset(out_dir: Path, classes: list[str], n_images: int = 400, width: int = 640, height: int = 480, max_objects: int = 5, seed: int = 42) -> Path:
    """Write ``n_images`` synthetic images + YOLO labels to ``out_dir/{images,labels}``.

    Also writes ``classes.txt`` so the folder is a self-describing YOLO dataset.
    """
    missing = [c for c in classes if c not in DRAWERS]
    if missing:
        raise ValueError(f"No synthetic drawer for classes: {missing}")
    rng = np.random.default_rng(seed)
    img_dir, lbl_dir = out_dir / "images", out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_images):
        img, labels = generate_image(rng, classes, width, height, max_objects)
        stem = f"synth_{i:05d}"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        (lbl_dir / f"{stem}.txt").write_text(
            "\n".join(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for c, xc, yc, w, h in labels), encoding="utf-8"
        )
    (out_dir / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
    return out_dir
