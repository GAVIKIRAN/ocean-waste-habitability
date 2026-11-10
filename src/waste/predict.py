"""Waste-detection inference: single image, folder batch, video file and live webcam.

Library use::

    det = WasteDetector.from_config(cfg)
    result = det.predict(img_bgr, conf=0.25, iou=0.45, enhance_input=True)
    result.annotated, result.detections  # BGR ndarray, pandas DataFrame

CLI::

    python -m src.waste.predict --source path/to/img.jpg|folder|video.mp4|webcam [--enhance] [--conf 0.3]
"""
from __future__ import annotations

import argparse
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np
import pandas as pd

from src.preprocessing.underwater_enhance import EnhanceOptions, enhance, options_from_config
from src.utils.common import ensure_dir, get_device, get_logger, load_config, resolve_path

log = get_logger(__name__)
DETECTION_COLUMNS = ["timestamp", "source", "frame", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class ModelNotAvailableError(RuntimeError):
    """Raised when no trained (or demo) weights can be loaded."""


@dataclass
class DetectionResult:
    """Output of a single-image prediction."""

    annotated: np.ndarray
    detections: pd.DataFrame
    input_image: np.ndarray
    inference_ms: float = 0.0
    class_counts: dict[str, int] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.detections)

    def coverage(self) -> float:
        """Fraction of the image area covered by detected boxes (union approximated by a mask)."""
        h, w = self.input_image.shape[:2]
        if self.detections.empty or h == 0 or w == 0:
            return 0.0
        mask = np.zeros((h, w), np.uint8)
        for r in self.detections.itertuples():
            mask[int(max(r.y1, 0)) : int(min(r.y2, h)), int(max(r.x1, 0)) : int(min(r.x2, w))] = 1
        return float(mask.mean())


def empty_detections() -> pd.DataFrame:
    """An empty detections DataFrame with the canonical columns."""
    return pd.DataFrame(columns=DETECTION_COLUMNS)


def boxes_to_dataframe(xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray, names: dict[int, str] | list[str], source: str = "image", frame: int = 0) -> pd.DataFrame:
    """Convert raw box arrays into the canonical detections DataFrame."""
    if len(xyxy) == 0:
        return empty_detections()
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    name_of = (lambda i: names[i]) if isinstance(names, (dict, list)) else str  # noqa: E731
    rows = [
        {
            "timestamp": ts,
            "source": source,
            "frame": frame,
            "class_id": int(c),
            "class_name": str(name_of(int(c))),
            "confidence": round(float(p), 4),
            "x1": round(float(b[0]), 1),
            "y1": round(float(b[1]), 1),
            "x2": round(float(b[2]), 1),
            "y2": round(float(b[3]), 1),
        }
        for b, p, c in zip(xyxy, conf, cls)
    ]
    return pd.DataFrame(rows, columns=DETECTION_COLUMNS)


def class_counts(df: pd.DataFrame) -> dict[str, int]:
    """``{class_name: count}`` sorted by frequency."""
    return dict(Counter(df["class_name"]).most_common()) if not df.empty else {}


class WasteDetector:
    """Thin wrapper around an Ultralytics YOLO model with optional underwater enhancement."""

    def __init__(self, weights: str | Path, cfg: dict[str, Any] | None = None, device: int | str | None = None, half: bool | None = None, is_demo: bool = False) -> None:
        from ultralytics import YOLO

        self.cfg = cfg or load_config()
        self.weights = str(weights)
        self.model = YOLO(self.weights)
        self.device = get_device() if device is None else device
        inf = self.cfg["waste"]["inference"]
        self.half = bool(inf.get("half", True) if half is None else half) and self.device != "cpu"
        self.imgsz = int(inf.get("imgsz", 640))
        self.names: dict[int, str] = dict(self.model.names)
        self.is_demo = is_demo
        self.enhance_options: EnhanceOptions = options_from_config(self.cfg)

    # ------------------------------------------------------------------ factory
    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None, allow_demo: bool = True, **kw: Any) -> "WasteDetector":
        """Load trained weights from config; optionally fall back to the COCO demo model.

        Raises:
            ModelNotAvailableError: if neither trained nor demo weights are available.
        """
        cfg = cfg or load_config()
        trained = resolve_path(cfg["waste"]["weights"])
        if trained.exists():
            return cls(trained, cfg, **kw)
        if allow_demo:
            demo = resolve_path("models") / cfg["waste"]["demo_weights"]
            try:
                if not demo.exists():
                    from ultralytics.utils.downloads import attempt_download_asset

                    attempt_download_asset(demo)
                if demo.exists():
                    log.warning("No trained waste model - using COCO demo weights %s", demo)
                    return cls(demo, cfg, is_demo=True, **kw)
            except Exception as exc:  # noqa: BLE001
                log.warning("Demo weights unavailable: %s", exc)
        raise ModelNotAvailableError(f"No waste model at {trained}. Train it first: python -m src.waste.train --mode demo")

    # --------------------------------------------------------------- inference
    def preprocess(self, img_bgr: np.ndarray, **toggles: bool) -> np.ndarray:
        """Apply underwater enhancement with config params (toggles override steps)."""
        opts = EnhanceOptions(**{**self.enhance_options.__dict__, **toggles})
        return enhance(img_bgr, opts)

    def predict(self, img_bgr: np.ndarray, conf: float | None = None, iou: float | None = None, enhance_input: bool = False, source: str = "image", frame: int = 0, enhance_toggles: dict[str, bool] | None = None, line_width: int | None = None, tiles: int = 1, augment: bool = False) -> DetectionResult:
        """Detect waste in one BGR image.

        Args:
            tiles: ``1`` = standard single pass. ``>= 2`` = high-recall mode: the full image plus a
                ``tiles x tiles`` grid of overlapping crops, each enlarged to the model input size,
                merged with class-aware NMS. Finds many more small and densely packed items, at
                roughly ``tiles**2 + 1`` times the inference cost.
            augment: test-time augmentation (flips/scales) for extra recall.
        """
        if not isinstance(img_bgr, np.ndarray) or img_bgr.ndim != 3:
            raise ValueError("predict() expects an HxWx3 BGR numpy image")
        inf = self.cfg["waste"]["inference"]
        img = self.preprocess(img_bgr, **(enhance_toggles or {})) if enhance_input else img_bgr
        conf_v = float(conf if conf is not None else inf["conf"])
        iou_v = float(iou if iou is not None else inf["iou"])
        if tiles <= 1:
            res = self._run(img, conf_v, iou_v, augment)
            b = res.boxes
            df = boxes_to_dataframe(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int), self.names, source, frame)
            annotated = res.plot(line_width=line_width)
            speed = sum(v for v in (res.speed or {}).values() if v)
            return DetectionResult(annotated, df, img, float(speed), class_counts(df))

        t0 = time.perf_counter()
        xyxy, scores, cls = self._predict_tiled(img, conf_v, iou_v, int(tiles), augment)
        df = boxes_to_dataframe(xyxy, scores, cls, self.names, source, frame)
        annotated = draw_detections(img, xyxy, scores, cls, self.names, line_width)
        return DetectionResult(annotated, df, img, (time.perf_counter() - t0) * 1000, class_counts(df))

    def _run(self, img: np.ndarray, conf: float, iou: float, augment: bool = False) -> Any:
        """One Ultralytics forward pass on ``img``; returns the first ``Results`` object."""
        return self.model.predict(img, conf=conf, iou=iou, imgsz=self.imgsz, device=self.device, verbose=False,
                                  augment=augment, max_det=int(self.cfg["waste"]["inference"].get("max_det", 1000)),
                                  **_precision_kwargs(self.half))[0]

    def _predict_tiled(self, img: np.ndarray, conf: float, iou: float, tiles: int, augment: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Full-image pass + overlapping tile passes, merged in original-image coordinates."""
        import torch
        from torchvision.ops import batched_nms

        h, w = img.shape[:2]
        boxes, scores, classes = [], [], []
        for (x0, y0, x1, y1) in [(0, 0, w, h)] + tile_grid(w, h, tiles, overlap=0.25):
            r = self._run(img[y0:y1, x0:x1], conf, iou, augment).boxes
            if len(r) == 0:
                continue
            b = r.xyxy.cpu().numpy().astype(np.float32)
            if (x1 - x0, y1 - y0) != (w, h):
                # drop boxes cut by an *interior* tile edge: the neighbouring tile or the
                # full-image pass sees those objects whole
                margin = 2.0
                cut = ((b[:, 0] <= margin) & (x0 > 0)) | ((b[:, 2] >= (x1 - x0) - margin) & (x1 < w)) | \
                      ((b[:, 1] <= margin) & (y0 > 0)) | ((b[:, 3] >= (y1 - y0) - margin) & (y1 < h))
                keep_mask = ~cut
                b = b[keep_mask]
                r_conf, r_cls = r.conf.cpu().numpy()[keep_mask], r.cls.cpu().numpy()[keep_mask]
            else:
                r_conf, r_cls = r.conf.cpu().numpy(), r.cls.cpu().numpy()
            b[:, [0, 2]] += x0
            b[:, [1, 3]] += y0
            boxes.append(b)
            scores.append(r_conf)
            classes.append(r_cls)
        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, int)
        bx, sc, cl = (np.concatenate(v) for v in (boxes, scores, classes))
        keep = batched_nms(torch.from_numpy(bx), torch.from_numpy(sc.astype(np.float32)), torch.from_numpy(cl.astype(np.int64)), iou).numpy()
        bx, sc, cl = bx[keep], sc[keep], cl[keep].astype(int)
        # suppress duplicates of one object that got different class labels in different passes
        keep2 = _cross_class_dedup(bx, sc, thr=0.7)
        return bx[keep2], sc[keep2], cl[keep2]

    def predict_batch(self, paths: list[Path], **kw: Any) -> Iterator[tuple[Path, DetectionResult]]:
        """Yield ``(path, result)`` for every readable image in ``paths``."""
        for p in paths:
            img = cv2.imread(str(p))
            if img is None:
                log.warning("Unreadable image skipped: %s", p)
                continue
            yield p, self.predict(img, source=p.name, **kw)

    def predict_video(self, src: str | Path | int, out_path: Path | None = None, stride: int = 1, max_frames: int | None = None, progress: Callable[[float], None] | None = None, **kw: Any) -> tuple[pd.DataFrame, Path | None, list[np.ndarray]]:
        """Run detection on a video file (or camera index).

        Returns ``(all detections, annotated video path or None, a few sample annotated frames)``.
        """
        cap = cv2.VideoCapture(src if isinstance(src, int) else str(src))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {src}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        max_frames = max_frames or self.cfg["waste"]["inference"]["max_video_frames"]
        writer, frames, samples = None, [], []
        idx = processed = 0
        try:
            while processed < max_frames:
                ok, frame = cap.read()
                if not ok:
                    break
                if idx % max(1, stride) == 0:
                    r = self.predict(frame, source=Path(str(src)).name, frame=idx, **kw)
                    frames.append(r.detections)
                    if out_path is not None:
                        if writer is None:
                            writer, out_path = _open_writer(out_path, fps / max(1, stride), r.annotated.shape[1], r.annotated.shape[0])
                        writer.write(r.annotated)
                    if len(samples) < 6 and (processed % 15 == 0):
                        samples.append(r.annotated)
                    processed += 1
                    if progress and total:
                        progress(min(1.0, idx / total))
                idx += 1
        finally:
            cap.release()
            if writer is not None:
                writer.release()
        df = pd.concat(frames, ignore_index=True) if frames and any(not f.empty for f in frames) else empty_detections()
        return df, (out_path if writer is not None else None), samples

    def webcam(self, camera: int = 0, **kw: Any) -> None:  # pragma: no cover - interactive
        """OpenCV window with live detections; press ``q`` to quit."""
        cap = cv2.VideoCapture(camera)
        if not cap.isOpened():
            raise ValueError(f"Cannot open webcam {camera}")
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                r = self.predict(frame, source=f"webcam{camera}", **kw)
                cv2.putText(r.annotated, f"{r.n} items | {r.inference_ms:.0f} ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Ocean waste detection (q to quit)", r.annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def tile_grid(w: int, h: int, n: int, overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
    """``n x n`` overlapping tiles ``(x0, y0, x1, y1)`` that together cover a ``w x h`` image."""
    if n <= 1:
        return [(0, 0, w, h)]
    tw = min(w, int(np.ceil(w / n * (1 + overlap))))
    th = min(h, int(np.ceil(h / n * (1 + overlap))))
    xs = np.linspace(0, w - tw, n).round().astype(int)
    ys = np.linspace(0, h - th, n).round().astype(int)
    return [(int(x), int(y), int(x) + tw, int(y) + th) for y in ys for x in xs]


def _cross_class_dedup(boxes: np.ndarray, scores: np.ndarray, thr: float = 0.7) -> np.ndarray:
    """Indices kept after removing lower-score boxes overlapping a kept box by IoU >= ``thr`` (any class)."""
    order = np.argsort(-scores)
    keep: list[int] = []
    for i in order:
        if all(_iou_xyxy(boxes[i], boxes[j]) < thr for j in keep):
            keep.append(int(i))
    return np.array(sorted(keep), dtype=int)


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def draw_detections(img: np.ndarray, xyxy: np.ndarray, scores: np.ndarray, cls: np.ndarray, names: dict[int, str], line_width: int | None = None) -> np.ndarray:
    """Draw boxes in the same style as Ultralytics ``Results.plot()``."""
    from ultralytics.utils.plotting import Annotator, colors

    ann = Annotator(img.copy(), line_width=line_width)
    for b, s, c in zip(xyxy, scores, cls):
        ann.box_label(b.tolist(), f"{names.get(int(c), int(c))} {s:.2f}", color=colors(int(c), True))
    return ann.result()


def _precision_kwargs(half: bool) -> dict[str, Any]:
    """FP16 flag for Ultralytics: ``quantize=16`` on >= 8.4 (``half`` is deprecated there), else ``half``."""
    from ultralytics.cfg import DEFAULT_CFG_DICT

    if "quantize" in DEFAULT_CFG_DICT:
        return {"quantize": 16 if half else None}
    return {"half": half}


def _open_writer(path: Path, fps: float, w: int, h: int) -> tuple[cv2.VideoWriter, Path]:
    """Open a browser-playable video writer.

    Prefers WebM/VP8 (bundled with opencv-python's FFmpeg, plays in every modern browser);
    falls back to MP4/mp4v (downloadable, but not playable inline in most browsers).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    for codec, ext in (("VP80", ".webm"), ("mp4v", ".mp4")):
        target = path.with_suffix(ext)
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*codec), max(1.0, fps), (w, h))
        if writer.isOpened():
            return writer, target
        writer.release()
    raise RuntimeError("No usable video codec found in OpenCV")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="image | folder | video | 'webcam' | camera index")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--iou", type=float, default=None)
    ap.add_argument("--enhance", action="store_true", help="dehaze + CLAHE before inference")
    ap.add_argument("--tiles", type=int, default=1, help="high-recall mode: N x N zoomed tiles + full image (e.g. 3)")
    ap.add_argument("--tta", action="store_true", help="test-time augmentation (more recall, slower)")
    ap.add_argument("--out", default="reports/waste/predictions")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    det = WasteDetector(args.weights, cfg) if args.weights else WasteDetector.from_config(cfg)
    out = ensure_dir(args.out)
    kw = {"conf": args.conf, "iou": args.iou, "enhance_input": args.enhance, "tiles": args.tiles, "augment": args.tta}
    src = args.source
    if src == "webcam" or src.isdigit():
        det.webcam(0 if src == "webcam" else int(src), **kw)
        return
    p = resolve_path(src) if not Path(src).is_absolute() else Path(src)
    if p.is_dir() or p.suffix.lower() in IMG_EXTS:
        paths = sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXTS) if p.is_dir() else [p]
        dfs = []
        for path, r in det.predict_batch(paths, **kw):
            cv2.imwrite(str(out / f"{path.stem}_pred.jpg"), r.annotated)
            dfs.append(r.detections)
            log.info("%s: %d detections %s", path.name, r.n, r.class_counts)
        df = pd.concat(dfs, ignore_index=True) if dfs else empty_detections()
    elif p.suffix.lower() in VID_EXTS:
        df, vid, _ = det.predict_video(p, out / f"{p.stem}_pred.webm", **kw)
        log.info("Annotated video -> %s", vid)
    else:
        raise SystemExit(f"Unsupported source: {src}")
    df.to_csv(out / "detections.csv", index=False)
    log.info("%d detections -> %s", len(df), out / "detections.csv")


if __name__ == "__main__":
    main()
