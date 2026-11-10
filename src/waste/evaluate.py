"""Evaluate the trained waste detector on the held-out test split.

Saves to ``reports/waste/``:
  * ``metrics.json``           - mAP@0.5, mAP@0.5:0.95, precision, recall, F1 (+ per class)
  * ``per_class_metrics.csv``
  * ``confusion_matrix*.png``, ``*PR_curve.png``, ``*F1_curve.png`` ... (Ultralytics plots)
  * ``prediction_grid.png``    - qualitative predictions vs. ground truth

Usage::

    python -m src.waste.evaluate [--split test] [--weights models/waste_best.pt]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data.prepare_yolo_dataset import write_dataset_yaml  # noqa: E402
from src.utils.common import describe_device, ensure_dir, get_device, get_logger, load_config, resolve_path, safe_workers  # noqa: E402

log = get_logger(__name__)


def f1(p: float, r: float) -> float:
    """Harmonic mean of precision and recall (0 when both are 0)."""
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def draw_gt(img: np.ndarray, label_file: Path, names: list[str]) -> np.ndarray:
    """Draw ground-truth YOLO boxes (green) on a copy of ``img``."""
    out = img.copy()
    h, w = out.shape[:2]
    if label_file.exists():
        for line in label_file.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:5])
            x1, y1, x2, y2 = int((xc - bw / 2) * w), int((yc - bh / 2) * h), int((xc + bw / 2) * w), int((yc + bh / 2) * h)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out, names[c] if c < len(names) else str(c), (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return out


def prediction_grid(model: Any, images: list[Path], names: list[str], out_file: Path, device: int | str, imgsz: int, n: int = 8) -> Path | None:
    """Save a grid with ground truth (left) vs. prediction (right) for ``n`` images."""
    images = images[:n]
    if not images:
        return None
    fig, axes = plt.subplots(len(images), 2, figsize=(10, 3.2 * len(images)))
    axes = np.atleast_2d(axes)
    for row, p in enumerate(images):
        img = cv2.imread(str(p))
        lbl = p.parents[2] / "labels" / p.parent.name / f"{p.stem}.txt"
        pred = model.predict(img, imgsz=imgsz, device=device, verbose=False, conf=0.25)[0].plot()
        for col, (im, title) in enumerate(((draw_gt(img, lbl, names), "Ground truth"), (pred, "Prediction"))):
            axes[row, col].imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            axes[row, col].set_title(f"{title} - {p.name}", fontsize=8)
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(out_file, dpi=110)
    plt.close(fig)
    return out_file


def evaluate(cfg: dict[str, Any], weights: str | Path | None = None, split: str = "test", batch: int = 8) -> dict[str, Any]:
    """Run ``model.val`` on ``split`` and persist metrics + plots. Returns the metrics dict."""
    from ultralytics import YOLO

    weights = Path(weights) if weights else resolve_path(cfg["waste"]["weights"])
    if not weights.exists():
        raise FileNotFoundError(f"No weights at {weights}. Train first: python -m src.waste.train --mode demo")
    device = get_device()
    print(f"[waste/evaluate] Device: {describe_device(device)}", flush=True)
    data_yaml = write_dataset_yaml(cfg)
    model = YOLO(str(weights))
    imgsz = int(model.ckpt.get("train_args", {}).get("imgsz", cfg["waste"]["inference"]["imgsz"])) if getattr(model, "ckpt", None) else cfg["waste"]["inference"]["imgsz"]
    rep = ensure_dir(Path(cfg["paths"]["reports"]) / "waste")
    val_dir = resolve_path("runs/waste")
    m = model.val(data=str(data_yaml), split=split, imgsz=imgsz, batch=batch, device=device, workers=safe_workers(2), plots=True, project=str(val_dir), name=f"eval_{split}", exist_ok=True, verbose=False)

    names = cfg["waste"]["classes"]
    p, r = float(m.box.mp), float(m.box.mr)
    metrics: dict[str, Any] = {
        "split": split,
        "weights": weights.name,
        "imgsz": imgsz,
        "mAP50": round(float(m.box.map50), 4),
        "mAP50_95": round(float(m.box.map), 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1": round(f1(p, r), 4),
        "speed_ms_per_image": {k: round(float(v), 2) for k, v in (m.speed or {}).items()},
    }
    rows = []
    for i, cid in enumerate(m.box.ap_class_index):
        pc, rc = float(m.box.p[i]), float(m.box.r[i])
        rows.append({"class": names[int(cid)] if int(cid) < len(names) else str(cid), "precision": round(pc, 4), "recall": round(rc, 4), "f1": round(f1(pc, rc), 4), "mAP50": round(float(m.box.ap50[i]), 4), "mAP50_95": round(float(m.box.ap[i]), 4)})
    per_class = pd.DataFrame(rows)
    per_class.to_csv(rep / "per_class_metrics.csv", index=False)
    metrics["per_class"] = rows
    (rep / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    for f in Path(m.save_dir).glob("*.png"):
        shutil.copy2(f, rep / f.name)
    test_imgs = sorted((resolve_path(cfg["waste"]["dataset_dir"]) / "images" / split).glob("*"))
    prediction_grid(model, test_imgs, names, rep / "prediction_grid.png", device, imgsz)
    log.info("mAP50=%.3f mAP50-95=%.3f P=%.3f R=%.3f F1=%.3f -> %s", metrics["mAP50"], metrics["mAP50_95"], p, r, metrics["f1"], rep)
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    metrics = evaluate(load_config(args.config), args.weights, args.split, args.batch)
    print(json.dumps({k: v for k, v in metrics.items() if k != "per_class"}, indent=2))


if __name__ == "__main__":
    main()
