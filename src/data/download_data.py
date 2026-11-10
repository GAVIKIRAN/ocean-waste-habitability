"""Fetch the datasets for both modules.

Waste detection sources (``--waste``):
  * ``roboflow``  - any Roboflow Universe underwater-trash project in YOLO format
                    (needs ``pip install roboflow`` and ``ROBOFLOW_API_KEY``).
  * ``trashcan``  - TrashCan 1.0 (Hong et al., 2020, Univ. of Minnesota). The archive
                    must be downloaded manually (licence click-through), see README.
                    Extract it under ``data/raw/trashcan/``; the COCO JSONs are converted
                    by ``prepare_yolo_dataset.py``.
  * ``synthetic`` - procedural underwater-trash images (offline fallback, always works).
  * ``auto``      - use whatever real dataset is already in ``data/raw``; otherwise
                    roboflow if configured; otherwise synthetic.

Water potability sources (``--water``): ``auto`` | ``kaggle`` | ``download`` | ``synthetic``.

Usage::

    python -m src.data.download_data                     # auto for both
    python -m src.data.download_data --waste synthetic --n-images 400
    python -m src.data.download_data --waste roboflow --rf-workspace WS --rf-project PROJ --rf-version 1
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from src.data.synthetic_waste import generate_dataset
from src.data.water_data import download_water_data
from src.utils.common import get_logger, load_config, resolve_path, set_seed

log = get_logger(__name__)

TRASHCAN_INSTRUCTIONS = """
TrashCan 1.0 must be downloaded manually:
  1. Visit https://conservancy.umn.edu/handle/11299/214865
  2. Download 'dataset.zip' and extract to data/raw/trashcan/
     (expected: data/raw/trashcan/material_version/{train,val}/ + instances_*_trashcan.json)
  3. Run: python -m src.data.prepare_yolo_dataset
"""


def _real_waste_present(raw: Path) -> list[Path]:
    """Return non-synthetic dataset folders already present under ``data/raw``."""
    found = []
    for d in raw.iterdir() if raw.exists() else []:
        if d.is_dir() and d.name not in ("synthetic_waste",) and (any(d.rglob("*.json")) or any(d.rglob("data.yaml"))):
            found.append(d)
    return found


def download_roboflow(workspace: str, project: str, version: int, out_dir: Path) -> Path | None:
    """Download a Roboflow project in YOLOv11 format. Returns the folder or None."""
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        log.warning("ROBOFLOW_API_KEY not set - skipping Roboflow download")
        return None
    try:
        from roboflow import Roboflow  # type: ignore[import-not-found]
    except ImportError:
        log.warning("roboflow package not installed (pip install -r requirements-optional.txt)")
        return None
    try:
        rf = Roboflow(api_key=key)
        ds = rf.workspace(workspace).project(project).version(version).download("yolov11", location=str(out_dir))
        return Path(ds.location)
    except Exception as exc:  # noqa: BLE001
        log.warning("Roboflow download failed: %s", exc)
        return None


def get_waste_data(cfg: dict[str, Any], source: str, args: argparse.Namespace) -> str:
    """Obtain a waste dataset; returns the source actually used."""
    raw = resolve_path(cfg["paths"]["data_raw"])
    raw.mkdir(parents=True, exist_ok=True)
    if source == "auto":
        existing = _real_waste_present(raw)
        if existing:
            log.info("Using existing dataset(s): %s", [p.name for p in existing])
            return "existing"
    if source in ("auto", "roboflow") and args.rf_workspace and args.rf_project:
        if download_roboflow(args.rf_workspace, args.rf_project, args.rf_version, raw / f"roboflow_{args.rf_project}"):
            return "roboflow"
    if source == "trashcan":
        if (raw / "trashcan").exists():
            return "trashcan"
        log.warning(TRASHCAN_INSTRUCTIONS)
    if source != "synthetic":
        log.info("No real waste dataset available - generating the synthetic demo dataset")
    syn = cfg["waste"]["synthetic"]
    out = raw / "synthetic_waste"
    n = args.n_images or syn["n_images"]
    if out.exists() and len(list((out / "images").glob("*.jpg"))) >= n and not args.force:
        log.info("Synthetic dataset already present (%s)", out)
    else:
        generate_dataset(out, cfg["waste"]["classes"], n, syn["width"], syn["height"], syn["max_objects"], cfg["project"]["seed"])
        log.info("Wrote %d synthetic images to %s", n, out)
    return "synthetic"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--waste", default="auto", choices=["auto", "roboflow", "trashcan", "synthetic", "skip"])
    ap.add_argument("--water", default="auto", choices=["auto", "kaggle", "download", "synthetic", "skip"])
    ap.add_argument("--n-images", type=int, default=None, help="synthetic image count")
    ap.add_argument("--rf-workspace", default=os.environ.get("ROBOFLOW_WORKSPACE"))
    ap.add_argument("--rf-project", default=os.environ.get("ROBOFLOW_PROJECT"))
    ap.add_argument("--rf-version", type=int, default=int(os.environ.get("ROBOFLOW_VERSION", 1)))
    ap.add_argument("--force", action="store_true", help="re-download / regenerate")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    if args.waste != "skip":
        log.info("Waste dataset source: %s", get_waste_data(cfg, args.waste, args))
    if args.water != "skip":
        path, used = download_water_data(cfg, args.water, args.force)
        log.info("Water dataset source: %s (%s)", used, path)


if __name__ == "__main__":
    main()
