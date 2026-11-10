"""Merge raw datasets into one YOLO dataset with the unified class list.

Supports, under ``data/raw/<source>/``:
  * YOLO-format datasets (``data.yaml`` with ``names`` or ``classes.txt``; labels in a
    sibling ``labels/`` folder mirroring ``images/``) - e.g. Roboflow exports, synthetic data.
  * COCO JSON datasets (``images``/``annotations``/``categories``) - e.g. TrashCan 1.0.

Class names are normalised and mapped through ``waste.class_mapping`` in config.yaml.
Output: ``data/processed/yolo/{images,labels}/{train,val,test}`` + ``data/waste.yaml``.

Usage::

    python -m src.data.prepare_yolo_dataset [--max-images 2000] [--enhance] [--sources synthetic_waste]
    python -m src.data.prepare_yolo_dataset --source-dir trashcan=dataset/dataset/material_version
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import yaml

from src.utils.common import ensure_dir, get_logger, load_config, resolve_path

log = get_logger(__name__)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


@dataclass
class Sample:
    """One image and its labels in unified class ids, YOLO-normalised xywh."""

    image: Path
    source: str
    labels: list[tuple[int, float, float, float, float]] = field(default_factory=list)


def normalise_name(name: str) -> str:
    """``'Plastic Bottle'`` / ``'plastic-bottle'`` -> ``'plastic_bottle'``."""
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def build_class_mapper(cfg: dict[str, Any], unknown_to_other: bool = False):
    """Return ``f(source_name) -> unified id | None``."""
    classes = cfg["waste"]["classes"]
    idx = {c: i for i, c in enumerate(classes)}
    mapping = {normalise_name(k): v for k, v in (cfg["waste"].get("class_mapping") or {}).items()}
    warned: set[str] = set()

    def mapper(name: str) -> int | None:
        n = normalise_name(name)
        if n in idx:
            return idx[n]
        if n in mapping:
            target = mapping[n]
            return None if target is None else idx[target]
        if unknown_to_other and "other_debris" in idx:
            return idx["other_debris"]
        if n not in warned:
            warned.add(n)
            log.warning("Unmapped class '%s' dropped (add it to waste.class_mapping)", name)
        return None

    return mapper


def _yolo_names(root: Path) -> list[str] | None:
    for yml in list(root.glob("data.yaml")) + list(root.glob("*.yaml")):
        try:
            names = (yaml.safe_load(yml.read_text(encoding="utf-8")) or {}).get("names")
        except yaml.YAMLError:
            continue
        if isinstance(names, dict):
            return [names[k] for k in sorted(names)]
        if isinstance(names, list):
            return names
    txt = root / "classes.txt"
    if txt.exists():
        return [ln.strip() for ln in txt.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return None


def _label_path(img: Path) -> Path:
    parts = list(img.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    return Path(*parts).with_suffix(".txt")


def collect_yolo(root: Path, mapper) -> list[Sample]:
    """Collect samples from a YOLO-format folder."""
    names = _yolo_names(root)
    if names is None:
        return []
    samples = []
    for img in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS and "images" in p.parts):
        s = Sample(img, root.name)
        lp = _label_path(img)
        if lp.exists():
            for line in lp.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                cid = int(float(parts[0]))
                if cid >= len(names):
                    continue
                uid = mapper(names[cid])
                if uid is None:
                    continue
                if len(parts) > 5:  # polygon/segment label -> bbox
                    xs, ys = list(map(float, parts[1::2])), list(map(float, parts[2::2]))
                    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                    s.labels.append((uid, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0))
                else:
                    s.labels.append((uid, *map(float, parts[1:5])))  # type: ignore[arg-type]
        samples.append(s)
    return samples


def collect_coco(root: Path, mapper) -> list[Sample]:
    """Collect samples from every COCO JSON under ``root`` (bbox annotations)."""
    samples: list[Sample] = []
    file_index: dict[str, Path] | None = None
    for js in sorted(root.rglob("*.json")):
        try:
            data = json.loads(js.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict) or not {"images", "annotations", "categories"} <= data.keys():
            continue
        if file_index is None:
            file_index = {p.name: p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS}
        cats = {c["id"]: mapper(c["name"]) for c in data["categories"]}
        by_img: dict[int, list[dict]] = {}
        for ann in data["annotations"]:
            by_img.setdefault(ann["image_id"], []).append(ann)
        for im in data["images"]:
            path = file_index.get(Path(im["file_name"]).name)
            if path is None:
                continue
            w, h = float(im["width"]), float(im["height"])
            s = Sample(path, root.name)
            for ann in by_img.get(im["id"], []):
                uid = cats.get(ann["category_id"])
                if uid is None or "bbox" not in ann:
                    continue
                x, y, bw, bh = ann["bbox"]
                if bw <= 1 or bh <= 1:
                    continue
                s.labels.append((uid, (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h))
            samples.append(s)
        log.info("COCO %s: %d images", js.name, len(data["images"]))
    return samples


_VIDEO_RE = re.compile(r"^(vid_\d+)_frame", re.IGNORECASE)
_ROBOFLOW_RE = re.compile(r"_(jpe?g|png|bmp|webp)\.rf\.[0-9a-f]+$", re.IGNORECASE)
_AUGMENT_RE = re.compile(r"__(flipv|fliph|rot\d*|blur|noise|bright\w*|crop\w*|aug\w*)$", re.IGNORECASE)


def group_key(sample: Sample) -> str:
    """Group id used to keep related images in the same split.

    Prevents data leakage between train and test:
    * video frames (``vid_000123_frame0000045.jpg``, TrashCan naming) are grouped by video;
    * Roboflow exports (``name_jpg.rf.<hash>.jpg``) and their augmented copies
      (``name__flipv_jpg.rf.<hash>.jpg``) are grouped by the original image name.
    Any other image is its own group.
    """
    if sample.image is None:
        return f"_{id(sample)}"
    p = Path(sample.image)
    m = _VIDEO_RE.match(p.name)
    if m:
        return f"{sample.source}/{m.group(1)}"
    stem = _ROBOFLOW_RE.sub("", p.stem)
    stem = _AUGMENT_RE.sub("", stem)
    return f"{sample.source}/{stem}"


def split_samples(samples: list[Sample], ratios: list[float], seed: int) -> dict[str, list[Sample]]:
    """Seeded, group-aware split into train/val/test by ``ratios`` (counted in images).

    Whole groups (e.g. all frames of one video) are assigned to a single split.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")
    groups: dict[str, list[Sample]] = {}
    for smp in samples:
        groups.setdefault(group_key(smp), []).append(smp)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(samples)
    t_train, t_val = n * ratios[0], n * (ratios[0] + ratios[1])
    out: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    seen = 0
    for k in keys:
        split = "train" if seen < t_train else "val" if seen < t_val else "test"
        out[split].extend(groups[k])
        seen += len(groups[k])
    return out


def split_score(splits: dict[str, list[Sample]], ratios: list[float], n_classes: int) -> float:
    """Lower is better: distance of per-class and per-image split fractions from ``ratios``.

    A class that has instances overall but none in some split costs a large penalty, so
    every class can be both trained on and evaluated.
    """
    names = ("train", "val", "test")
    totals = [0] * n_classes
    per = {k: [0] * n_classes for k in names}
    for k in names:
        for smp in splits[k]:
            for lab in smp.labels:
                per[k][lab[0]] += 1
                totals[lab[0]] += 1
    n_img = sum(len(v) for v in splits.values()) or 1
    score = sum(abs(len(splits[k]) / n_img - r) for k, r in zip(names, ratios)) * 3
    for c, tot in enumerate(totals):
        if not tot:
            continue
        for k, r in zip(names, ratios):
            score += abs(per[k][c] / tot - r) + (10.0 if per[k][c] == 0 else 0.0)
    return score


def balanced_split(samples: list[Sample], ratios: list[float], seed: int, n_classes: int, tries: int = 300) -> tuple[dict[str, list[Sample]], int]:
    """Group-aware split; tries ``tries`` seeds and keeps the most class-balanced one.

    Returns ``(splits, chosen_seed)``. Deterministic for a given ``seed``.
    """
    best, best_seed, best_score = None, seed, float("inf")
    for t in range(max(1, tries)):
        cand = split_samples(samples, ratios, seed + t)
        sc = split_score(cand, ratios, n_classes)
        if sc < best_score:
            best, best_seed, best_score = cand, seed + t, sc
    assert best is not None
    return best, best_seed


def write_dataset_yaml(cfg: dict[str, Any]) -> Path:
    """(Re)write ``data/waste.yaml`` with the absolute dataset path (Ultralytics requirement)."""
    ds = resolve_path(cfg["waste"]["dataset_dir"]).resolve()
    out = resolve_path(cfg["waste"]["dataset_yaml"])
    content = {
        "path": ds.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": dict(enumerate(cfg["waste"]["classes"])),
    }
    out.write_text("# Auto-generated by prepare_yolo_dataset.py - do not edit by hand\n" + yaml.safe_dump(content, sort_keys=False), encoding="utf-8")
    return out


def parse_source_dir(spec: str) -> tuple[str, Path]:
    """``"name=path"`` or ``"path"`` -> ``(name, resolved path)``; the name defaults to the folder name."""
    name, sep, path = spec.partition("=")
    if not sep or len(name) == 1:  # no '=' or a Windows drive letter like 'D:\...'
        name, path = "", spec
    p = resolve_path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"--source-dir {spec!r}: folder not found ({p})")
    return (name or p.name), p


def prepare(cfg: dict[str, Any], sources: list[str] | None = None, max_images: int | None = None, enhance_images: bool = False, unknown_to_other: bool = False, source_dirs: list[str] | None = None) -> dict[str, Any]:
    """Build the processed dataset. Returns statistics (also saved to reports/waste/).

    ``source_dirs`` are extra dataset folders anywhere on disk (``name=path`` or ``path``). When given
    and ``sources`` is None, ``data/raw`` is not scanned, so only those datasets are used.
    """
    raw = resolve_path(cfg["paths"]["data_raw"])
    mapper = build_class_mapper(cfg, unknown_to_other)
    roots: list[tuple[str, Path]] = []
    if not source_dirs or sources is not None:
        roots += [(d.name, d) for d in sorted(raw.iterdir()) if d.is_dir() and (not sources or d.name in sources)] if raw.exists() else []
    roots += [parse_source_dir(sd) for sd in (source_dirs or [])]
    samples: list[Sample] = []
    for name, root in roots:
        got = collect_yolo(root, mapper) or collect_coco(root, mapper)
        for smp in got:
            smp.source = name
        log.info("Source %-25s -> %d images (%s)", name, len(got), root)
        samples.extend(got)
    if not samples:
        raise FileNotFoundError(f"No datasets found under {raw}. Run: python -m src.data.download_data")
    if max_images and len(samples) > max_images:
        random.Random(cfg["project"]["seed"]).shuffle(samples)
        samples = samples[:max_images]

    out = resolve_path(cfg["waste"]["dataset_dir"])
    if out.exists():
        shutil.rmtree(out)
    enh_opts = None
    if enhance_images:
        from src.preprocessing.underwater_enhance import enhance, options_from_config

        enh_opts = options_from_config(cfg)

    splits, split_seed = balanced_split(samples, cfg["waste"]["split"], cfg["project"]["seed"], len(cfg["waste"]["classes"]))
    stats: dict[str, Any] = {
        "split_seed": split_seed,
        "n_images": len(samples),
        "sources": sorted({s.source for s in samples}),
        "split_groups": {k: len({group_key(x) for x in v}) for k, v in splits.items()},
        "splits": {},
        "classes": cfg["waste"]["classes"],
    }
    for split, items in splits.items():
        img_dir, lbl_dir = ensure_dir(out / "images" / split), ensure_dir(out / "labels" / split)
        counts: Counter[str] = Counter()
        for i, s in enumerate(items):
            stem = f"{s.source}_{i:06d}"
            dst = img_dir / f"{stem}{s.image.suffix.lower()}"
            if enh_opts is not None:
                img = cv2.imread(str(s.image))
                if img is None:
                    continue
                cv2.imwrite(str(dst), enhance(img, enh_opts))
            else:
                shutil.copy2(s.image, dst)
            lines = []
            for uid, xc, yc, w, h in s.labels:
                xc, yc = min(max(xc, 0.0), 1.0), min(max(yc, 0.0), 1.0)
                w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
                if w > 0 and h > 0:
                    lines.append(f"{uid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
                    counts[cfg["waste"]["classes"][uid]] += 1
            (lbl_dir / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        stats["splits"][split] = {"images": len(items), "instances": dict(counts)}
        log.info("%-5s: %5d images, %6d boxes", split, len(items), sum(counts.values()))

    yaml_path = write_dataset_yaml(cfg)
    rep = ensure_dir(Path(cfg["paths"]["reports"]) / "waste")
    (rep / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    log.info("Dataset ready: %s (yaml: %s)", out, yaml_path)
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", nargs="*", default=None, help="folder names under data/raw (default: all)")
    ap.add_argument("--source-dir", action="append", default=None, metavar="[NAME=]PATH",
                    help="dataset folder anywhere on disk (repeatable); when used without --sources, data/raw is skipped")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--enhance", action="store_true", help="apply dehaze+CLAHE to training images")
    ap.add_argument("--unknown-to-other", action="store_true", help="map unknown labels to other_debris")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    source_dirs = args.source_dir
    if source_dirs is None and args.sources is None:
        source_dirs = [sd for sd in cfg["waste"].get("source_dirs") or [] if resolve_path(sd.partition("=")[2] or sd).is_dir()] or None
        if source_dirs:
            log.info("Using configured dataset(s): %s", source_dirs)
    prepare(cfg, args.sources, args.max_images, args.enhance, args.unknown_to_other, source_dirs)


if __name__ == "__main__":
    main()
