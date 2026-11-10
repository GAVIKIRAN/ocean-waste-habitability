"""Water-potability data: download (Kaggle / public mirror), synthetic fallback, loading."""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.common import get_logger, resolve_path

log = get_logger(__name__)

# Summary statistics of the Kaggle "Water Potability" dataset (mean, std, missing-rate).
_KAGGLE_STATS: dict[str, tuple[float, float, float]] = {
    "ph": (7.08, 1.59, 0.15),
    "Hardness": (196.4, 32.9, 0.0),
    "Solids": (22014.0, 8768.0, 0.0),
    "Chloramines": (7.12, 1.58, 0.0),
    "Sulfate": (333.8, 41.4, 0.24),
    "Conductivity": (426.2, 80.8, 0.0),
    "Organic_carbon": (14.28, 3.31, 0.0),
    "Trihalomethanes": (66.4, 16.2, 0.05),
    "Turbidity": (3.97, 0.78, 0.0),
}


def validate_water_df(df: pd.DataFrame, features: list[str], target: str | None = None) -> None:
    """Raise ``ValueError`` if required columns are missing or non-numeric."""
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required water-quality columns: {missing}")
    if target and target not in df.columns:
        raise ValueError(f"Missing target column '{target}'")
    bad = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise ValueError(f"Non-numeric values in columns: {bad}")


def generate_synthetic_water(n: int = 3276, seed: int = 42, heavy_metals: bool = False) -> pd.DataFrame:
    """Synthetic data with the same schema/marginals as the Kaggle dataset.

    The label follows a noisy logistic rule on physically meaningful deviations
    (pH distance from neutral, chloramines, THMs, turbidity, solids, sulfate),
    so that models trained on it learn sensible relationships.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.normal(m, s, n) for c, (m, s, _) in _KAGGLE_STATS.items()})
    df["ph"] = df["ph"].clip(0, 14)
    df = df.clip(lower=0)
    z = lambda c: (df[c] - _KAGGLE_STATS[c][0]) / _KAGGLE_STATS[c][1]  # noqa: E731
    logit = (
        -0.4
        - 1.1 * (df["ph"] - 7.0).abs() / 1.5
        - 0.6 * z("Chloramines")
        - 0.5 * z("Trihalomethanes")
        - 0.6 * z("Turbidity")
        - 0.4 * z("Solids")
        - 0.3 * z("Sulfate")
        - 0.2 * z("Organic_carbon")
        + rng.normal(0, 0.6, n)
    )
    if heavy_metals:
        df["Lead"] = rng.gamma(2.0, 0.004, n)
        df["Mercury"] = rng.gamma(2.0, 0.0015, n)
        logit -= 3.0 * (df["Lead"] > 0.01) + 3.0 * (df["Mercury"] > 0.006)
    df["Potability"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    for c, (_, _, miss) in _KAGGLE_STATS.items():
        if miss:
            df.loc[rng.random(n) < miss, c] = np.nan
    return df


def _try_kaggle(dataset: str, out_dir: Path) -> Path | None:
    """Download via the Kaggle CLI if credentials exist (~/.kaggle/kaggle.json or env vars)."""
    has_creds = (Path.home() / ".kaggle" / "kaggle.json").exists() or os.environ.get("KAGGLE_KEY")
    if not has_creds or not shutil.which("kaggle"):
        return None
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_dir), "--unzip"], check=True, timeout=300)
        hits = list(out_dir.glob("water_potability*.csv"))
        return hits[0] if hits else None
    except Exception as exc:  # noqa: BLE001
        log.warning("Kaggle download failed: %s", exc)
        return None


def _try_url(url: str, features: list[str], target: str) -> pd.DataFrame | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed https mirror from config
            raw = resp.read(20_000_000)
        df = pd.read_csv(io.BytesIO(raw))
        validate_water_df(df, features, target)
        return df
    except Exception as exc:  # noqa: BLE001
        log.warning("Download from %s failed: %s", url, exc)
        return None


def _mark(csv: Path, source: str) -> None:
    """Record where the CSV came from (``<csv>.source``)."""
    csv.with_suffix(".source").write_text(source, encoding="utf-8")


def data_source(csv: Path) -> str:
    """Provenance recorded by :func:`_mark` (``unknown`` if absent)."""
    marker = csv.with_suffix(".source")
    return marker.read_text(encoding="utf-8").strip() if marker.exists() else "user-provided"


def download_water_data(cfg: dict[str, Any], source: str = "auto", force: bool = False) -> tuple[Path, str]:
    """Make sure the water CSV exists. Returns ``(path, source_used)``.

    ``source``: ``auto`` (kaggle -> mirror -> synthetic), ``kaggle``, ``download`` or ``synthetic``.
    """
    wcfg = cfg["water"]
    out = resolve_path(wcfg["data_file"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        log.info("Water dataset already present: %s", out)
        return out, "existing"
    feats, target = wcfg["features"], wcfg["target"]

    if source in ("auto", "kaggle"):
        p = _try_kaggle(wcfg["kaggle_dataset"], out.parent)
        if p is not None:
            if p != out:
                p.replace(out)
            _mark(out, "kaggle")
            return out, "kaggle"
    if source in ("auto", "download"):
        for url in wcfg.get("download_urls", []):
            df = _try_url(url, feats, target)
            if df is not None:
                df.to_csv(out, index=False)
                _mark(out, f"mirror:{url}")
                log.info("Downloaded water dataset (%d rows) from %s", len(df), url)
                return out, "mirror"
    if source == "kaggle" or source == "download":
        log.warning("Requested source '%s' unavailable - falling back to synthetic data", source)
    df = generate_synthetic_water(seed=cfg["project"]["seed"])
    df.to_csv(out, index=False)
    _mark(out, "synthetic")
    log.info("Generated synthetic water dataset (%d rows) -> %s", len(df), out)
    return out, "synthetic"


def available_features(df: pd.DataFrame, cfg: dict[str, Any]) -> list[str]:
    """Core features plus any optional (heavy-metal) columns present in ``df``."""
    w = cfg["water"]
    return list(w["features"]) + [c for c in w.get("optional_features", []) if c in df.columns]


def load_water_data(cfg: dict[str, Any], path: str | Path | None = None) -> pd.DataFrame:
    """Load the water CSV (downloading / synthesising it if absent) and validate it."""
    p = resolve_path(path) if path else resolve_path(cfg["water"]["data_file"])
    if not p.exists():
        p, _ = download_water_data(cfg)
    df = pd.read_csv(p)
    df.attrs["source"] = data_source(p)
    validate_water_df(df, cfg["water"]["features"], cfg["water"]["target"])
    df = df.dropna(subset=[cfg["water"]["target"]])
    df[cfg["water"]["target"]] = df[cfg["water"]["target"]].astype(int)
    return df
