"""Water-potability inference, WHO-guideline checks, habitability score and explanations.

Per-prediction explanations use XGBoost's native TreeSHAP (``pred_contribs=True``),
which gives exact SHAP values in log-odds space without depending on the ``shap``
package's model parsers (the ``shap`` package is only used for plotting).

Library use::

    wp = WaterPredictor.from_config(cfg)
    a = wp.assess({"ph": 7.1, "Hardness": 190, ...})
    a.label, a.probability, a.habitability_score, a.checks, a.explanation

CLI::

    python -m src.water.predict --csv samples.csv        # batch -> reports/water/predictions.csv
    python -m src.water.predict --ph 7.2 --Turbidity 3.1 # single sample (missing -> imputed)
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import xgboost as xgb

from src.utils.common import get_logger, load_config, resolve_path

log = get_logger(__name__)
STATUS_SCORE = {"green": 1.0, "amber": 0.5, "red": 0.0}


class WaterModelNotAvailableError(RuntimeError):
    """Raised when the trained XGBoost model/metadata are missing."""


# --------------------------------------------------------------------------- guidelines
def check_parameter(value: float | None, guideline: Mapping[str, Any]) -> str:
    """Traffic-light status of one value against ``{safe: [lo, hi], warn: [lo, hi]}``.

    Returns ``green`` (inside safe), ``amber`` (inside warn), ``red`` (outside warn)
    or ``unknown`` (missing value).
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "unknown"

    def inside(rng: list | tuple | None) -> bool:
        lo, hi = (rng or [None, None])
        return (lo is None or value >= lo) and (hi is None or value <= hi)

    if inside(guideline.get("safe")):
        return "green"
    if inside(guideline.get("warn")):
        return "amber"
    return "red"


def format_range(rng: list | tuple | None, unit: str = "") -> str:
    """``[6.5, 8.5]`` -> ``'6.5-8.5'``; ``[None, 4]`` -> ``'<= 4 ppm'``."""
    lo, hi = (rng or [None, None])
    u = f" {unit}" if unit else ""
    if lo is not None and hi is not None:
        return f"{lo:g}-{hi:g}{u}"
    if hi is not None:
        return f"<= {hi:g}{u}"
    if lo is not None:
        return f">= {lo:g}{u}"
    return "n/a"


def guideline_checks(sample: Mapping[str, Any], guidelines: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Return one row per parameter: value, safe range, status (green/amber/red/unknown)."""
    rows = []
    for key, g in guidelines.items():
        if key not in sample:
            continue
        v = sample.get(key)
        v = None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        rows.append({"parameter": key, "label": g.get("label", key), "value": v, "unit": g.get("unit", ""), "safe_range": format_range(g.get("safe"), g.get("unit", "")), "status": check_parameter(v, g)})
    return pd.DataFrame(rows, columns=["parameter", "label", "value", "unit", "safe_range", "status"])


def compliance_score(checks: pd.DataFrame) -> float:
    """Mean of green=1 / amber=0.5 / red=0 over known parameters (1.0 if none known)."""
    known = checks[checks["status"].isin(STATUS_SCORE)]
    return 1.0 if known.empty else float(known["status"].map(STATUS_SCORE).mean())


def habitability_score(p_potable: float, compliance: float, model_weight: float = 0.6) -> float:
    """0-100 score blending model probability and guideline compliance."""
    p = min(max(float(p_potable), 0.0), 1.0)
    c = min(max(float(compliance), 0.0), 1.0)
    return round(100.0 * (model_weight * p + (1 - model_weight) * c), 1)


def habitability_band(score: float) -> str:
    """Qualitative band for a 0-100 habitability score."""
    return "Good" if score >= 70 else "Moderate" if score >= 50 else "Poor" if score >= 30 else "Critical"


# --------------------------------------------------------------------------- model
@dataclass
class WaterAssessment:
    """Full result for one water sample."""

    label: str
    potable: bool
    probability: float
    habitability_score: float
    band: str
    compliance: float
    checks: pd.DataFrame
    contributions: pd.DataFrame
    base_value: float
    explanation: str
    inputs: dict[str, float | None] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "potable": self.potable,
            "probability": self.probability,
            "habitability_score": self.habitability_score,
            "band": self.band,
            "compliance": self.compliance,
            "inputs": self.inputs,
            "checks": self.checks.to_dict(orient="records"),
            "top_factors": self.contributions.head(5).to_dict(orient="records"),
            "explanation": self.explanation,
        }


class WaterPredictor:
    """Load the saved booster + metadata and score water samples."""

    def __init__(self, model_path: str | Path, meta_path: str | Path, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or load_config()
        model_path, meta_path = Path(model_path), Path(meta_path)
        if not model_path.exists() or not meta_path.exists():
            raise WaterModelNotAvailableError(f"Water model not found ({model_path}). Train it first: python -m src.water.train --mode demo")
        self.booster = xgb.Booster()
        self.booster.load_model(str(model_path))
        self.meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        self.features: list[str] = list(self.meta["features"])
        self.medians = np.asarray(self.meta["impute_medians"], dtype=float)
        self.threshold = float(self.meta.get("threshold", 0.5))
        self.guidelines: dict[str, Any] = self.cfg["water"]["guidelines"]

    @classmethod
    def from_config(cls, cfg: dict[str, Any] | None = None) -> "WaterPredictor":
        cfg = cfg or load_config()
        return cls(resolve_path(cfg["water"]["model_path"]), resolve_path(cfg["water"]["meta_path"]), cfg)

    # ---------------------------------------------------------------- helpers
    def prepare(self, df: pd.DataFrame) -> np.ndarray:
        """Select/order features, coerce to numeric, impute medians, apply scaler if any."""
        frame = pd.DataFrame({f: pd.to_numeric(df[f], errors="coerce") if f in df.columns else np.nan for f in self.features}, index=df.index)
        X = np.array(frame.to_numpy(dtype=float), dtype=float, copy=True)  # writable (pandas CoW returns read-only views)
        nan = np.isnan(X)
        X[nan] = np.take(self.medians, np.nonzero(nan)[1])
        if "scaler" in self.meta:
            X = (X - np.asarray(self.meta["scaler"]["mean"])) / np.asarray(self.meta["scaler"]["scale"])
        return X

    def _dmatrix(self, X: np.ndarray) -> xgb.DMatrix:
        return xgb.DMatrix(X, feature_names=self.booster.feature_names or None)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """P(potable) for each row of ``df``."""
        return self.booster.predict(self._dmatrix(self.prepare(df)))

    def contributions(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Exact TreeSHAP values (log-odds) per feature and the base value per row."""
        contrib = self.booster.predict(self._dmatrix(self.prepare(df)), pred_contribs=True)
        return contrib[:, :-1], contrib[:, -1]

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Batch prediction: input columns + probability, label, habitability score."""
        if df.empty:
            raise ValueError("Input CSV has no rows")
        missing_all = [f for f in self.cfg["water"]["features"] if f not in df.columns]
        if len(missing_all) == len(self.cfg["water"]["features"]):
            raise ValueError(f"None of the expected columns found. Expected: {self.cfg['water']['features']}")
        proba = self.predict_proba(df)
        out = df.copy()
        out["p_potable"] = np.round(proba, 4)
        out["prediction"] = np.where(proba >= self.threshold, "Potable", "Non-potable")
        w = self.cfg["water"]["habitability"]["model_weight"]
        out["habitability_score"] = [habitability_score(p, compliance_score(guideline_checks(r, self.guidelines)), w) for p, r in zip(proba, df.to_dict(orient="records"))]
        return out

    # ---------------------------------------------------------------- single
    def assess(self, sample: Mapping[str, Any]) -> WaterAssessment:
        """Predict + explain one sample (dict of parameter -> value; missing allowed)."""
        clean: dict[str, float | None] = {}
        for k, v in sample.items():
            try:
                clean[k] = None if v is None or v == "" else float(v)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Parameter '{k}' must be numeric, got {v!r}") from exc
        if all(clean.get(f) is None for f in self.features):
            raise ValueError("Provide at least one water-quality parameter")
        row = pd.DataFrame([{f: (np.nan if clean.get(f) is None else clean[f]) for f in self.features}])
        p = float(self.predict_proba(row)[0])
        shap_vals, base = self.contributions(row)
        contrib = pd.DataFrame({"feature": self.features, "value": [clean.get(f) for f in self.features], "contribution": shap_vals[0]})
        contrib["imputed"] = contrib["value"].isna()
        contrib = contrib.reindex(contrib["contribution"].abs().sort_values(ascending=False).index).reset_index(drop=True)
        checks = guideline_checks({f: clean.get(f) for f in self.features if f in clean}, self.guidelines)
        comp = compliance_score(checks)
        score = habitability_score(p, comp, self.cfg["water"]["habitability"]["model_weight"])
        potable = p >= self.threshold
        label = "Potable" if potable else "Non-potable"
        return WaterAssessment(label, potable, round(p, 4), score, habitability_band(score), round(comp, 3), checks, contrib, float(base[0]), self.explain_text(label, p, contrib, checks), clean)

    def explain_text(self, label: str, p: float, contrib: pd.DataFrame, checks: pd.DataFrame) -> str:
        """Plain-English explanation of the drivers behind a prediction."""
        g = self.guidelines
        fmt = lambda r: f"{g.get(r.feature, {}).get('label', r.feature)} ({'imputed' if r.imputed else f'{r.value:g}'})"  # noqa: E731
        up = [fmt(r) for r in contrib.itertuples() if r.contribution > 0.02][:3]
        down = [fmt(r) for r in contrib.itertuples() if r.contribution < -0.02][:3]
        parts = [f"The model classifies this sample as **{label}** (probability of being potable: {p:.0%})."]
        if down:
            parts.append("Parameters pushing towards *non-potable*: " + ", ".join(down) + ".")
        if up:
            parts.append("Parameters supporting *potable*: " + ", ".join(up) + ".")
        red = checks[checks["status"] == "red"]
        amber = checks[checks["status"] == "amber"]
        if not red.empty:
            parts.append("Outside WHO/EPA guideline limits: " + ", ".join(f"{r.label} = {r.value:g} (safe {r.safe_range})" for r in red.itertuples()) + ".")
        if not amber.empty:
            parts.append("Borderline: " + ", ".join(f"{r.label} = {r.value:g}" for r in amber.itertuples()) + ".")
        if red.empty and amber.empty and not checks.empty:
            parts.append("All provided parameters are within guideline ranges.")
        return " ".join(parts)


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default="reports/water/predictions.csv")
    for f in list(cfg["water"]["features"]) + list(cfg["water"]["optional_features"]):
        ap.add_argument(f"--{f}", type=float, default=None)
    args = ap.parse_args(argv)
    wp = WaterPredictor.from_config(cfg)
    if args.csv:
        res = wp.predict_df(pd.read_csv(args.csv))
        out = resolve_path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(out, index=False)
        print(res.head(20).to_string())
        log.info("%d predictions -> %s", len(res), out)
    else:
        sample = {f: getattr(args, f) for f in wp.features}
        a = wp.assess(sample)
        print(json.dumps(a.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
