"""Train the XGBoost water-potability classifier.

Pipeline: median imputation -> (optional scaling) -> (optional SMOTE) -> XGBClassifier.
Class imbalance: ``scale_pos_weight`` (default) or SMOTE (``water.imbalance`` in config).
Model selection: stratified 5-fold CV on the training split, tuned with GridSearchCV,
Optuna (TPE / Bayesian) or both (best CV ROC-AUC wins). Final metrics on a held-out
stratified test split (saved for ``evaluate.py``).

Artefacts: ``models/xgb_water.json`` (native XGBoost JSON) + ``models/xgb_water_meta.json``
(feature list, imputation medians, scaler, threshold, params, CV/test metrics).
No pickles are used, so the model is portable and safe to load.

Usage::

    python -m src.water.train --mode demo            # ~15 Optuna trials + small grid
    python -m src.water.train --mode full --method both --device cuda
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data.water_data import available_features, load_water_data
from src.utils.common import get_logger, load_config, resolve_path, set_seed

log = get_logger(__name__)
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def make_pipeline(params: dict[str, Any], cfg: dict[str, Any], scale_pos_weight: float, device: str = "cpu", n_jobs: int = -1) -> Pipeline:
    """Build the imputer -> [scaler] -> [SMOTE] -> XGBoost pipeline."""
    w = cfg["water"]
    seed = cfg["project"]["seed"]
    steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median").set_output(transform="pandas"))]
    if w.get("scale_features"):
        steps.append(("scaler", StandardScaler().set_output(transform="pandas")))
    if w["imbalance"] == "smote":
        steps.append(("smote", SMOTE(random_state=seed)))
    clf_params = {**w["base_params"], **params, "random_state": seed, "n_jobs": n_jobs, "device": device}
    if w["imbalance"] == "scale_pos_weight":
        clf_params["scale_pos_weight"] = scale_pos_weight
    steps.append(("clf", XGBClassifier(**clf_params)))
    return Pipeline(steps)


def tune_optuna(X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any], spw: float, device: str, n_trials: int, cv: StratifiedKFold) -> tuple[dict[str, Any], float]:
    """Bayesian (TPE) search over the space in ``water.tuning.optuna_space``."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    space = cfg["water"]["tuning"]["optuna_space"]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", *space["max_depth"]),
            "learning_rate": trial.suggest_float("learning_rate", *space["learning_rate"], log=True),
            "n_estimators": trial.suggest_int("n_estimators", *space["n_estimators"], step=50),
            "subsample": trial.suggest_float("subsample", *space["subsample"]),
            "colsample_bytree": trial.suggest_float("colsample_bytree", *space["colsample_bytree"]),
            "gamma": trial.suggest_float("gamma", *space["gamma"]),
            "min_child_weight": trial.suggest_int("min_child_weight", *space["min_child_weight"]),
            "reg_lambda": trial.suggest_float("reg_lambda", *space["reg_lambda"]),
        }
        return float(cross_val_score(make_pipeline(params, cfg, spw, device), X, y, cv=cv, scoring="roc_auc").mean())

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=cfg["project"]["seed"]))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info("Optuna: best CV ROC-AUC=%.4f after %d trials", study.best_value, len(study.trials))
    return study.best_params, float(study.best_value)


def tune_grid(X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any], spw: float, device: str, grid: dict[str, list], cv: StratifiedKFold) -> tuple[dict[str, Any], float]:
    """Exhaustive GridSearchCV over ``grid`` (keys are XGBoost params)."""
    gs = GridSearchCV(make_pipeline({}, cfg, spw, device), {f"clf__{k}": v for k, v in grid.items()}, scoring="roc_auc", cv=cv, n_jobs=1, refit=False)
    gs.fit(X, y)
    best = {k.replace("clf__", ""): v for k, v in gs.best_params_.items()}
    log.info("GridSearchCV: best CV ROC-AUC=%.4f over %d candidates", gs.best_score_, len(gs.cv_results_["params"]))
    return best, float(gs.best_score_)


def evaluate_binary(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Accuracy / precision / recall / F1 / ROC-AUC for a probability vector."""
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4) if len(np.unique(y_true)) > 1 else float("nan"),
    }


def save_model(pipe: Pipeline, features: list[str], cfg: dict[str, Any], meta_extra: dict[str, Any], model_path: Path | None = None, meta_path: Path | None = None) -> tuple[Path, Path]:
    """Persist the booster as native JSON and the preprocessing state as a JSON sidecar."""
    model_path = model_path or resolve_path(cfg["water"]["model_path"])
    meta_path = meta_path or resolve_path(cfg["water"]["meta_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    pipe.named_steps["clf"].save_model(str(model_path))
    meta: dict[str, Any] = {
        "features": features,
        "impute_medians": [float(v) for v in pipe.named_steps["imputer"].statistics_],
        "threshold": float(cfg["water"]["threshold"]),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **meta_extra,
    }
    if "scaler" in pipe.named_steps:
        sc = pipe.named_steps["scaler"]
        meta["scaler"] = {"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()}
    meta_path.write_text(json.dumps(meta, indent=2, default=float), encoding="utf-8")
    return model_path, meta_path


def train(cfg: dict[str, Any], mode: str = "demo", method: str | None = None, device: str | None = None, data_path: str | None = None, n_trials: int | None = None) -> dict[str, Any]:
    """Full training run. Returns the metadata dict that was saved."""
    w = cfg["water"]
    seed = cfg["project"]["seed"]
    set_seed(seed)
    device = device or w.get("device", "cpu")
    if device == "cuda":
        try:
            import torch

            if not torch.cuda.is_available():
                log.warning("CUDA requested but unavailable - using CPU")
                device = "cpu"
        except ImportError:
            device = "cpu"
    print(f"[water/train] Device: {device}", flush=True)

    df = load_water_data(cfg, data_path)
    features = available_features(df, cfg)
    X, y = df[features].apply(pd.to_numeric, errors="coerce"), df[w["target"]].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=w["test_size"], stratify=y, random_state=seed)
    test_path = resolve_path(w["test_split_path"])
    test_path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([X_te, y_te], axis=1).to_csv(test_path, index=False)

    n_pos, n_neg = int(y_tr.sum()), int((y_tr == 0).sum())
    spw = n_neg / max(n_pos, 1)
    log.info("Rows=%d features=%s | train pos=%d neg=%d (scale_pos_weight=%.2f, imbalance=%s)", len(df), features, n_pos, n_neg, spw, w["imbalance"])
    cv = StratifiedKFold(n_splits=w["cv_folds"], shuffle=True, random_state=seed)

    method = method or w["tuning"]["method"]
    t0 = time.time()
    candidates: dict[str, tuple[dict[str, Any], float]] = {}
    if method in ("grid", "both"):
        grid = w["tuning"]["demo_grid"] if mode == "demo" else w["tuning"]["grid"]
        candidates["grid"] = tune_grid(X_tr, y_tr, cfg, spw, device, grid, cv)
    if method in ("optuna", "both"):
        trials = n_trials or (w["tuning"]["demo_n_trials"] if mode == "demo" else w["tuning"]["n_trials"])
        candidates["optuna"] = tune_optuna(X_tr, y_tr, cfg, spw, device, trials, cv)
    if not candidates:
        candidates["default"] = ({"max_depth": 6, "learning_rate": 0.1, "n_estimators": 300}, float("nan"))
    best_method = max(candidates, key=lambda k: -1 if np.isnan(candidates[k][1]) else candidates[k][1])
    best_params, best_cv = candidates[best_method]
    log.info("Selected %s params (CV ROC-AUC %.4f): %s", best_method, best_cv, best_params)

    pipe = make_pipeline(best_params, cfg, spw, device)
    cvres = cross_validate(pipe, X_tr, y_tr, cv=cv, scoring=SCORING)
    cv_metrics = {m: {"mean": round(float(np.mean(cvres[f"test_{m}"])), 4), "std": round(float(np.std(cvres[f"test_{m}"])), 4)} for m in SCORING}
    pipe.fit(X_tr, y_tr)
    test_metrics = evaluate_binary(y_te.to_numpy(), pipe.predict_proba(X_te)[:, 1], w["threshold"])
    log.info("CV: %s", {k: v["mean"] for k, v in cv_metrics.items()})
    log.info("Test: %s", test_metrics)

    source = str(df.attrs.get("source", "unknown"))
    meta = {
        "mode": mode,
        "device": device,
        "tuning": {k: {"params": v[0], "cv_roc_auc": round(v[1], 4)} for k, v in candidates.items()},
        "selected": best_method,
        "params": best_params,
        "imbalance": w["imbalance"],
        "scale_pos_weight": round(spw, 4),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "class_balance": {"potable": n_pos, "non_potable": n_neg},
        "cv_metrics": cv_metrics,
        "test_metrics": test_metrics,
        "train_seconds": round(time.time() - t0, 1),
        "data_source": source,
    }
    mp, metap = save_model(pipe, features, cfg, meta)
    log.info("Saved model -> %s, meta -> %s", mp, metap)
    return meta


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["demo", "full"], default="demo")
    ap.add_argument("--method", choices=["optuna", "grid", "both", "none"], default=None)
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None)
    ap.add_argument("--data", default=None, help="CSV path (default: water.data_file)")
    ap.add_argument("--trials", type=int, default=None)
    ap.add_argument("--imbalance", choices=["scale_pos_weight", "smote", "none"], default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if args.imbalance:
        cfg["water"]["imbalance"] = args.imbalance
    meta = train(cfg, args.mode, args.method, args.device, args.data, args.trials)
    print(json.dumps({"selected": meta["selected"], "cv": {k: v["mean"] for k, v in meta["cv_metrics"].items()}, "test": meta["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
