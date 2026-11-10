"""Evaluate the XGBoost water model on the held-out test split.

Saves to ``reports/water/``: ``metrics.json``, ``confusion_matrix.png``, ``roc_curve.png``,
``pr_curve.png``, ``feature_importance.png`` (+ csv), ``shap_summary.png``, ``shap_bar.png``.

Usage::

    python -m src.water.evaluate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

from src.utils.common import ensure_dir, get_logger, load_config, resolve_path  # noqa: E402
from src.utils.plots import save_confusion_matrix  # noqa: E402
from src.water.predict import WaterPredictor  # noqa: E402
from src.water.train import evaluate_binary  # noqa: E402

log = get_logger(__name__)
plt.style.use("dark_background")


def evaluate(cfg: dict[str, Any]) -> dict[str, Any]:
    """Compute and save all water-model metrics and plots."""
    w = cfg["water"]
    test_path = resolve_path(w["test_split_path"])
    if not test_path.exists():
        raise FileNotFoundError(f"{test_path} missing - run python -m src.water.train first")
    wp = WaterPredictor.from_config(cfg)
    df = pd.read_csv(test_path)
    y = df[w["target"]].to_numpy().astype(int)
    proba = wp.predict_proba(df)
    pred = (proba >= wp.threshold).astype(int)
    rep = ensure_dir(Path(cfg["paths"]["reports"]) / "water")

    metrics: dict[str, Any] = {"n_test": int(len(df)), **evaluate_binary(y, proba, wp.threshold)}
    cm = confusion_matrix(y, pred, labels=[0, 1])
    metrics["confusion_matrix"] = cm.tolist()
    metrics["cv_metrics"] = wp.meta.get("cv_metrics", {})
    metrics["selected_tuner"] = wp.meta.get("selected")
    metrics["params"] = wp.meta.get("params")
    metrics["data_source"] = wp.meta.get("data_source")
    save_confusion_matrix(cm, ["Non-potable", "Potable"], rep / "confusion_matrix.png", "Water potability - confusion matrix")

    fpr, tpr, _ = roc_curve(y, proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color="#2ec4b6", lw=2, label=f"ROC-AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#888")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(rep / "roc_curve.png", dpi=120)
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y, proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(rec, prec, color="#ff9f1c", lw=2)
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve")
    fig.tight_layout()
    fig.savefig(rep / "pr_curve.png", dpi=120)
    plt.close(fig)

    gain = wp.booster.get_score(importance_type="gain")
    imp = pd.DataFrame({"feature": wp.features, "gain": [gain.get(f, gain.get(f"f{i}", 0.0)) for i, f in enumerate(wp.features)]})
    imp["gain_pct"] = (100 * imp["gain"] / max(imp["gain"].sum(), 1e-9)).round(2)
    imp = imp.sort_values("gain", ascending=True)
    imp.to_csv(rep / "feature_importance.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(imp["feature"], imp["gain_pct"], color="#2ec4b6")
    ax.set(xlabel="Importance (% of total gain)", title="XGBoost feature importance")
    fig.tight_layout()
    fig.savefig(rep / "feature_importance.png", dpi=120)
    plt.close(fig)
    metrics["feature_importance"] = dict(zip(imp["feature"], imp["gain_pct"]))

    shap_vals, _ = wp.contributions(df)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    metrics["mean_abs_shap"] = {f: round(float(v), 4) for f, v in zip(wp.features, mean_abs)}
    X_disp = pd.DataFrame(wp.prepare(df), columns=wp.features)
    try:
        import shap

        plt.figure()
        shap.summary_plot(shap_vals, X_disp, show=False, plot_size=(7, 4.5))
        plt.title("SHAP summary (impact on log-odds of potable)")
        plt.tight_layout()
        plt.savefig(rep / "shap_summary.png", dpi=120)
        plt.close("all")
    except Exception as exc:  # noqa: BLE001
        log.warning("shap summary plot failed (%s) - bar chart only", exc)
    order = np.argsort(mean_abs)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(np.array(wp.features)[order], mean_abs[order], color="#ff9f1c")
    ax.set(xlabel="mean |SHAP value|", title="Global SHAP importance")
    fig.tight_layout()
    fig.savefig(rep / "shap_bar.png", dpi=120)
    plt.close(fig)

    (rep / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float), encoding="utf-8")
    log.info("Water test metrics: %s", {k: metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")})
    return metrics


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    m = evaluate(load_config(args.config))
    print(json.dumps({k: m[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix")}, indent=2))


if __name__ == "__main__":
    main()
