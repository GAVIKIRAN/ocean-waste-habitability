"""Model performance: saved metrics, confusion matrices, PR/ROC curves, SHAP and training curves."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from ui import page_header, get_config

from src.utils.app_helpers import load_json
from src.utils.common import resolve_path

cfg = get_config()
page_header("Model performance", ":material/analytics:", "Metrics and plots saved by the evaluation scripts in `reports/`. Re-run `python -m src.waste.evaluate` / `python -m src.water.evaluate` to refresh.")
rep = resolve_path(cfg["paths"]["reports"])


def show_images(folder: Path, names: list[tuple[str, str]], per_row: int = 2) -> None:
    """Show the PNGs that exist, ``per_row`` per row, with captions."""
    found = [(folder / f, cap) for f, cap in names if (folder / f).exists()]
    if not found:
        st.caption("No plots saved yet.")
        return
    for i in range(0, len(found), per_row):
        cols = st.columns(per_row)
        for col, (p, cap) in zip(cols, found[i : i + per_row]):
            col.image(str(p), caption=cap, width="stretch", alt=cap)


module = st.segmented_control("Module", ["Waste detection (YOLO11)", "Water potability (XGBoost)"], default="Waste detection (YOLO11)")

if module == "Water potability (XGBoost)":
    m = load_json(rep / "water" / "metrics.json")
    if not m:
        st.info("Run `python -m src.water.train` then `python -m src.water.evaluate`.", title="No water evaluation yet", icon=":material/info:")
        st.stop()
    st.subheader("Held-out test set", anchor=False)
    with st.container(horizontal=True):
        for k, label in (("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"), ("roc_auc", "ROC-AUC")):
            st.metric(label, f"{m[k]:.3f}", border=True)
    st.caption(f"{m['n_test']} test samples · data: {m.get('data_source', 'unknown')} · tuner selected: {m.get('selected_tuner')}")
    cv = m.get("cv_metrics") or {}
    if cv:
        st.subheader("Stratified 5-fold cross-validation (training split)", anchor=False)
        st.dataframe(pd.DataFrame(cv).T.rename_axis("metric").reset_index(), hide_index=True,
                     column_config={"mean": st.column_config.NumberColumn("Mean", format="%.3f"), "std": st.column_config.NumberColumn("Std", format="%.3f")})
    with st.expander("Selected hyperparameters"):
        st.json(m.get("params", {}))
    st.subheader("Plots", anchor=False)
    show_images(rep / "water", [("confusion_matrix.png", "Confusion matrix"), ("roc_curve.png", "ROC curve"), ("pr_curve.png", "Precision-recall curve"),
                                ("feature_importance.png", "Feature importance (gain)"), ("shap_summary.png", "SHAP summary (beeswarm)"), ("shap_bar.png", "Global SHAP importance")])
else:
    m = load_json(rep / "waste" / "metrics.json")
    if not m:
        st.info("Run `python -m src.waste.train --mode demo` then `python -m src.waste.evaluate`.", title="No waste evaluation yet", icon=":material/info:")
        st.stop()
    stats = load_json(rep / "waste" / "dataset_stats.json") or {}
    summ = load_json(rep / "waste" / "training" / "train_summary.json") or {}
    st.subheader(f"Test split · {m['weights']} @ {m['imgsz']}px", anchor=False)
    with st.container(horizontal=True):
        for k, label in (("mAP50", "mAP@0.5"), ("mAP50_95", "mAP@0.5:0.95"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")):
            st.metric(label, f"{m[k]:.3f}", border=True)
    if summ:
        st.caption(f"Model {summ.get('model')} · {summ.get('mode')} mode · batch {summ.get('batch')} · {summ.get('device')} · trained in {summ.get('duration_min')} min")
    if stats:
        st.caption(f"Dataset: {stats.get('n_images')} images · split " + ", ".join(f"{k} {v['images']}" for k, v in stats.get("splits", {}).items()))
    if m.get("per_class"):
        st.subheader("Per-class metrics", anchor=False)
        st.dataframe(pd.DataFrame(m["per_class"]), hide_index=True,
                     column_config={c: st.column_config.ProgressColumn(c, min_value=0.0, max_value=1.0, format="%.3f") for c in ("precision", "recall", "f1", "mAP50", "mAP50_95")})
    results = rep / "waste" / "training" / "results.csv"
    if results.exists():
        st.subheader("Training curves", anchor=False)
        df = pd.read_csv(results)
        df.columns = [c.strip() for c in df.columns]
        c1, c2 = st.columns(2)
        loss_cols = [c for c in df.columns if "loss" in c and c.startswith(("train", "val"))]
        with c1.container(border=True):
            st.markdown("**Losses**")
            st.line_chart(df.set_index("epoch")[loss_cols], x_label="Epoch", y_label="Loss")
        with c2.container(border=True):
            st.markdown("**Validation mAP**")
            st.line_chart(df.set_index("epoch")[[c for c in df.columns if c.startswith("metrics/mAP")]], x_label="Epoch", y_label="mAP")
    st.subheader("Plots", anchor=False)
    show_images(rep / "waste", [("confusion_matrix_normalized.png", "Confusion matrix (normalised)"), ("confusion_matrix.png", "Confusion matrix"),
                                ("BoxPR_curve.png", "Precision-recall curves"), ("BoxF1_curve.png", "F1-confidence curves"),
                                ("BoxP_curve.png", "Precision-confidence"), ("BoxR_curve.png", "Recall-confidence")])
    if (rep / "waste" / "prediction_grid.png").exists():
        with st.expander("Qualitative predictions (ground truth vs prediction)"):
            st.image(str(rep / "waste" / "prediction_grid.png"), width="stretch", alt="Ground truth versus predicted boxes on test images")
