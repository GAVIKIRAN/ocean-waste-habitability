"""Overview page: project summary, key metrics and how it works."""
from __future__ import annotations

import streamlit as st

from ui import get_config, get_db

from src.utils.app_helpers import load_json
from src.utils.common import resolve_path

cfg = get_config()
st.title("Ocean waste detection & water habitability", anchor=False)
st.caption("One platform for environmental teams and policymakers: detect underwater litter in images, video and live "
           "camera feeds, test whether water is potable and habitable, and track sites over time.")

reports = resolve_path(cfg["paths"]["reports"])
waste_m = load_json(reports / "waste" / "metrics.json") or {}
water_m = load_json(reports / "water" / "metrics.json") or {}
db = get_db()
n_scans, n_tests, n_reports = len(db.waste_history()), len(db.water_history()), len(db.report_history())

with st.container(horizontal=True):
    st.metric("Waste detector mAP@0.5", f"{waste_m['mAP50']:.3f}" if "mAP50" in waste_m else "Not evaluated", help="Test split · reports/waste/metrics.json", border=True)
    st.metric("Waste detector mAP@0.5:0.95", f"{waste_m['mAP50_95']:.3f}" if "mAP50_95" in waste_m else "Not evaluated", border=True)
    st.metric("Water model ROC-AUC", f"{water_m['roc_auc']:.3f}" if "roc_auc" in water_m else "Not evaluated", help="Held-out test split · reports/water/metrics.json", border=True)
    st.metric("Records logged", f"{n_scans + n_tests + n_reports}", help=f"{n_scans} waste scans, {n_tests} water tests, {n_reports} site reports", border=True)

if waste_m and any("synthetic" in src for src in (load_json(reports / "waste" / "dataset_stats.json") or {}).get("sources", [])):
    st.info("The waste detector shown here was trained on the bundled **synthetic** demo dataset. Its scores measure the pipeline, not real-world accuracy. "
            "See the README for training on TrashCan 1.0 or a Roboflow underwater-trash dataset.", icon=":material/science:")

st.subheader("What you can do", anchor=False)
cards = [
    ("Waste detection", ":material/delete:", "Upload an underwater image or video, or use a webcam. Optional dehazing and CLAHE enhancement, adjustable thresholds, CSV export.", "app_pages/waste_detection.py"),
    ("Water quality test", ":material/water_drop:", "Enter water-quality parameters (or upload a CSV) for a potable / non-potable verdict, WHO range checks and a SHAP explanation.", "app_pages/water_quality.py"),
    ("Site assessment", ":material/assignment:", "Combine an image and water readings for one site into an Ocean Health Report with a grade, recommendations and PDF export.", "app_pages/site_assessment.py"),
]
cols = st.columns(3)
for col, (title, icon, text, page) in zip(cols, cards):
    with col.container(border=True, height="stretch"):
        st.markdown(f"**{icon} {title}**")
        st.caption(text)
        st.page_link(page, label=f"Open {title.lower()}", icon=":material/arrow_forward:")

st.subheader("How it works", anchor=False)
st.code(
    """underwater image / video / webcam
   └─► enhancement (dark-channel dehaze, CLAHE, denoise)
        └─► YOLO11 detector ─► items, classes, area covered ─► waste cleanliness score ─┐
                                                                                         ├─► combined Ocean Health score, grade, actions
water parameters (pH, TDS, turbidity, ...)                                               │        └─► SQLite history ─► dashboard & map
   └─► median imputation ─► XGBoost ─► P(potable) + TreeSHAP ─┐                          │
   └─► WHO / EPA guideline range checks ──────────────────────┴─► habitability score ────┘""",
    language="text",
)
left, right = st.columns(2)
with left.container(border=True, height="stretch"):
    st.markdown("**Module 1: underwater waste detection**")
    st.markdown(f"- YOLO11 (Ultralytics), {len(cfg['waste']['classes'])} unified waste classes\n"
                "- Adam optimiser, mosaic / flip / HSV augmentation, early stopping\n"
                "- GPU FP16 inference for real-time webcam detection")
with right.container(border=True, height="stretch"):
    st.markdown("**Module 2: water potability & habitability**")
    st.markdown("- XGBoost tuned with GridSearchCV and Optuna (stratified 5-fold CV)\n"
                "- Class-imbalance handling (scale_pos_weight or SMOTE)\n"
                f"- Habitability = {cfg['water']['habitability']['model_weight']:.0%} model probability + "
                f"{1 - cfg['water']['habitability']['model_weight']:.0%} guideline compliance")
