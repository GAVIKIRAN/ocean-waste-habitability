"""Ocean Waste Detection and Water Habitability Test - Streamlit entry point.

Run from the project root:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

from ui import get_config, render_sidebar_status

st.set_page_config(page_title="Ocean Health Platform", page_icon=":material/water:", layout="wide")

PAGES = [
    st.Page("app_pages/home.py", title="Overview", icon=":material/home:", default=True,
            help="Project summary, key model metrics and how the two modules fit together."),
    st.Page("app_pages/waste_detection.py", title="Waste detection", icon=":material/delete:",
            help="Detect marine litter in images, videos, webcam snapshots or a live camera with YOLO11. Export annotated images and CSV."),
    st.Page("app_pages/water_quality.py", title="Water quality test", icon=":material/water_drop:",
            help="Predict potability from water parameters or a CSV, with WHO range checks and a SHAP explanation."),
    st.Page("app_pages/site_assessment.py", title="Site assessment", icon=":material/assignment:",
            help="Combine an image and water readings for one site into an Ocean Health Report with PDF export."),
    st.Page("app_pages/dashboard.py", title="Dashboard", icon=":material/monitoring:",
            help="History of logged scans, tests and reports: trends, waste mix, potability ratio and a site map."),
    st.Page("app_pages/model_performance.py", title="Model performance", icon=":material/analytics:",
            help="Saved evaluation metrics, confusion matrices, PR/ROC curves, SHAP plots and training curves."),
    st.Page("app_pages/about.py", title="About", icon=":material/info:",
            help="Methodology, data sources, limitations and future work."),
]

page = st.navigation(PAGES)
with st.sidebar:
    st.markdown("**Ocean Health Platform**")
    st.caption("YOLO11 waste detection · XGBoost water habitability")
render_sidebar_status(get_config())
page.run()
