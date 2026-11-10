"""Ocean Waste Detection and Water Habitability Test - Streamlit entry point.

Run from the project root:  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import streamlit as st

from ui import LOGO, LOGO_ICON, get_config, render_sidebar_status

st.set_page_config(page_title="OceanWatch AI", page_icon=":material/waves:", layout="wide")
st.logo(str(LOGO), size="large", icon_image=str(LOGO_ICON))

PAGES = {
    "Workspace": [
        st.Page("app_pages/home.py", title="Overview", icon=":material/space_dashboard:", default=True,
                help="Project summary, key model metrics and how the two modules fit together."),
        st.Page("app_pages/waste_detection.py", title="Live detection", icon=":material/radar:",
                help="Detect marine litter in images, videos, webcam snapshots or a live camera with YOLO11. Export annotated images and CSV."),
        st.Page("app_pages/water_quality.py", title="Water quality", icon=":material/water_drop:",
                help="Predict potability from water parameters or a CSV, with WHO range checks and a SHAP explanation."),
        st.Page("app_pages/site_assessment.py", title="Site reports", icon=":material/assignment:",
                help="Combine an image and water readings for one site into an Ocean Health Report with PDF export."),
    ],
    "Insights": [
        st.Page("app_pages/dashboard.py", title="History", icon=":material/monitoring:",
                help="History of logged scans, tests and reports: trends, waste mix, potability ratio and a site map."),
        st.Page("app_pages/model_performance.py", title="Model performance", icon=":material/analytics:",
                help="Saved evaluation metrics, confusion matrices, PR/ROC curves, SHAP plots and training curves."),
        st.Page("app_pages/about.py", title="About", icon=":material/info:",
                help="Methodology, data sources, limitations and future work."),
    ],
}

page = st.navigation(PAGES)
render_sidebar_status(get_config())
page.run()
