"""OceanWatch overview, built from live local model and history state."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ui import device_label, get_db, get_detector, get_water_predictor, page_header

db = get_db()
waste = db.waste_history()
water = db.water_history()
reports = db.report_history()
detector = get_detector()
water_model = get_water_predictor()

n_scans = len(waste)
n_items = int(waste["n_detections"].sum()) if not waste.empty else 0
last_scan = "No saved scans yet" if waste.empty else waste["timestamp"].iloc[-1].strftime("%d %b, %H:%M UTC")
model_status = "Unavailable" if detector is None else ("Demo fallback" if detector.is_demo else "Loaded")
totals = db.waste_class_totals()


def daily(df: pd.DataFrame, col: str | None = None, days: int = 14) -> list[float]:
    """Per-day totals (or row counts) for the last ``days`` days, zero-filled, for metric sparklines."""
    end = pd.Timestamp.now(tz=df["timestamp"].dt.tz if not df.empty else None).normalize()
    idx = pd.date_range(end - pd.Timedelta(days=days - 1), end, freq="D")
    if df.empty:
        return [0.0] * days
    s = df.set_index("timestamp")[col] if col else df.set_index("timestamp").assign(n=1)["n"]
    return s.resample("D").sum().reindex(idx, fill_value=0).astype(float).tolist()


page_header(
    "Ocean monitoring",
    ":material/waves:",
    "AI-powered underwater waste intelligence for Survey Zone 01. Every figure comes from records saved on this machine.",
)

with st.container(horizontal=True):
    st.metric("Objects detected", f"{n_items:,}", icon=":material/scatter_plot:", border=True,
              chart_data=daily(waste, "n_detections"), chart_type="bar", help="Objects across all saved waste scans. Sparkline: last 14 days.")
    st.metric("Images analysed", f"{n_scans:,}", icon=":material/photo_library:", border=True,
              chart_data=daily(waste), chart_type="bar", help="Saved scans. Sparkline: last 14 days.")
    st.metric("Water tests", f"{len(water):,}", icon=":material/water_drop:", border=True,
              chart_data=daily(water), chart_type="bar", help="Saved potability tests. Sparkline: last 14 days.")
    st.metric("Site reports", f"{len(reports):,}", icon=":material/assignment:", border=True,
              chart_data=daily(reports), chart_type="bar", help="Saved Ocean Health Reports. Sparkline: last 14 days.")

main, side = st.columns([2, 1], gap="medium")

with main:
    with st.container(border=True, height="stretch"):
        with st.container(horizontal=True, vertical_alignment="center"):
            st.subheader("Detection activity", anchor=False, icon=":material/bar_chart:")
            period = st.segmented_control("Period", ["7 days", "30 days"], default="7 days", key="overview_period",
                                          label_visibility="collapsed", width="content")
        st.caption("Objects detected per day across saved scans")
        if waste.empty:
            st.info("Run a detection and save it to start the activity timeline.", title="No activity yet", icon=":material/insights:")
        else:
            days = 30 if period == "30 days" else 7
            st.bar_chart(pd.Series(daily(waste, "n_detections", days),
                                   index=pd.date_range(end=pd.Timestamp.now().normalize(), periods=days, freq="D"), name="Objects"),
                         height=260, y_label="Objects", x_label="", alt=f"Objects detected per day over the last {days} days")

with side:
    with st.container(border=True):
        st.subheader("Start a workflow", anchor=False, icon=":material/bolt:")
        st.page_link("app_pages/waste_detection.py", label="Detect waste in imagery", icon=":material/radar:", width="stretch")
        st.page_link("app_pages/water_quality.py", label="Test water quality", icon=":material/water_drop:", width="stretch")
        st.page_link("app_pages/site_assessment.py", label="Create a site report", icon=":material/assignment:", width="stretch")
        st.page_link("app_pages/dashboard.py", label="Browse history", icon=":material/monitoring:", width="stretch")

    with st.container(border=True):
        st.subheader("System", anchor=False, icon=":material/memory:")
        waste_color = "green" if model_status == "Loaded" else ("orange" if detector is not None else "red")
        st.markdown(
            f"YOLO11 waste detector  \n:{waste_color}-badge[{model_status}]  \n"
            f"XGBoost water model  \n:{'green' if water_model is not None else 'red'}-badge[{'Loaded' if water_model is not None else 'Unavailable'}]"
        )
        st.caption(f"Runtime: {device_label()} · Last scan: {last_scan}")

recent, intel = st.columns([2, 1], gap="medium")

with recent:
    with st.container(border=True, height="stretch"):
        st.subheader("Recent detections", anchor=False, icon=":material/history:")
        if waste.empty:
            st.caption("Saved scans will appear here.")
        else:
            rows = []
            for row in waste.tail(5).iloc[::-1].itertuples():
                classes = json.loads(row.class_counts)
                rows.append({
                    "Main waste type": next(iter(classes), "marine_debris").replace("_", " ").capitalize(),
                    "Objects": int(row.n_detections),
                    "Confidence": row.mean_confidence if pd.notna(row.mean_confidence) else None,
                    "Site": row.site_name or "Unnamed site",
                    "Saved": row.timestamp,
                })
            st.dataframe(
                rows, hide_index=True, alt="Five most recent saved detections",
                column_config={
                    "Confidence": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                    "Saved": st.column_config.DatetimeColumn(format="D MMM, HH:mm"),
                },
            )

with intel:
    with st.container(border=True, height="stretch"):
        st.subheader("Insights", anchor=False, icon=":material/lightbulb:")
        top = next(iter(totals), None)
        highest = waste["mean_confidence"].max() if not waste.empty and waste["mean_confidence"].notna().any() else None
        st.metric("Most frequent waste", top.replace("_", " ").capitalize() if top else "No data")
        st.metric("Best scan confidence", f"{highest:.0%}" if highest is not None else "No data")
