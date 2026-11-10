"""Monitoring dashboard: history from SQLite, trends, waste mix, potability ratio, site map."""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from ui import get_db, page_header

from src.utils.plots import SERIES, class_counts_bar, site_map, trend_chart, two_way_bar

page_header("History", ":material/monitoring:", "Trends, waste mix, water screening and survey locations from the records saved by this installation.")

db = get_db()
waste, water, reports = db.waste_history(), db.water_history(), db.report_history()

if waste.empty and water.empty and reports.empty:
    st.info("Run a detection, a water test or a site assessment and save it; records will appear here.", title="No records yet", icon=":material/inbox:")
    st.stop()

# ---- filters (one row, above the charts)
all_ts = pd.concat([d["timestamp"] for d in (waste, water, reports) if not d.empty])
with st.container(border=True):
    with st.container(horizontal=True, vertical_alignment="bottom"):
        lo, hi = all_ts.min().date(), all_ts.max().date()
        rng = st.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
        sites = sorted({s for d in (waste, water, reports) if not d.empty for s in d["site_name"].dropna().unique()})
        pick = st.multiselect("Sites", sites, placeholder="All sites")


def filt(d: pd.DataFrame) -> pd.DataFrame:
    if d.empty:
        return d
    start, end = (rng if isinstance(rng, tuple) and len(rng) == 2 else (lo, hi))
    m = (d["timestamp"].dt.date >= start) & (d["timestamp"].dt.date <= end)
    if pick:
        m &= d["site_name"].isin(pick)
    return d[m]


waste, water, reports = filt(waste), filt(water), filt(reports)

with st.container(horizontal=True):
    st.metric("Waste scans", len(waste), border=True, icon=":material/photo_library:")
    st.metric("Items detected", int(waste["n_detections"].sum()) if not waste.empty else 0, border=True, icon=":material/scatter_plot:")
    st.metric("Water tests", len(water), border=True, icon=":material/water_drop:")
    st.metric("Potable share", f"{water['potable'].mean():.0%}" if not water.empty else "–", border=True, icon=":material/verified:")
    st.metric("Mean site score", f"{reports['combined_score'].mean():.0f}" if not reports.empty else "–", border=True, icon=":material/eco:")

left, right = st.columns(2)
with left.container(border=True):
    if not water.empty:
        st.plotly_chart(trend_chart(water, "timestamp", "habitability_score", "Water habitability over time", "Habitability (0-100)", SERIES[0]), theme=None, alt="Habitability score over time")
    else:
        st.caption("No water tests in range.")
with right.container(border=True):
    if not waste.empty:
        st.plotly_chart(trend_chart(waste, "timestamp", "n_detections", "Waste items per scan over time", "Items per scan", SERIES[1]), theme=None, alt="Items per scan over time")
    else:
        st.caption("No waste scans in range.")

left, right = st.columns(2)
with left.container(border=True):
    totals: dict[str, int] = {}
    for raw in waste.get("class_counts", []):
        for k, v in json.loads(raw).items():
            totals[str(k)] = totals.get(str(k), 0) + int(v)
    if totals:
        st.plotly_chart(class_counts_bar(totals, "Waste-type distribution"), theme=None, alt="Total detected items per waste type")
    else:
        st.caption("No waste detected in range.")
with right.container(border=True):
    if not water.empty:
        n_p = int(water["potable"].sum())
        st.plotly_chart(two_way_bar(["Potable", "Non-potable"], [n_p, len(water) - n_p], "Potable vs non-potable tests"), theme=None, alt="Count of potable and non-potable tests")
    else:
        st.caption("No water tests in range.")

geo = db.sites()
geo = geo[geo["timestamp"].isin(pd.concat([d["timestamp"] for d in (waste, water, reports) if not d.empty]))] if not geo.empty else geo
with st.container(border=True):
    if geo.empty:
        st.caption("No geo-tagged records in range. Tick *Add GPS coordinates* when saving a record to see it on the map.")
    else:
        st.plotly_chart(site_map(geo), theme=None, alt="Map of logged sites coloured by score band")

with st.expander("Raw records"):
    which = st.segmented_control("Table", ["Site reports", "Water tests", "Waste scans"], default="Site reports")
    table = {"Site reports": reports.drop(columns=["report"], errors="ignore"), "Water tests": water, "Waste scans": waste}[which or "Site reports"]
    st.dataframe(table, hide_index=True)
    st.download_button("Download CSV", table.to_csv(index=False).encode(), f"{(which or 'records').lower().replace(' ', '_')}.csv", "text/csv", icon=":material/download:")

with st.popover("Reset history", icon=":material/delete_forever:"):
    st.warning("This permanently deletes all logged records.")
    if st.button("Delete all records", type="primary"):
        db.clear()
        st.rerun()
