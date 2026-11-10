"""Combined site assessment: one image + water parameters -> Ocean Health Report (PDF)."""
from __future__ import annotations

import re

import streamlit as st

from ui import get_config, get_db, get_detector, get_water_predictor, site_inputs, status_badge

from src.utils.app_helpers import bgr_to_rgb, decode_image, validate_coordinates, validate_water_inputs
from src.utils.plots import class_counts_bar, parameter_range_chart, score_gauge
from src.utils.report import build_report, report_to_pdf, status_text

cfg = get_config()
g = cfg["water"]["guidelines"]
st.title("Combined site assessment", anchor=False)
st.caption("Survey one location: an underwater image for litter plus water-quality readings produce a single Ocean Health Report with a grade and recommended actions.")

det, wp = get_detector(), get_water_predictor()
if det is None and wp is None:
    st.warning("Neither model is available. Train at least one model (`scripts\\run_all.bat demo`).", title="No models", icon=":material/model_training:")
    st.stop()

with st.form("site_form"):
    st.markdown("**1 · Site**")
    name, lat, lon = site_inputs("site", required_name=True, default_geo=True)
    notes = st.text_area("Field notes (optional)", placeholder="Visibility, weather, depth, observers...")
    left, right = st.columns(2)
    with left:
        st.markdown("**2 · Underwater image**" + ("" if det else " · *waste model unavailable*"))
        img_file = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "webp"], disabled=det is None)
        enhance_on = st.toggle("Underwater enhancement", False, help="Helps dark, murky footage; leave off for clear photos.")
        conf = st.slider("Confidence threshold", 0.05, 0.95, float(cfg["waste"]["inference"]["conf"]), 0.05)
        high_recall = st.toggle("High-recall mode", False, help="Also scans zoomed-in tiles: finds small and densely packed items (slower).")
    with right:
        st.markdown("**3 · Water parameters**" + ("" if wp else " · *water model unavailable*"))
        include_water = st.toggle("Include water test", value=wp is not None, disabled=wp is None)
        values = {}
        cols = st.columns(2)
        for i, f in enumerate(cfg["water"]["features"]):
            gi = g[f]
            values[f] = cols[i % 2].number_input(f"{gi['label']}" + (f" ({gi['unit']})" if gi["unit"] else ""), float(gi["slider"][0]), float(gi["slider"][1]) * 10, float(gi["default"]), format="%.4g", key=f"site_{f}")
    submitted = st.form_submit_button("Generate Ocean Health Report", type="primary", icon=":material/assignment:")

if submitted:
    errors = []
    if not name:
        errors.append("Site name is required")
    try:
        lat, lon = validate_coordinates(lat, lon)
    except ValueError as exc:
        errors.append(str(exc))
    if img_file is None and not include_water:
        errors.append("Provide an image, water parameters, or both")
    clean, werr = validate_water_inputs(values, g) if include_water else ({}, [])
    errors += werr
    if errors:
        st.error("\n".join(f"- {e}" for e in errors), title="Please fix the inputs", icon=":material/error:")
    else:
        with st.status("Assessing site...", expanded=False) as status:
            detection = assessment = None
            img = None
            if img_file is not None and det is not None:
                status.update(label="Detecting waste...")
                try:
                    img = decode_image(img_file.getvalue())
                    detection = det.predict(img, conf=conf, enhance_input=enhance_on, source=img_file.name,
                                            tiles=int(cfg["waste"]["inference"].get("high_recall_tiles", 3)) if high_recall else 1)
                except ValueError as exc:
                    st.error(f"Image error: {exc}")
            if include_water and wp is not None:
                status.update(label="Assessing water quality...")
                assessment = wp.assess(clean)
            if detection is None and assessment is None:
                status.update(label="Nothing to assess", state="error")
                st.stop()
            report = build_report(cfg, name, lat, lon, detection, assessment, notes)
            status.update(label="Building PDF...")
            pdf = report_to_pdf(report, detection.annotated if detection else None)
            get_db().log_report(name, report.combined_score, report.grade, report.waste_score, report.water_score, lat, lon, report.summary())
            status.update(label="Report ready", state="complete")
        st.session_state["site_report"] = (report, detection, assessment, pdf)

if "site_report" in st.session_state:
    report, detection, assessment, pdf = st.session_state["site_report"]
    st.divider()
    st.subheader(f"Ocean Health Report · {report.site_name}", anchor=False)
    band = "green" if report.combined_score >= 70 else "amber" if report.combined_score >= 50 else "red"
    st.markdown(f"{status_badge(band, f'Grade {report.grade} · {status_text(report.combined_score)}')} &nbsp; :gray-badge[{report.timestamp}]")
    c0, c1, c2 = st.columns(3)
    with c0.container(border=True):
        st.plotly_chart(score_gauge(report.combined_score, "Combined ocean health"), theme=None, alt=f"Combined score {report.combined_score:.0f}")
    c1.metric("Waste cleanliness", "–" if report.waste_score is None else f"{report.waste_score:.0f}/100", help=f"{report.n_detections} item(s) detected", border=True)
    c2.metric("Water habitability", "–" if report.water_score is None else f"{report.water_score:.0f}/100", help=report.water_label or "no water test", border=True)
    left, right = st.columns(2)
    if detection is not None:
        with left.container(border=True):
            st.image(bgr_to_rgb(detection.annotated), caption=f"{detection.n} item(s) detected", width="stretch", alt="Annotated site image")
            if detection.class_counts:
                st.plotly_chart(class_counts_bar(detection.class_counts), theme=None, alt="Items per class")
    if assessment is not None:
        with right.container(border=True):
            st.markdown(assessment.explanation)
            st.plotly_chart(parameter_range_chart(assessment.checks, g), theme=None, alt="Parameters vs guideline limits")
    with st.container(border=True):
        st.markdown("**Recommendations**")
        st.markdown("\n".join(f"{i}. {r}" for i, r in enumerate(report.recommendations, 1)))
    fname = re.sub(r"[^A-Za-z0-9_-]+", "_", report.site_name).strip("_") or "site"
    st.download_button("Download PDF report", pdf, f"ocean_health_report_{fname}.pdf", "application/pdf", type="primary", icon=":material/picture_as_pdf:")
