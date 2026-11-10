"""Water quality test page: single-sample form and CSV batch scoring."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from ui import get_config, get_db, get_water_predictor, site_inputs, status_badge, train_first

from src.utils.app_helpers import df_to_csv_bytes, read_water_csv, validate_coordinates, validate_water_inputs
from src.utils.plots import parameter_range_chart, probability_gauge, score_gauge, shap_waterfall
from src.water.predict import WaterAssessment

cfg = get_config()
g = cfg["water"]["guidelines"]
st.title("Water quality test", anchor=False)
st.caption("Predict whether water is potable from its physico-chemical parameters, check each one against WHO/EPA guideline ranges, and see which parameters drove the result.")

wp = get_water_predictor()
if wp is None:
    train_first("water")
    st.stop()

if str(wp.meta.get("data_source", "")).startswith(("mirror", "kaggle")):
    st.caption(f":material/dataset: Model trained on the Kaggle *Water Potability* dataset ({wp.meta.get('n_train', '?')} training rows) · "
               f"test ROC-AUC {wp.meta.get('test_metrics', {}).get('roc_auc', float('nan')):.2f}. Treat results as screening, not certification.")


def render_assessment(a: WaterAssessment) -> None:
    """Verdict, gauges, range chart, SHAP waterfall, checks table."""
    tone = "green" if a.potable else "red"
    st.markdown(f"### {status_badge(tone, a.label)} &nbsp; {status_badge('green' if a.habitability_score >= 70 else 'amber' if a.habitability_score >= 50 else 'red', f'Habitability: {a.band}')}")
    st.markdown(a.explanation)
    c1, c2 = st.columns(2)
    with c1.container(border=True):
        st.plotly_chart(probability_gauge(a.probability), theme=None, alt=f"Gauge: probability potable {a.probability:.0%}")
    with c2.container(border=True):
        st.plotly_chart(score_gauge(a.habitability_score, "Habitability score (0-100)"), theme=None, alt=f"Gauge: habitability {a.habitability_score:.0f}")
    left, right = st.columns(2)
    with left.container(border=True):
        st.plotly_chart(parameter_range_chart(a.checks, g), theme=None, alt="Each parameter as a percentage of its guideline limit")
    with right.container(border=True):
        st.plotly_chart(shap_waterfall(a.contributions, a.base_value, g), theme=None, alt="SHAP waterfall of parameter contributions")
    table = a.checks.assign(status=a.checks["status"].map({"green": "Within guideline", "amber": "Borderline", "red": "Outside guideline", "unknown": "Not provided"}))
    st.dataframe(table[["label", "value", "unit", "safe_range", "status"]], hide_index=True,
                 column_config={"label": "Parameter", "value": st.column_config.NumberColumn("Value", format="%.3g"), "unit": "Unit", "safe_range": "Safe range", "status": "Status"})


mode = st.segmented_control("Input", ["Single sample", "CSV batch"], default="Single sample", key="water_mode",
                            format_func=lambda m: {"Single sample": ":material/tune: Single sample", "CSV batch": ":material/table: CSV batch"}[m])

if mode == "Single sample":
    with st.form("water_form"):
        st.markdown("**Water parameters** · untick a parameter if it was not measured (the model imputes the training median)")
        values: dict[str, float | None] = {}
        cols = st.columns(3)
        params = list(wp.features)
        for i, f in enumerate(params):
            gi = g.get(f, {"label": f, "unit": "", "slider": [0.0, 1000.0], "default": 0.0})
            lo, hi = gi["slider"]
            with cols[i % 3]:
                label = f"{gi['label']}" + (f" ({gi['unit']})" if gi.get("unit") else "")
                v = st.number_input(label, min_value=float(lo), max_value=float(hi) * 10, value=float(gi["default"]), step=(hi - lo) / 1000.0, format="%.4g", key=f"v_{f}")
                measured = st.checkbox("Measured", value=True, key=f"m_{f}", help=f"Untick if {gi['label']} was not measured")
                values[f] = v if measured else None
        st.divider()
        name, lat, lon = site_inputs("water")
        submitted = st.form_submit_button("Assess water", type="primary", icon=":material/science:")
    if submitted:
        clean, errors = validate_water_inputs(values, g)
        try:
            lat, lon = validate_coordinates(lat, lon)
        except ValueError as exc:
            errors.append(str(exc))
        if errors:
            st.error("\n".join(f"- {e}" for e in errors), title="Please fix the inputs", icon=":material/error:")
        else:
            try:
                a = wp.assess(clean)
            except ValueError as exc:
                st.error(str(exc), icon=":material/error:")
            else:
                st.session_state["last_water"] = a
                get_db().log_water(clean, a.potable, a.probability, a.habitability_score, name or None, lat, lon)
                st.toast("Saved to history", icon=":material/save:")
    if "last_water" in st.session_state:
        render_assessment(st.session_state["last_water"])

else:
    st.markdown(f"Upload a CSV with any of these columns: `{'`, `'.join(cfg['water']['features'] + cfg['water']['optional_features'])}`. Missing values are imputed.")
    template = pd.DataFrame([{f: g[f]["default"] for f in cfg["water"]["features"]}])
    st.download_button("CSV template", df_to_csv_bytes(template), "water_template.csv", "text/csv", icon=":material/download:")
    up = st.file_uploader("Water samples CSV", type=["csv"])
    if up is not None:
        try:
            df = read_water_csv(up.getvalue(), cfg["water"]["features"])
            with st.spinner(f"Scoring {len(df)} samples..."):
                out = wp.predict_df(df)
        except ValueError as exc:
            st.error(str(exc), title="Could not score this CSV", icon=":material/error:")
        else:
            n_pot = int((out["prediction"] == "Potable").sum())
            with st.container(horizontal=True):
                st.metric("Samples", len(out), border=True)
                st.metric("Potable", n_pot, border=True)
                st.metric("Non-potable", len(out) - n_pot, border=True)
                st.metric("Mean habitability", f"{out['habitability_score'].mean():.0f}/100", border=True)
            st.dataframe(out, hide_index=True, column_config={"p_potable": st.column_config.ProgressColumn("P(potable)", min_value=0.0, max_value=1.0, format="%.2f"),
                                                              "habitability_score": st.column_config.NumberColumn("Habitability", format="%.0f")})
            st.download_button("Predictions CSV", df_to_csv_bytes(out), "water_predictions.csv", "text/csv", icon=":material/download:")
            row = st.selectbox("Explain sample (row index)", out.index.tolist())
            render_assessment(wp.assess({f: (None if pd.isna(df.at[row, f]) else df.at[row, f]) for f in wp.features if f in df.columns}))
