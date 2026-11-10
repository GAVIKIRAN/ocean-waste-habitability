"""Plotting helpers (Plotly for the app, Matplotlib for reports/PDF).

Colour roles follow a validated palette (checked for colour-vision-deficiency
separation and >= 3:1 contrast against the app's dark navy surface):
* categorical slots are assigned in a fixed order, never cycled;
* status colours (good / warning / critical) are reserved for traffic-light states
  and always shown with a text label, never colour alone;
* single-series charts use one hue (slot 1).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

SURFACE = "#0b2236"
PAGE_BG = "#04121f"
TEXT = "#e6f1f8"
TEXT_MUTED = "#9db4c6"
GRID = "rgba(157,180,198,0.15)"
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
STATUS = {"green": "#0ca30c", "amber": "#fab219", "red": "#d03b3b", "unknown": "#6b7f90"}
STATUS_LABEL = {"green": "Within guideline", "amber": "Borderline", "red": "Outside guideline", "unknown": "Not provided"}
STATUS_ICON = {"green": "✔", "amber": "▲", "red": "✖", "unknown": "–"}


def _layout(fig: go.Figure, title: str | None = None, height: int = 360) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=TEXT), x=0, xanchor="left") if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, size=12),
        margin=dict(l=10, r=20, t=50 if title else 15, b=10),
        height=height,
        hoverlabel=dict(bgcolor=SURFACE, font_color=TEXT, bordercolor=SERIES[0]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, color=TEXT_MUTED)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, color=TEXT_MUTED)
    return fig


def pretty(name: str) -> str:
    """``plastic_bottle`` -> ``Plastic bottle``."""
    return str(name).replace("_", " ").capitalize()


# ---------------------------------------------------------------- waste charts
def class_counts_bar(counts: Mapping[str, int], title: str = "Detected items by class") -> go.Figure:
    """Horizontal bar of detections per class (single series, sorted)."""
    items = sorted(counts.items(), key=lambda kv: kv[1])
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[pretty(k) for k, _ in items], orientation="h",
        marker=dict(color=SERIES[0], line=dict(color=SURFACE, width=2), cornerradius=4),
        text=[v for _, v in items], textposition="outside", textfont=dict(color=TEXT_MUTED),
        hovertemplate="%{y}: %{x} item(s)<extra></extra>",
    ))
    fig.update_xaxes(title="Count", rangemode="tozero")
    return _layout(fig, title, height=max(220, 60 + 32 * len(items)))


# ---------------------------------------------------------------- water charts
def probability_gauge(p: float, title: str = "Probability potable") -> go.Figure:
    """Gauge of P(potable) in % with status-coloured bands (labels give the meaning)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=round(100 * p, 1), number=dict(suffix="%", font=dict(color=TEXT, size=40)),
        title=dict(text=title, font=dict(color=TEXT_MUTED, size=14)),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor=TEXT_MUTED, tickvals=[0, 30, 50, 70, 100]),
            bar=dict(color=TEXT, thickness=0.25),
            bgcolor=SURFACE,
            borderwidth=0,
            steps=[dict(range=[0, 30], color="rgba(208,59,59,0.55)"), dict(range=[30, 50], color="rgba(236,131,90,0.45)"),
                   dict(range=[50, 70], color="rgba(250,178,25,0.40)"), dict(range=[70, 100], color="rgba(12,163,12,0.45)")],
        ),
    ))
    return _layout(fig, None, height=260)


def score_gauge(score: float, title: str) -> go.Figure:
    """0-100 score gauge (habitability / combined ocean health)."""
    fig = probability_gauge(score / 100.0, title)
    fig.update_traces(number=dict(suffix="", font=dict(color=TEXT, size=40)))
    return fig


def parameter_range_chart(checks: pd.DataFrame, guidelines: Mapping[str, Mapping[str, Any]]) -> go.Figure:
    """Each parameter's value as % of its guideline upper limit (100% = limit).

    Normalising lets parameters with very different units share one axis honestly.
    Bars are coloured by status and carry the status label in the hover + text.
    """
    rows = []
    for r in checks.itertuples():
        g = guidelines.get(r.parameter, {})
        lo, hi = (g.get("safe") or [None, None])
        if r.value is None or (isinstance(r.value, float) and np.isnan(r.value)):
            continue
        if hi is not None:
            pct = 100.0 * r.value / hi if hi else 0.0
        elif lo is not None:
            pct = 100.0 * lo / r.value if r.value else 200.0
        else:
            continue
        if r.parameter == "ph":  # two-sided: distance from neutral relative to half-band
            mid, half = (lo + hi) / 2, (hi - lo) / 2
            pct = 100.0 * abs(r.value - mid) / half
        rows.append((r.label, min(pct, 400.0), r.status, f"{r.value:g} {r.unit}".strip(), r.safe_range))
    rows.sort(key=lambda t: t[1])
    fig = go.Figure(go.Bar(
        y=[t[0] for t in rows], x=[t[1] for t in rows], orientation="h",
        marker=dict(color=[STATUS[t[2]] for t in rows], line=dict(color=SURFACE, width=2), cornerradius=4),
        text=[f"{STATUS_ICON[t[2]]} {t[3]}" for t in rows], textposition="outside", textfont=dict(color=TEXT_MUTED, size=11),
        customdata=[[STATUS_LABEL[t[2]], t[4]] for t in rows],
        hovertemplate="<b>%{y}</b><br>%{text}<br>Safe: %{customdata[1]}<br>%{customdata[0]}<extra></extra>",
    ))
    fig.add_vline(x=100, line=dict(color=TEXT, dash="dash", width=1.5), annotation_text="guideline limit", annotation_font_color=TEXT_MUTED)
    fig.update_xaxes(title="% of guideline limit (pH: % of allowed deviation from 7.5)", rangemode="tozero")
    return _layout(fig, "Parameters vs WHO/EPA safe limits", height=max(260, 70 + 34 * len(rows)))


def shap_waterfall(contrib: pd.DataFrame, base_value: float, guidelines: Mapping[str, Mapping[str, Any]] | None = None, max_features: int = 10) -> go.Figure:
    """Plotly waterfall of SHAP contributions (log-odds of potable)."""
    c = contrib.head(max_features).iloc[::-1]
    g = guidelines or {}
    labels = [f"{g.get(f, {}).get('label', f)} = {'imputed' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:g}'}" for f, v in zip(c["feature"], c["value"])]
    final = base_value + contrib["contribution"].sum()
    fig = go.Figure(go.Waterfall(
        orientation="h", measure=["absolute"] + ["relative"] * len(c) + ["total"],
        y=["Base rate"] + labels + ["Model output"], x=[base_value] + c["contribution"].tolist() + [final],
        increasing=dict(marker=dict(color=SERIES[0])), decreasing=dict(marker=dict(color=SERIES[1])), totals=dict(marker=dict(color=TEXT_MUTED)),
        connector=dict(line=dict(color=GRID)), hovertemplate="%{y}<br>%{x:+.3f} log-odds<extra></extra>",
    ))
    fig.update_xaxes(title="Log-odds of potable (blue raises, orange lowers)")
    return _layout(fig, "SHAP explanation", height=max(300, 80 + 30 * (len(c) + 2)))


# ---------------------------------------------------------------- dashboard
def trend_chart(df: pd.DataFrame, x: str, y: str, title: str, y_title: str, color: str = SERIES[0]) -> go.Figure:
    """Single-series line over time with >= 8px markers."""
    fig = go.Figure(go.Scatter(x=df[x], y=df[y], mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=8, color=color, line=dict(color=SURFACE, width=2)),
                               hovertemplate="%{x|%Y-%m-%d %H:%M}<br>" + y_title + ": %{y:.1f}<extra></extra>"))
    fig.update_yaxes(title=y_title)
    return _layout(fig, title, height=300)


def two_way_bar(labels: Sequence[str], values: Sequence[int], title: str) -> go.Figure:
    """Bar for a 2-category split (e.g. potable vs non-potable), direct-labelled."""
    fig = go.Figure(go.Bar(x=list(labels), y=list(values), marker=dict(color=SERIES[: len(labels)], line=dict(color=SURFACE, width=2), cornerradius=4),
                           text=list(values), textposition="outside", hovertemplate="%{x}: %{y}<extra></extra>"))
    fig.update_yaxes(title="Tests", rangemode="tozero")
    return _layout(fig, title, height=300)


def site_map(df: pd.DataFrame, title: str = "Logged sites") -> go.Figure:
    """Map of sites coloured by combined-score band (band named in the hover)."""
    def band(s: float) -> str:
        return "green" if s >= 70 else "amber" if s >= 50 else "red"

    d = df.dropna(subset=["lat", "lon"]).copy()
    d["band"] = d["score"].apply(band)
    fig = go.Figure()
    for b, name in (("green", "Healthy (>= 70)"), ("amber", "At risk (50-69)"), ("red", "Degraded (< 50)")):
        part = d[d["band"] == b]
        if part.empty:
            continue
        fig.add_trace(go.Scattermap(lat=part["lat"], lon=part["lon"], mode="markers", name=name,
                                    marker=dict(size=14, color=STATUS[b]), text=part["site_name"], customdata=part[["score", "kind"]].to_numpy(),
                                    hovertemplate="<b>%{text}</b><br>Score %{customdata[0]:.0f} (%{customdata[1]})<extra></extra>"))
    center = dict(lat=float(d["lat"].mean()), lon=float(d["lon"].mean())) if not d.empty else dict(lat=0, lon=0)
    fig.update_layout(map=dict(style="carto-darkmatter", center=center, zoom=2 if len(d) > 1 else 5))
    return _layout(fig, title, height=420)


# ---------------------------------------------------------------- matplotlib (reports / PDF)
def save_confusion_matrix(cm: np.ndarray, labels: Sequence[str], out: Path, title: str) -> Path:
    """Annotated confusion-matrix PNG (single-hue sequential blues)."""
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(title, fontsize=11)
        thresh = cm.max() / 2 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] > thresh else "white", fontsize=13)
        fig.tight_layout()
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
    return out


def mpl_counts_png(counts: Mapping[str, int], out: Path, title: str = "Detected items by class") -> Path:
    """Matplotlib version of :func:`class_counts_bar` (for PDF export, light background)."""
    items = sorted(counts.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(6, max(1.8, 0.4 * len(items) + 0.8)))
    if items:
        ax.barh([pretty(k) for k, _ in items], [v for _, v in items], color="#2a78d6")
        for i, (_, v) in enumerate(items):
            ax.text(v, i, f" {v}", va="center", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No waste detected", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
    ax.set_title(title, fontsize=11, loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def mpl_checks_png(checks: pd.DataFrame, out: Path) -> Path:
    """Traffic-light table of guideline checks as a PNG (for PDF export)."""
    fig, ax = plt.subplots(figsize=(6.5, 0.35 * max(len(checks), 1) + 0.6))
    ax.axis("off")
    light = {"green": "#0ca30c", "amber": "#fab219", "red": "#d03b3b", "unknown": "#b0b0b0"}
    for i, r in enumerate(checks.itertuples()):
        y = len(checks) - i - 1
        ax.add_patch(plt.Rectangle((0, y + 0.15), 0.25, 0.7, color=light[r.status]))
        val = "n/a" if r.value is None or (isinstance(r.value, float) and np.isnan(r.value)) else f"{r.value:g} {r.unit}"
        ax.text(0.4, y + 0.5, f"{r.label}: {val}   (safe {r.safe_range}) - {STATUS_LABEL[r.status]}", va="center", fontsize=9)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, max(len(checks), 1))
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
