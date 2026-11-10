"""Optional Gradio demo (alternative to the Streamlit app).

    pip install -r requirements-optional.txt
    python app_gradio.py            # http://localhost:7860
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.common import get_logger, load_config
from src.water.predict import WaterModelNotAvailableError, WaterPredictor
from src.waste.predict import ModelNotAvailableError, WasteDetector

log = get_logger(__name__)
CFG = load_config()
_cache: dict[str, Any] = {}


def _detector() -> WasteDetector:
    if "det" not in _cache:
        _cache["det"] = WasteDetector.from_config(CFG)
    return _cache["det"]


def _water() -> WaterPredictor:
    if "wp" not in _cache:
        _cache["wp"] = WaterPredictor.from_config(CFG)
    return _cache["wp"]


def detect(image_rgb: np.ndarray | None, conf: float, iou: float, enhance: bool) -> tuple[np.ndarray | None, pd.DataFrame, str]:
    """Gradio callback: RGB image in, annotated RGB image + detections table + summary out."""
    if image_rgb is None:
        return None, pd.DataFrame(), "Upload an image first."
    try:
        det = _detector()
    except ModelNotAvailableError as exc:
        return None, pd.DataFrame(), str(exc)
    r = det.predict(np.ascontiguousarray(image_rgb[..., ::-1]), conf=conf, iou=iou, enhance_input=enhance)
    note = " (COCO demo model - train the waste model for real classes)" if det.is_demo else ""
    return r.annotated[..., ::-1], r.detections, f"{r.n} item(s): {r.class_counts}{note}"


def water(*values: float) -> tuple[str, pd.DataFrame]:
    """Gradio callback: water parameters in, verdict + guideline table out."""
    try:
        wp = _water()
    except WaterModelNotAvailableError as exc:
        return str(exc), pd.DataFrame()
    feats = CFG["water"]["features"]
    a = wp.assess(dict(zip(feats, values)))
    md = f"### {a.label} — P(potable) {a.probability:.0%} — habitability {a.habitability_score:.0f}/100 ({a.band})\n\n{a.explanation}"
    return md, a.checks[["label", "value", "unit", "safe_range", "status"]]


def build() -> Any:
    import gradio as gr

    g = CFG["water"]["guidelines"]
    inf = CFG["waste"]["inference"]
    with gr.Blocks(title="Ocean Waste & Water Habitability") as demo:
        gr.Markdown("# Ocean Waste Detection & Water Habitability Test")
        with gr.Tab("Waste detection"):
            with gr.Row():
                inp = gr.Image(type="numpy", label="Underwater image")
                out = gr.Image(type="numpy", label="Detections")
            with gr.Row():
                conf = gr.Slider(0.05, 0.95, inf["conf"], step=0.05, label="Confidence")
                iou = gr.Slider(0.1, 0.9, inf["iou"], step=0.05, label="IoU")
                enh = gr.Checkbox(True, label="Dehaze + CLAHE")
            summary = gr.Markdown()
            table = gr.Dataframe()
            gr.Button("Detect", variant="primary").click(detect, [inp, conf, iou, enh], [out, table, summary])
        with gr.Tab("Water quality"):
            sliders = [gr.Slider(*g[f]["slider"], value=g[f]["default"], label=f"{g[f]['label']} ({g[f]['unit']})" if g[f]["unit"] else g[f]["label"]) for f in CFG["water"]["features"]]
            verdict = gr.Markdown()
            checks = gr.Dataframe()
            gr.Button("Assess", variant="primary").click(water, sliders, [verdict, checks])
    return demo


if __name__ == "__main__":
    build().launch()
