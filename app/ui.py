"""Shared Streamlit plumbing: path setup, cached model loaders, sidebar and small UI helpers."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.common import load_config, resolve_path  # noqa: E402
from src.utils.db import HistoryDB  # noqa: E402

BADGE = {"green": "green", "amber": "orange", "red": "red", "unknown": "gray"}


@st.cache_resource(show_spinner=False)
def get_config() -> dict[str, Any]:
    return load_config()


@st.cache_resource(show_spinner=False)
def get_db() -> HistoryDB:
    return HistoryDB(resolve_path(get_config()["paths"]["db"]))


@st.cache_resource(show_spinner=False)
def device_label() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return f"GPU · {torch.cuda.get_device_name(0)} (FP16)"
    except Exception:  # noqa: BLE001
        pass
    return "CPU"


def _mtime(p: Path) -> float:
    return p.stat().st_mtime if p.exists() else 0.0


def render_sidebar_status(cfg: dict[str, Any]) -> None:
    """Model status + compute device, shown on every page."""
    waste_ok = resolve_path(cfg["waste"]["weights"]).exists()
    water_ok = resolve_path(cfg["water"]["model_path"]).exists()
    with st.sidebar:
        st.caption("System status")
        st.markdown(
            f"Waste model {':green-badge[:material/check: trained]' if waste_ok else ':orange-badge[:material/warning: demo fallback]'}  \n"
            f"Water model {':green-badge[:material/check: trained]' if water_ok else ':red-badge[:material/close: not trained]'}  \n"
            f":gray-badge[:material/memory: {device_label()}]"
        )


@st.cache_resource(show_spinner="Loading YOLO11 waste detector...", max_entries=2)
def _load_detector(weights_mtime: float, allow_demo: bool):  # noqa: ANN202 - cached resource
    from src.waste.predict import WasteDetector

    det = WasteDetector.from_config(get_config(), allow_demo=allow_demo)
    if det.device != "cpu":  # warm-up so the first real request is fast
        import numpy as np

        det.predict(np.zeros((320, 320, 3), np.uint8))
    return det


def get_detector(allow_demo: bool = True):  # noqa: ANN201
    """Cached detector, refreshed automatically when the weights file changes. None if unavailable."""
    from src.waste.predict import ModelNotAvailableError

    try:
        return _load_detector(_mtime(resolve_path(get_config()["waste"]["weights"])), allow_demo)
    except ModelNotAvailableError:
        return None


@st.cache_resource(show_spinner="Loading XGBoost water model...", max_entries=2)
def _load_water(model_mtime: float):  # noqa: ANN202
    from src.water.predict import WaterPredictor

    return WaterPredictor.from_config(get_config())


def get_water_predictor():  # noqa: ANN201
    """Cached water predictor or None if the model has not been trained."""
    from src.water.predict import WaterModelNotAvailableError

    try:
        return _load_water(_mtime(resolve_path(get_config()["water"]["model_path"])))
    except WaterModelNotAvailableError:
        return None


def train_first(module: str) -> None:
    """Standard 'train the model first' notice."""
    cmd = "python -m src.waste.train --mode demo" if module == "waste" else "python -m src.water.train --mode demo"
    st.warning(f"Train the {module} model first, then reload this page.", title=f"No {module} model found", icon=":material/model_training:")
    st.code(cmd, language="bash")
    st.caption("Or run the whole pipeline: `scripts\\run_all.bat demo` (Windows) or `bash scripts/run_all.sh demo`.")


def site_inputs(key: str, required_name: bool = False, default_geo: bool = False) -> tuple[str, float | None, float | None]:
    """Site name + optional coordinates.

    The coordinate inputs are always rendered (these widgets live inside ``st.form``s, where a
    checkbox cannot reveal other widgets before submit); the checkbox decides whether they are used.
    """
    name = st.text_input("Site name" + (" (required)" if required_name else ""), key=f"{key}_site", placeholder="e.g. North reef, buoy 4")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        lat = st.number_input("Latitude", -90.0, 90.0, 12.97, format="%.5f", key=f"{key}_lat")
        lon = st.number_input("Longitude", -180.0, 180.0, 74.80, format="%.5f", key=f"{key}_lon")
    use_geo = st.checkbox("Save GPS coordinates", key=f"{key}_geo", value=default_geo)
    return name.strip(), (lat if use_geo else None), (lon if use_geo else None)


def status_badge(status: str, text: str) -> str:
    """Markdown badge for a traffic-light status (always carries a text label)."""
    icon = {"green": "check_circle", "amber": "warning", "red": "cancel", "unknown": "help"}[status]
    return f":{BADGE[status]}-badge[:material/{icon}: {text}]"
