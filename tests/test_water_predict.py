import numpy as np
import pandas as pd
import pytest

from src.water.predict import (
    WaterModelNotAvailableError,
    WaterPredictor,
    check_parameter,
    compliance_score,
    format_range,
    guideline_checks,
    habitability_band,
    habitability_score,
)

SAMPLE = {"ph": 7.0, "Hardness": 190, "Solids": 20000, "Chloramines": 7, "Sulfate": 330, "Conductivity": 420, "Organic_carbon": 14, "Trihalomethanes": 66, "Turbidity": 4}


def test_check_parameter_traffic_lights():
    g = {"safe": [6.5, 8.5], "warn": [6.0, 9.0]}
    assert check_parameter(7.0, g) == "green"
    assert check_parameter(8.8, g) == "amber"
    assert check_parameter(10.0, g) == "red"
    assert check_parameter(None, g) == "unknown"
    assert check_parameter(float("nan"), g) == "unknown"
    assert check_parameter(3.0, {"safe": [None, 4], "warn": [None, 5]}) == "green"


def test_format_range():
    assert format_range([6.5, 8.5]) == "6.5-8.5"
    assert format_range([None, 4], "ppm") == "<= 4 ppm"


def test_scores(cfg):
    checks = guideline_checks({"ph": 7.0, "Turbidity": 3.0, "Sulfate": 600}, cfg["water"]["guidelines"])
    assert dict(zip(checks["parameter"], checks["status"])) == {"ph": "green", "Turbidity": "amber", "Sulfate": "red"}
    assert compliance_score(checks) == pytest.approx(0.5)
    assert habitability_score(1.0, 1.0) == 100.0
    assert habitability_score(0.0, 0.0) == 0.0
    assert habitability_score(1.5, -1) == 60.0  # clipped
    assert habitability_band(80) == "Good" and habitability_band(10) == "Critical"


def test_training_metadata(water_model):
    _, meta = water_model
    assert set(meta["test_metrics"]) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert meta["test_metrics"]["roc_auc"] > 0.6  # synthetic labels are learnable
    assert meta["cv_metrics"]["roc_auc"]["mean"] > 0.6


def test_assess_single_sample(water_model):
    c, _ = water_model
    wp = WaterPredictor.from_config(c)
    a = wp.assess(SAMPLE)
    assert a.label in ("Potable", "Non-potable") and 0 <= a.probability <= 1
    assert 0 <= a.habitability_score <= 100
    assert len(a.contributions) == len(wp.features)
    # SHAP additivity: base + sum(contributions) == logit(p)
    logit = np.log(a.probability / (1 - a.probability))
    assert a.base_value + a.contributions["contribution"].sum() == pytest.approx(logit, abs=1e-2)
    assert "probability" in a.explanation


def test_assess_handles_missing_and_bad_values(water_model):
    c, _ = water_model
    wp = WaterPredictor.from_config(c)
    a = wp.assess({"ph": 7.2})
    assert a.contributions["imputed"].sum() == len(wp.features) - 1
    with pytest.raises(ValueError):
        wp.assess({"ph": "abc"})
    with pytest.raises(ValueError):
        wp.assess({})


def test_batch_predict(water_model):
    c, _ = water_model
    wp = WaterPredictor.from_config(c)
    df = pd.read_csv(c["water"]["test_split_path"]).head(10)
    out = wp.predict_df(df)
    assert {"p_potable", "prediction", "habitability_score"} <= set(out.columns) and len(out) == 10
    with pytest.raises(ValueError):
        wp.predict_df(pd.DataFrame({"foo": [1]}))


def test_missing_model_raises(tmp_path, cfg):
    with pytest.raises(WaterModelNotAvailableError):
        WaterPredictor(tmp_path / "nope.json", tmp_path / "nope_meta.json", cfg)
