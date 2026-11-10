"""Shared fixtures: config, synthetic images, a tiny trained water model."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.common import load_config  # noqa: E402


@pytest.fixture(scope="session")
def cfg() -> dict:
    return load_config()


@pytest.fixture()
def underwater_img() -> np.ndarray:
    """Hazy blue-green synthetic image with a few objects."""
    from src.data.synthetic_waste import generate_image

    rng = np.random.default_rng(0)
    img, _ = generate_image(rng, ["plastic_bottle", "can", "rubber_tire"], 160, 120, 3)
    return img


@pytest.fixture(scope="session")
def water_model(tmp_path_factory):
    """Train a very small XGBoost model on synthetic data into a temp dir."""
    from src.data.water_data import generate_synthetic_water
    from src.water.train import train

    tmp = tmp_path_factory.mktemp("water")
    csv = tmp / "water.csv"
    generate_synthetic_water(n=600, seed=1).to_csv(csv, index=False)
    c = load_config()
    c["water"].update(model_path=str(tmp / "m.json"), meta_path=str(tmp / "meta.json"), test_split_path=str(tmp / "test.csv"), cv_folds=3)
    c["water"]["tuning"]["demo_n_trials"] = 2
    meta = train(c, mode="demo", method="optuna", data_path=str(csv))
    return c, meta
