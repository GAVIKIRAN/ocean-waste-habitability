import numpy as np
import pandas as pd
import pytest

from src.utils.app_helpers import (
    decode_image,
    encode_png,
    limit_size,
    mean_confidence,
    read_water_csv,
    side_by_side,
    validate_coordinates,
    validate_water_inputs,
)


def test_decode_and_encode_roundtrip(underwater_img):
    assert np.array_equal(decode_image(encode_png(underwater_img)), underwater_img)
    with pytest.raises(ValueError):
        decode_image(b"not an image")
    with pytest.raises(ValueError):
        decode_image(b"")


def test_limit_size():
    big = np.zeros((5000, 2500, 3), np.uint8)
    assert max(limit_size(big, 1000).shape[:2]) == 1000


def test_validate_water_inputs(cfg):
    clean, errs = validate_water_inputs({"ph": "7.1", "Sulfate": None, "Turbidity": -1, "Hardness": "x"}, cfg["water"]["guidelines"])
    assert clean["ph"] == 7.1 and clean["Sulfate"] is None
    assert any("Turbidity" in e for e in errs) and any("not a number" in e for e in errs)
    _, errs = validate_water_inputs({"ph": 15}, cfg["water"]["guidelines"])
    assert errs


def test_coordinates():
    assert validate_coordinates(None, None) == (None, None)
    assert validate_coordinates(10, 20) == (10.0, 20.0)
    for bad in [(100, 0), (0, 200), (1, None)]:
        with pytest.raises(ValueError):
            validate_coordinates(*bad)


def test_read_water_csv():
    df = read_water_csv(b"ph,Turbidity\n7,3\nabc,2\n", ["ph", "Turbidity", "Sulfate"])
    assert df["ph"].isna().sum() == 1
    with pytest.raises(ValueError):
        read_water_csv(b"foo\n1\n", ["ph"])


def test_misc(underwater_img):
    assert mean_confidence(pd.DataFrame({"confidence": [0.5, 1.0]})) == 0.75
    assert mean_confidence(pd.DataFrame(columns=["confidence"])) is None
    assert side_by_side(underwater_img, underwater_img).shape[1] == underwater_img.shape[1] * 2 + 6
