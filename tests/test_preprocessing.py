import numpy as np
import pytest

from src.preprocessing.underwater_enhance import (
    EnhanceOptions,
    apply_clahe,
    dark_channel,
    dehaze,
    denoise,
    enhance,
    gray_world_white_balance,
    options_from_config,
)


def test_dark_channel_shape_and_range(underwater_img):
    d = dark_channel(underwater_img.astype(np.float32) / 255.0, 7)
    assert d.shape == underwater_img.shape[:2]
    assert 0.0 <= d.min() <= d.max() <= 1.0


def test_dehaze_preserves_shape_and_reduces_dark_channel(underwater_img):
    out = dehaze(underwater_img, patch_size=7)
    assert out.shape == underwater_img.shape and out.dtype == np.uint8
    before = dark_channel(underwater_img.astype(np.float32) / 255, 7).mean()
    after = dark_channel(out.astype(np.float32) / 255, 7).mean()
    assert after <= before  # haze removal lowers the dark channel


def test_clahe_increases_contrast(underwater_img):
    low = (underwater_img * 0.3 + 90).astype(np.uint8)
    out = apply_clahe(low, clip_limit=3.0)
    assert out.std() > low.std()


def test_white_balance_equalises_channel_means(underwater_img):
    out = gray_world_white_balance(underwater_img)
    spread_after = np.ptp(out.reshape(-1, 3).mean(0))
    spread_before = np.ptp(underwater_img.reshape(-1, 3).mean(0))
    assert spread_after < spread_before


@pytest.mark.parametrize("fast", [True, False])
def test_denoise(underwater_img, fast):
    assert denoise(underwater_img, fast=fast).shape == underwater_img.shape


def test_enhance_pipeline_and_toggles(underwater_img, cfg):
    opts = options_from_config(cfg, denoise=True, white_balance=True)
    assert enhance(underwater_img, opts).shape == underwater_img.shape
    same = enhance(underwater_img, EnhanceOptions(dehaze=False, clahe=False))
    assert np.array_equal(same, underwater_img)


@pytest.mark.parametrize(
    "bad",
    [np.zeros((10, 10), np.uint8), np.zeros((10, 10, 3), np.float32), np.zeros((1, 1, 3), np.uint8), "x"],
)
def test_invalid_inputs_raise(bad):
    with pytest.raises(ValueError):
        enhance(bad)
