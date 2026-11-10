import numpy as np
import pandas as pd
import pytest

from src.utils.common import resolve_path
from src.waste.predict import DETECTION_COLUMNS, DetectionResult, boxes_to_dataframe, class_counts, empty_detections
from src.waste.train import build_train_args, is_oom, next_batch_after_oom


def test_boxes_to_dataframe():
    df = boxes_to_dataframe(np.array([[0, 0, 10, 10], [5, 5, 20, 30]], float), np.array([0.9, 0.4]), np.array([0, 2]), {0: "plastic_bottle", 2: "can"}, "img.jpg")
    assert list(df.columns) == DETECTION_COLUMNS and len(df) == 2
    assert class_counts(df) == {"plastic_bottle": 1, "can": 1}
    assert boxes_to_dataframe(np.zeros((0, 4)), np.zeros(0), np.zeros(0), {}).empty


def test_coverage():
    img = np.zeros((100, 100, 3), np.uint8)
    df = boxes_to_dataframe(np.array([[0, 0, 50, 50]], float), np.array([0.9]), np.array([0]), ["a"])
    r = DetectionResult(img, df, img, 1.0, class_counts(df))
    assert r.coverage() == pytest.approx(0.25)
    assert DetectionResult(img, empty_detections(), img).coverage() == 0.0


def test_oom_helpers():
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate 20 MiB"))
    assert not is_oom(ValueError("shape mismatch"))
    assert next_batch_after_oom(8) == 4 and next_batch_after_oom(1) == 0 and next_batch_after_oom(-1) == 8


def test_train_arg_presets(cfg):
    full = build_train_args(cfg, "full", False, {})
    assert full["model"] == "yolo11s.pt" and full["imgsz"] == 640 and full["batch"] == 8 and full["amp"] is True
    assert full["patience"] == 20 and full["epochs"] == 100 and full["optimizer"] == "Adam"
    low = build_train_args(cfg, "full", True, {})
    assert (low["model"], low["imgsz"], low["batch"]) == ("yolo11n.pt", 512, 4)
    demo = build_train_args(cfg, "demo", False, {"epochs": 2})
    assert demo["model"] == "yolo11n.pt" and demo["epochs"] == 2


@pytest.mark.skipif(not resolve_path("models/waste_best.pt").exists(), reason="trained waste weights not present")
def test_detector_end_to_end(cfg, underwater_img):
    from src.waste.predict import WasteDetector

    det = WasteDetector.from_config(cfg, allow_demo=False, device="cpu")
    r = det.predict(underwater_img, conf=0.1, enhance_input=True)
    assert r.annotated.shape == underwater_img.shape
    assert isinstance(r.detections, pd.DataFrame)
    with pytest.raises(ValueError):
        det.predict(np.zeros((5, 5), np.uint8))


class _FakeYOLO:
    """Stand-in for ultralytics.YOLO: raises CUDA OOM while batch > ``limit``."""

    calls: list[tuple[str, int]] = []
    limit = 2

    def __init__(self, weights: str) -> None:
        self.weights = weights

    def train(self, **kw):  # noqa: ANN003, ANN201
        _FakeYOLO.calls.append((self.weights, kw["batch"]))
        if kw["batch"] > _FakeYOLO.limit:
            raise RuntimeError("CUDA out of memory. Tried to allocate 512.00 MiB")
        return None


def test_oom_retry_halves_batch_then_falls_back(monkeypatch):
    import ultralytics

    from src.waste import train as wt

    monkeypatch.setattr(ultralytics, "YOLO", _FakeYOLO)
    monkeypatch.setattr(wt, "resolve_weights", lambda name: name)
    _FakeYOLO.calls, _FakeYOLO.limit = [], 2
    _, used = wt.train_with_oom_retry("yolo11s.pt", "yolo11n.pt", {"batch": 16, "imgsz": 640, "epochs": 1})
    assert [b for _, b in _FakeYOLO.calls] == [16, 8, 4, 2] and used["batch"] == 2 and used["model"] == "yolo11s.pt"

    # never fits with the default model -> switches to the fallback model at batch 4
    _FakeYOLO.calls, _FakeYOLO.limit = [], 0
    with pytest.raises(RuntimeError):
        wt.train_with_oom_retry("yolo11s.pt", "yolo11n.pt", {"batch": 2, "imgsz": 640, "epochs": 1})
    assert ("yolo11n.pt", 4) in _FakeYOLO.calls


def test_non_oom_errors_propagate(monkeypatch):
    import ultralytics

    from src.waste import train as wt

    class Broken(_FakeYOLO):
        def train(self, **kw):  # noqa: ANN003, ANN201
            raise ValueError("bad dataset yaml")

    monkeypatch.setattr(ultralytics, "YOLO", Broken)
    monkeypatch.setattr(wt, "resolve_weights", lambda name: name)
    with pytest.raises(ValueError):
        wt.train_with_oom_retry("yolo11s.pt", "yolo11n.pt", {"batch": 8, "imgsz": 640, "epochs": 1})


def test_tile_grid_covers_image():
    from src.waste.predict import tile_grid

    assert tile_grid(800, 450, 1) == [(0, 0, 800, 450)]
    t = tile_grid(800, 450, 3, overlap=0.25)
    assert len(t) == 9
    assert min(x0 for x0, *_ in t) == 0 and max(x1 for *_, x1, _ in t) == 800
    assert min(y0 for _, y0, *_ in t) == 0 and max(y1 for *_, y1 in t) == 450
    assert all(x1 - x0 > 800 / 3 and y1 - y0 > 450 / 3 for x0, y0, x1, y1 in t)  # tiles overlap


def test_cross_class_dedup():
    from src.waste.predict import _cross_class_dedup

    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 9.5], [20, 20, 30, 30]], np.float32)
    keep = _cross_class_dedup(boxes, np.array([0.9, 0.5, 0.4]), thr=0.7)
    assert keep.tolist() == [0, 2]


@pytest.mark.skipif(not resolve_path("models/waste_best.pt").exists(), reason="trained waste weights not present")
def test_high_recall_mode_runs(cfg, underwater_img):
    from src.waste.predict import WasteDetector

    det = WasteDetector.from_config(cfg, allow_demo=False, device="cpu")
    r1 = det.predict(underwater_img, conf=0.05)
    r3 = det.predict(underwater_img, conf=0.05, tiles=3)
    assert r3.annotated.shape == underwater_img.shape
    assert list(r3.detections.columns) == list(r1.detections.columns)
    if not r3.detections.empty:
        d = r3.detections
        h, w = underwater_img.shape[:2]
        assert (d["x1"] >= -1).all() and (d["x2"] <= w + 1).all() and (d["y1"] >= -1).all() and (d["y2"] <= h + 1).all()
