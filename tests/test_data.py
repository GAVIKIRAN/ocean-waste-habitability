import pandas as pd
import pytest

from src.data.prepare_yolo_dataset import Sample, build_class_mapper, normalise_name, split_samples
from src.data.synthetic_waste import DRAWERS, generate_dataset
from src.data.water_data import generate_synthetic_water, validate_water_df


def test_every_config_class_has_a_drawer(cfg):
    assert set(cfg["waste"]["classes"]) <= set(DRAWERS)


def test_synthetic_dataset_labels_are_valid(tmp_path, cfg):
    out = generate_dataset(tmp_path, cfg["waste"]["classes"], n_images=5, width=200, height=150, max_objects=3, seed=3)
    labels = list((out / "labels").glob("*.txt"))
    assert len(labels) == 5 and len(list((out / "images").glob("*.jpg"))) == 5
    for lf in labels:
        for line in lf.read_text().splitlines():
            c, *box = line.split()
            assert 0 <= int(c) < len(cfg["waste"]["classes"])
            assert all(0.0 <= float(v) <= 1.0 for v in box)


def test_class_mapping(cfg):
    m = build_class_mapper(cfg)
    classes = cfg["waste"]["classes"]
    assert m("trash_bottle") == classes.index("plastic_bottle")
    assert m("Plastic Bottle") == classes.index("plastic_bottle")
    assert m("animal_fish") is None
    assert m("totally-unknown") is None
    assert build_class_mapper(cfg, unknown_to_other=True)("totally-unknown") == classes.index("other_debris")
    assert normalise_name("Fishing-Net ") == "fishing_net"


def test_split_ratios():
    samples = [Sample(image=None, source="s") for _ in range(100)]  # type: ignore[arg-type]
    s = split_samples(samples, [0.7, 0.2, 0.1], seed=1)
    assert [len(s[k]) for k in ("train", "val", "test")] == [70, 20, 10]
    with pytest.raises(ValueError):
        split_samples(samples, [0.5, 0.2, 0.1], seed=1)


def test_synthetic_water_schema(cfg):
    df = generate_synthetic_water(n=300, seed=0, heavy_metals=True)
    validate_water_df(df, cfg["water"]["features"], "Potability")
    assert {"Lead", "Mercury"} <= set(df.columns)
    assert df["ph"].isna().any() and 0.1 < df["Potability"].mean() < 0.9


def test_validate_water_df_errors(cfg):
    with pytest.raises(ValueError):
        validate_water_df(pd.DataFrame({"ph": [7]}), cfg["water"]["features"])


def test_split_keeps_video_frames_together():
    from pathlib import Path

    from src.data.prepare_yolo_dataset import group_key

    samples = [Sample(image=Path(f"vid_{v:06d}_frame{f:07d}.jpg"), source="trashcan") for v in range(30) for f in range(10)]
    samples += [Sample(image=Path(f"img_{i}.jpg"), source="other") for i in range(20)]
    s = split_samples(samples, [0.7, 0.2, 0.1], seed=3)
    assert sum(len(v) for v in s.values()) == len(samples)
    vids = {k: {group_key(x) for x in v if "vid_" in str(x.image)} for k, v in s.items()}
    assert not (vids["train"] & vids["val"]) and not (vids["train"] & vids["test"]) and not (vids["val"] & vids["test"])
    assert 0.6 < len(s["train"]) / len(samples) < 0.8


def test_parse_source_dir(tmp_path):
    from src.data.prepare_yolo_dataset import parse_source_dir

    assert parse_source_dir(f"tc={tmp_path}") == ("tc", tmp_path)
    assert parse_source_dir(str(tmp_path)) == (tmp_path.name, tmp_path)
    with pytest.raises(FileNotFoundError):
        parse_source_dir(str(tmp_path / "missing"))


def test_balanced_split_covers_rare_class():
    from pathlib import Path

    from src.data.prepare_yolo_dataset import balanced_split, split_score

    # 40 videos; the rare class 1 appears in only 4 of them
    samples = []
    for v in range(40):
        for f in range(5):
            labels = [(1 if v % 10 == 0 else 0, 0.5, 0.5, 0.1, 0.1)]
            samples.append(Sample(image=Path(f"vid_{v:06d}_frame{f:07d}.jpg"), source="t", labels=labels))
    splits, _ = balanced_split(samples, [0.7, 0.2, 0.1], seed=0, n_classes=2, tries=200)
    for k in ("train", "val", "test"):
        assert any(lab[0] == 1 for smp in splits[k] for lab in smp.labels), k
    assert split_score(splits, [0.7, 0.2, 0.1], 2) < 10


def test_group_key_merges_roboflow_augmentations():
    from pathlib import Path

    from src.data.prepare_yolo_dataset import group_key

    a = Sample(image=Path("uwg_g-838_jpg.rf.4cb0a5ecfe2561ad5979ba148a56b6b4.jpg"), source="up")
    b = Sample(image=Path("uwg_g-838__flipv_jpg.rf.0123456789abcdef0123456789abcdef.jpg"), source="up")
    c = Sample(image=Path("uwg_g-839_jpg.rf.4cb0a5ecfe2561ad5979ba148a56b6b4.jpg"), source="up")
    assert group_key(a) == group_key(b) != group_key(c)
