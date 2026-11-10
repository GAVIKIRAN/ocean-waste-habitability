from src.utils.db import HistoryDB


def test_db_roundtrip(tmp_path):
    db = HistoryDB(tmp_path / "h.db")
    db.log_waste(3, {"can": 2, "rubber_tire": 1}, 0.8, 0.1, 70.0, "Reef A", 10.0, 20.0, "img.jpg", True)
    db.log_waste(1, {"can": 1}, site_name="Reef B")
    db.log_water({"ph": 7}, True, 0.8, 75.0, "Reef A", 10.0, 20.0)
    db.log_report("Reef A", 72.0, "B", 70.0, 75.0, 10.0, 20.0, {"x": 1})
    assert len(db.waste_history()) == 2 and len(db.water_history()) == 1 and len(db.report_history()) == 1
    assert db.waste_class_totals() == {"can": 3, "rubber_tire": 1}
    sites = db.sites()
    assert set(sites["kind"]) == {"site report", "water test", "waste scan"}
    assert sites[["lat", "lon"]].notna().all().all()
    db.clear()
    assert db.waste_history().empty and db.sites().empty
