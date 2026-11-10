import numpy as np
import pandas as pd

from src.utils.report import build_report, combine_scores, grade_for, recommendations, report_to_pdf, waste_score
from src.waste.predict import DetectionResult, boxes_to_dataframe, class_counts


def test_scores(cfg):
    assert waste_score(0, 0.0, cfg) == 100.0
    assert waste_score(100, 1.0, cfg) == 0.0
    assert combine_scores(80, 60, cfg) == 70.0
    assert combine_scores(None, 60, cfg) == 60.0
    assert grade_for(90, cfg) == "A" and grade_for(10, cfg) == "E"


def test_recommendations():
    checks = pd.DataFrame([{"parameter": "Turbidity", "status": "red"}])
    recs = recommendations({"rope_net": 2}, checks, 30)
    assert "critical" in recs[0]
    assert any("ghost" in r for r in recs) and any("urgent" in r for r in recs)
    assert "routine" in recommendations({}, None, 95)[0]


def test_pdf_generation(cfg):
    img = np.full((120, 160, 3), 80, np.uint8)
    df = boxes_to_dataframe(np.array([[10, 10, 60, 60]], float), np.array([0.8]), np.array([5]), cfg["waste"]["classes"])
    det = DetectionResult(img, df, img, 1.0, class_counts(df))
    rep = build_report(cfg, "Test reef – north", 1.0, 2.0, detection=det, notes="Visibility 5 m ≤ normal")
    pdf = report_to_pdf(rep, img)
    assert pdf[:4] == b"%PDF" and len(pdf) > 1000
    assert rep.water_score is None and rep.combined_score == rep.waste_score
