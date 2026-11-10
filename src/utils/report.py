"""Combined "Ocean Health Report": scoring, grading, recommendations and PDF export."""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import pandas as pd

from src.utils.plots import mpl_checks_png, mpl_counts_png, pretty

WASTE_ADVICE: dict[str, str] = {
    "plastic_bottle": "Plastic bottles found: organise targeted clean-ups and support deposit-return schemes upstream.",
    "plastic": "Plastic (bags, film, fragments) present: high entanglement and ingestion risk for turtles - prioritise removal.",
    "can": "Metal cans detected: remove before corrosion releases metals; check nearby litter sources.",
    "glass": "Glass debris: remove with gloves; it persists for centuries and injures divers and wildlife.",
    "metal": "Metal wreckage/scrap: report to authorities; large items may need specialised salvage.",
    "rope_net": "Ropes/nets (possible ghost gear): report to a ghost-gear retrieval programme - nets keep fishing indefinitely.",
    "fishing_gear": "Fishing gear (lines, hooks, floats): engage local fisheries in gear-marking and recovery.",
    "rubber_tire": "Tyres/rubber: remove - they leach zinc and other toxic compounds as they degrade.",
    "cloth": "Textiles: a source of microfibres; remove and investigate wastewater outfalls nearby.",
    "paper_cardboard": "Paper/cardboard: indicates recent, likely local dumping - review shoreline waste management.",
    "wood": "Wood debris: usually natural or low risk, but treated timber should be removed.",
    "cup_container": "Cups/containers: single-use packaging - target nearby food outlets with reduction campaigns.",
    "wrapper": "Food wrappers: fragment into microplastics quickly - increase litter bins and enforcement on shore.",
    "pipe": "Pipes: may indicate infrastructure leaks or illegal discharge - report for inspection.",
    "other_debris": "Unclassified debris: schedule a diver survey to identify and remove it.",
}

PARAM_ADVICE: dict[str, str] = {
    "ph": "pH outside 6.5-8.5: check for acid drainage or alkaline effluents; corals and shellfish are pH sensitive.",
    "Hardness": "High hardness: monitor calcium/magnesium sources (geology or industrial discharge).",
    "Solids": "High total dissolved solids: investigate runoff, saline intrusion or effluent discharge.",
    "Chloramines": "Chloramines above 4 ppm: likely treated-wastewater influence - toxic to fish; check outfalls.",
    "Sulfate": "High sulfate: possible industrial or mining discharge.",
    "Conductivity": "High conductivity: elevated ion load - correlate with TDS and salinity sources.",
    "Organic_carbon": "High organic carbon: nutrient/organic pollution risk (algal blooms, oxygen depletion).",
    "Trihalomethanes": "Trihalomethanes above 80 ug/L: disinfection by-products - review chlorination upstream.",
    "Turbidity": "High turbidity: sediment or runoff blocks light for seagrass and corals - check erosion sources.",
    "Lead": "Lead above 0.01 mg/L: serious toxicity - trace industrial/pipe sources immediately.",
    "Mercury": "Mercury above guideline: bioaccumulates in fish - issue consumption advisory and trace source.",
}


def waste_score(n_detections: int, coverage: float, cfg: Mapping[str, Any]) -> float:
    """0-100 (100 = clean). Penalises item count and the fraction of image area covered."""
    r = cfg["report"]
    score = 100.0 - r["per_object_penalty"] * max(0, int(n_detections)) - r["area_penalty"] * max(0.0, float(coverage))
    return round(float(np.clip(score, 0.0, 100.0)), 1)


def grade_for(score: float, cfg: Mapping[str, Any]) -> str:
    """Letter grade from ``report.grades`` thresholds."""
    for g, threshold in sorted(cfg["report"]["grades"].items(), key=lambda kv: -kv[1]):
        if score >= threshold:
            return g
    return "E"


def combine_scores(waste: float | None, water: float | None, cfg: Mapping[str, Any]) -> float:
    """Weighted combination; if one component is missing the other is used alone."""
    if waste is None and water is None:
        raise ValueError("At least one of waste/water score is required")
    if waste is None:
        return round(float(water), 1)  # type: ignore[arg-type]
    if water is None:
        return round(float(waste), 1)
    w = cfg["report"]["waste_weight"]
    return round(w * waste + (1 - w) * water, 1)


def status_text(score: float) -> str:
    return "Healthy" if score >= 70 else "At risk" if score >= 50 else "Degraded"


@dataclass
class OceanHealthReport:
    """Everything needed to render the combined site report."""

    site_name: str
    lat: float | None
    lon: float | None
    combined_score: float
    grade: str
    waste_score: float | None = None
    water_score: float | None = None
    class_counts: dict[str, int] = field(default_factory=dict)
    n_detections: int = 0
    water_label: str | None = None
    water_probability: float | None = None
    checks: pd.DataFrame | None = None
    explanation: str = ""
    recommendations: list[str] = field(default_factory=list)
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    def summary(self) -> dict[str, Any]:
        return {
            "site_name": self.site_name, "lat": self.lat, "lon": self.lon, "timestamp": self.timestamp,
            "combined_score": self.combined_score, "grade": self.grade, "status": status_text(self.combined_score),
            "waste_score": self.waste_score, "water_score": self.water_score, "class_counts": self.class_counts,
            "water_label": self.water_label, "water_probability": self.water_probability, "recommendations": self.recommendations,
        }


def recommendations(class_counts: Mapping[str, int], checks: pd.DataFrame | None, combined: float) -> list[str]:
    """Rule-based, prioritised recommendations from detected waste and failed checks."""
    recs: list[str] = []
    if combined < 40:
        recs.append("Overall condition is critical: notify the environmental authority and schedule an urgent multi-disciplinary survey.")
    elif combined < 55:
        recs.append("Condition is poor: plan a clean-up and re-test within 2 weeks.")
    for cls in sorted(class_counts, key=lambda k: -class_counts[k]):
        if cls in WASTE_ADVICE:
            recs.append(WASTE_ADVICE[cls])
    if checks is not None and not checks.empty:
        for status in ("red", "amber"):
            for p in checks.loc[checks["status"] == status, "parameter"]:
                if p in PARAM_ADVICE and PARAM_ADVICE[p] not in recs:
                    recs.append(("[urgent] " if status == "red" else "[monitor] ") + PARAM_ADVICE[p])
    if not recs:
        recs.append("No significant issues found: continue routine monitoring (monthly water tests, quarterly visual surveys).")
    return recs


def build_report(cfg: Mapping[str, Any], site_name: str, lat: float | None, lon: float | None, detection: Any | None = None, assessment: Any | None = None, notes: str = "") -> OceanHealthReport:
    """Assemble the report from a ``DetectionResult`` and/or ``WaterAssessment``."""
    ws = wq = None
    counts: dict[str, int] = {}
    n = 0
    if detection is not None:
        counts, n = dict(detection.class_counts), int(detection.n)
        ws = waste_score(n, detection.coverage(), cfg)
    if assessment is not None:
        wq = float(assessment.habitability_score)
    combined = combine_scores(ws, wq, cfg)
    checks = assessment.checks if assessment is not None else None
    return OceanHealthReport(
        site_name=site_name or "Unnamed site", lat=lat, lon=lon, combined_score=combined, grade=grade_for(combined, cfg),
        waste_score=ws, water_score=wq, class_counts=counts, n_detections=n,
        water_label=getattr(assessment, "label", None), water_probability=getattr(assessment, "probability", None),
        checks=checks, explanation=getattr(assessment, "explanation", ""), recommendations=recommendations(counts, checks, combined), notes=notes,
    )


# ---------------------------------------------------------------- PDF
def _latin1(text: str) -> str:
    """fpdf core fonts are latin-1 only: strip markdown + replace unsupported characters."""
    t = str(text).replace("**", "").replace("*", "").replace("–", "-").replace("—", "-").replace("≤", "<=").replace("≥", ">=").replace("µ", "u")
    return t.encode("latin-1", "replace").decode("latin-1")


def report_to_pdf(report: OceanHealthReport, annotated_bgr: np.ndarray | None = None) -> bytes:
    """Render the report to PDF bytes (fpdf2, core fonts, embedded PNG charts)."""
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_fill_color(4, 18, 31)
    pdf.rect(0, 0, 210, 32, "F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_xy(12, 9)
    pdf.cell(0, 8, "Ocean Health Report")
    pdf.set_font("Helvetica", "", 10)
    pdf.set_xy(12, 19)
    loc = f"  |  {report.lat:.4f}, {report.lon:.4f}" if report.lat is not None and report.lon is not None else ""
    pdf.cell(0, 6, _latin1(f"{report.site_name}{loc}  |  {report.timestamp}"))
    pdf.set_text_color(20, 20, 20)
    pdf.set_y(40)

    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 8, _latin1(f"Combined score: {report.combined_score:.0f}/100   Grade {report.grade}   ({status_text(report.combined_score)})"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    ws = "n/a" if report.waste_score is None else f"{report.waste_score:.0f}/100 ({report.n_detections} items detected)"
    wq = "n/a" if report.water_score is None else f"{report.water_score:.0f}/100 ({report.water_label}, P(potable) = {report.water_probability:.0%})"
    pdf.multi_cell(0, 6, _latin1(f"Waste cleanliness score: {ws}\nWater habitability score: {wq}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        if annotated_bgr is not None:
            p = tdp / "det.jpg"
            h, w = annotated_bgr.shape[:2]
            scale = 1000 / max(h, w)
            cv2.imwrite(str(p), cv2.resize(annotated_bgr, (int(w * scale), int(h * scale))) if scale < 1 else annotated_bgr)
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Waste detection", new_x="LMARGIN", new_y="NEXT")
            pdf.image(str(p), w=120)
            pdf.ln(2)
        if report.waste_score is not None:
            pdf.image(str(mpl_counts_png(report.class_counts, tdp / "counts.png")), w=120)
        if report.checks is not None and not report.checks.empty:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Water quality vs WHO/EPA guidelines", new_x="LMARGIN", new_y="NEXT")
            pdf.image(str(mpl_checks_png(report.checks, tdp / "checks.png")), w=170)
        if report.explanation:
            pdf.set_font("Helvetica", "B", 11)
            pdf.cell(0, 7, "Model explanation", new_x="LMARGIN", new_y="NEXT")
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 5, _latin1(report.explanation), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, "Recommendations", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for i, r in enumerate(report.recommendations, 1):
            pdf.multi_cell(0, 5, _latin1(f"{i}. {r}"), new_x="LMARGIN", new_y="NEXT")
        if report.notes:
            pdf.ln(2)
            pdf.set_font("Helvetica", "I", 9)
            pdf.multi_cell(0, 5, _latin1(f"Field notes: {report.notes}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(110, 110, 110)
        pdf.multi_cell(0, 4, _latin1("Automated screening report generated by ML models (YOLO11 + XGBoost). Not a substitute for accredited laboratory analysis. Classes: " + ", ".join(pretty(c) for c in report.class_counts) if report.class_counts else "Automated screening report generated by ML models (YOLO11 + XGBoost). Not a substitute for accredited laboratory analysis."), new_x="LMARGIN", new_y="NEXT")
        return bytes(pdf.output())
