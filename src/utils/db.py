"""SQLite history store for detections, water tests and combined site reports."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import pandas as pd

from src.utils.common import load_config, resolve_path

SCHEMA = """
CREATE TABLE IF NOT EXISTS waste_detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    site_name TEXT,
    lat REAL,
    lon REAL,
    source TEXT,
    n_detections INTEGER NOT NULL,
    class_counts TEXT NOT NULL,
    mean_confidence REAL,
    coverage REAL,
    waste_score REAL,
    enhanced INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS water_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    site_name TEXT,
    lat REAL,
    lon REAL,
    params TEXT NOT NULL,
    potable INTEGER NOT NULL,
    probability REAL NOT NULL,
    habitability_score REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS site_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    site_name TEXT NOT NULL,
    lat REAL,
    lon REAL,
    waste_score REAL,
    water_score REAL,
    combined_score REAL NOT NULL,
    grade TEXT,
    report TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class HistoryDB:
    """Small wrapper around a SQLite file. Safe to create per request (connections are short-lived)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else resolve_path(load_config()["paths"]["db"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------ writes
    def log_waste(self, n_detections: int, class_counts: Mapping[str, int], mean_confidence: float | None = None, coverage: float | None = None, waste_score: float | None = None, site_name: str | None = None, lat: float | None = None, lon: float | None = None, source: str | None = None, enhanced: bool = False) -> int:
        """Insert a waste-detection record; returns its id."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO waste_detections (timestamp, site_name, lat, lon, source, n_detections, class_counts, mean_confidence, coverage, waste_score, enhanced) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (_now(), site_name, lat, lon, source, int(n_detections), json.dumps(dict(class_counts)), mean_confidence, coverage, waste_score, int(enhanced)),
            )
            return int(cur.lastrowid)

    def log_water(self, params: Mapping[str, Any], potable: bool, probability: float, habitability_score: float, site_name: str | None = None, lat: float | None = None, lon: float | None = None) -> int:
        """Insert a water-test record; returns its id."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO water_tests (timestamp, site_name, lat, lon, params, potable, probability, habitability_score) VALUES (?,?,?,?,?,?,?,?)",
                (_now(), site_name, lat, lon, json.dumps(dict(params), default=float), int(bool(potable)), float(probability), float(habitability_score)),
            )
            return int(cur.lastrowid)

    def log_report(self, site_name: str, combined_score: float, grade: str, waste_score: float | None, water_score: float | None, lat: float | None = None, lon: float | None = None, report: Mapping[str, Any] | None = None) -> int:
        """Insert a combined site report; returns its id."""
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO site_reports (timestamp, site_name, lat, lon, waste_score, water_score, combined_score, grade, report) VALUES (?,?,?,?,?,?,?,?,?)",
                (_now(), site_name, lat, lon, waste_score, water_score, float(combined_score), grade, json.dumps(dict(report or {}), default=str)),
            )
            return int(cur.lastrowid)

    # ------------------------------------------------------------------ reads
    def _df(self, table: str) -> pd.DataFrame:
        if table not in ("waste_detections", "water_tests", "site_reports"):
            raise ValueError(f"Unknown table {table}")
        with self._conn() as c:
            df = pd.read_sql_query(f"SELECT * FROM {table} ORDER BY timestamp", c)  # noqa: S608 - whitelisted
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def waste_history(self) -> pd.DataFrame:
        return self._df("waste_detections")

    def water_history(self) -> pd.DataFrame:
        return self._df("water_tests")

    def report_history(self) -> pd.DataFrame:
        return self._df("site_reports")

    def waste_class_totals(self) -> dict[str, int]:
        """Sum of class counts across all logged detections."""
        totals: dict[str, int] = {}
        for raw in self.waste_history().get("class_counts", []):
            for k, v in json.loads(raw).items():
                totals[k] = totals.get(k, 0) + int(v)
        return dict(sorted(totals.items(), key=lambda kv: -kv[1]))

    def sites(self) -> pd.DataFrame:
        """All geo-tagged records as ``site_name, lat, lon, score, kind, timestamp``."""
        parts = []
        r = self.report_history()
        if not r.empty:
            parts.append(r.assign(score=r["combined_score"], kind="site report")[["site_name", "lat", "lon", "score", "kind", "timestamp"]])
        w = self.water_history()
        if not w.empty:
            parts.append(w.assign(score=w["habitability_score"], kind="water test")[["site_name", "lat", "lon", "score", "kind", "timestamp"]])
        d = self.waste_history()
        if not d.empty and "waste_score" in d:
            parts.append(d.assign(score=d["waste_score"], kind="waste scan")[["site_name", "lat", "lon", "score", "kind", "timestamp"]])
        if not parts:
            return pd.DataFrame(columns=["site_name", "lat", "lon", "score", "kind", "timestamp"])
        out = pd.concat(parts, ignore_index=True).dropna(subset=["lat", "lon", "score"])
        out["site_name"] = out["site_name"].fillna("Unnamed site")
        return out

    def clear(self) -> None:
        """Delete all records (used by tests / the dashboard 'reset' button)."""
        with self._conn() as c:
            for t in ("waste_detections", "water_tests", "site_reports"):
                c.execute(f"DELETE FROM {t}")  # noqa: S608
