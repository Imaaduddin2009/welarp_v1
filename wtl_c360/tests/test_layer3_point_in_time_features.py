from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import layer3


def _feature_db(path: Path) -> None:
    with duckdb.connect(str(path)) as con:
        con.execute("""
            CREATE TABLE cases (
                golden_id VARCHAR, case_id VARCHAR, stage VARCHAR, status VARCHAR,
                status_date DATE, opened_date DATE, days_case_open INTEGER,
                hardship_flag BOOLEAN, hardship_start_date DATE
            );
            INSERT INTO cases VALUES
                ('G1', 'C1', 'early', 'open', DATE '2024-01-01', DATE '2023-01-01', 0, false, NULL),
                ('G2', 'C2', 'early', 'open', DATE '2024-01-01', DATE '2023-01-01', 0, false, NULL);
            CREATE TABLE customers (
                golden_id VARCHAR, consent_as_of_date DATE, consent_sms BOOLEAN,
                consent_email BOOLEAN, consent_phone BOOLEAN, hardship_flag BOOLEAN
            );
            INSERT INTO customers VALUES
                ('G1', DATE '2023-01-01', false, false, false, false),
                ('G2', DATE '2023-01-01', false, false, false, false);
            CREATE TABLE accounts (
                account_id VARCHAR, golden_id VARCHAR, balance DOUBLE,
                overdue_amount DOUBLE, days_past_due INTEGER, snapshot_date DATE
            );
            INSERT INTO accounts VALUES
                ('A1', 'G1', 100, 10, 20, DATE '2024-01-02'),
                ('A1', 'G1', 150, 20, 60, DATE '2024-04-01'),
                ('A1', 'G1', 999, 999, 999, DATE '2024-04-02'),
                ('A2', 'G2', 50, 0, 0, DATE '2024-04-01');
            CREATE TABLE promises (
                golden_id VARCHAR, promised_date DATE, status_date DATE, status VARCHAR
            );
            CREATE TABLE contacts (
                golden_id VARCHAR, contact_date DATE, outcome VARCHAR,
                direction VARCHAR, channel VARCHAR, note_id VARCHAR
            );
            CREATE TABLE notes (note_id VARCHAR, note_text_masked VARCHAR);
            CREATE TABLE bureau_scores (
                golden_id VARCHAR, score_date DATE, bureau_score DOUBLE
            );
            INSERT INTO bureau_scores VALUES
                ('G1', DATE '2024-01-02', 600),
                ('G1', DATE '2024-03-01', 650),
                ('G1', DATE '2024-04-02', 999),
                ('G2', DATE '2024-03-01', 700);
        """)


def test_temporal_change_features_are_point_in_time_and_shared(tmp_path, monkeypatch):
    source = tmp_path / "features.duckdb"
    _feature_db(source)
    monkeypatch.setattr(layer3, "SRC", str(source))
    as_of = "2024-04-01"

    bulk = layer3.compute(as_of).set_index("golden_id")
    g1 = bulk.loc["G1"]
    g2 = bulk.loc["G2"]

    assert g1.balance_change_90d == 50
    assert g1.max_days_past_due_change_90d == 40
    assert g1.bureau_score_change_90d == 50
    assert pd_isna(g2.balance_change_90d)
    assert pd_isna(g2.max_days_past_due_change_90d)
    assert pd_isna(g2.bureau_score_change_90d)

    live = layer3.online("G1", as_of)
    assert live["balance_change_90d"] == g1.balance_change_90d
    assert live["max_days_past_due_change_90d"] == g1.max_days_past_due_change_90d
    assert live["bureau_score_change_90d"] == g1.bureau_score_change_90d
    assert layer3.test_point_in_time(as_of).startswith(f"as_of {as_of}:")
    assert "bulk == live" in layer3.test_parity(n=25)


def pd_isna(value: object) -> bool:
    import pandas as pd

    return bool(pd.isna(value))


@pytest.mark.parametrize(
    "feature",
    ["balance_change_90d", "max_days_past_due_change_90d", "bureau_score_change_90d"],
)
def test_temporal_features_are_registered(feature):
    assert feature in layer3.REGISTRY
