from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from diagnose_layer4_labels import build_report


def test_label_diagnostic_counts_censoring_and_snapshot_cohort_changes():
    panel = pd.DataFrame({
        "golden_id": ["A", "B", "A", "C"],
        "as_of": ["2026-07-31", "2026-07-31", "2026-08-14", "2026-08-14"],
        "label_kept_30d": [1, 0, 1, 0],
    })
    promises = pd.DataFrame({
        "golden_id": ["A", "B", "C"],
        "promised_date": ["2026-08-01", "2026-08-02", "2026-08-15"],
        "status_date": ["2026-08-04", None, "2026-09-20"],
        "status": ["KEPT", "PENDING", "BROKEN"],
    })

    reports, _ = build_report(
        panel, promises, ["2026-07-31", "2026-08-14"], cutoff="2026-09-28"
    )

    assert reports[0]["rows"] == 2
    assert reports[0]["kept"] == 1
    assert reports[0]["broken"] == 1
    assert reports[0]["kept_rate"] == 0.5
    assert reports[0]["complete"] is True
    assert reports[0]["unresolved"] == 2
    assert reports[0]["unresolved_share"] == 2 / 3
    assert reports[0]["new_labeled_customers"] == 2
    assert reports[1]["new_labeled_customers"] == 1


def test_label_diagnostic_marks_snapshot_horizon_incomplete_after_cutoff():
    panel = pd.DataFrame({
        "golden_id": ["A"],
        "as_of": ["2026-09-01"],
        "label_kept_30d": [1],
    })
    promises = pd.DataFrame(columns=[
        "golden_id", "promised_date", "status_date", "status"
    ])

    reports, _ = build_report(
        panel, promises, ["2026-09-01"], cutoff="2026-09-28"
    )

    assert reports[0]["complete"] is False
