from __future__ import annotations

import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import layer4


def _temporal_panel() -> pd.DataFrame:
    rows = []
    for date in pd.date_range("2026-01-01", periods=12, freq="14D"):
        for idx in range(60):
            rows.append({
                "golden_id": f"G{idx}",
                "as_of": date,
                "label_kept_30d": idx % 2,
                "total_balance": float(idx * (date.day / 10)),
                "case_status": "open",
                "best_channel": "sms",
                "action_channel": "sms",
                "action_outcome": "answered" if idx % 2 else "no_answer",
            })
    return pd.DataFrame(rows)


def test_temporal_split_gaps_train_from_validation_and_holds_out_future():
    panel = _temporal_panel()
    train, validation, holdout, split = layer4._temporal_split(panel, "2026-04-09")

    assert train.as_of.max() < validation.as_of.min()
    assert (validation.as_of.min() - train.as_of.max()).days >= 30
    assert (holdout.as_of.min() - validation.as_of.max()).days >= 30
    assert split["label_horizon_days"] == "30"


def test_recency_weights_are_positive_normalized_and_newer_heavier():
    weights = layer4._recency_weights(pd.Series(pd.to_datetime([
        "2026-01-01", "2026-04-01",
    ])))

    assert np.all(weights > 0)
    assert weights.mean() == pytest.approx(1)
    assert weights[1] > weights[0]


def test_rolling_origin_validation_uses_expanding_past_only_folds():
    panel = _temporal_panel()
    development = panel.loc[panel.as_of <= pd.Timestamp("2026-04-23")]

    results = layer4._rolling_origin_validation(development)

    assert results
    assert {row["model"] for row in results} == {"logistic", "lightgbm_15"}
    assert all(pd.Timestamp(row["origin"]) <= development.as_of.max() for row in results)


def test_calibration_uses_validation_arrays_and_outputs_bounded_probabilities():
    labels = pd.Series([0, 1] * 40)
    probabilities = np.linspace(0.05, 0.95, len(labels))
    method, calibrator, calibrated, details = layer4._fit_calibrator(probabilities, labels)

    assert method in {"sigmoid", "isotonic"}
    assert details["selected_on_validation_only"] is True
    assert details["validation_fit_rows"] + details["validation_selection_rows"] == len(labels)
    assert np.all((0 <= calibrated) & (calibrated <= 1))
    holdout = layer4._apply_calibrator(method, calibrator, np.array([-2.0, 2.0]))
    assert np.all((0 <= holdout) & (holdout <= 1))


def test_calibration_compares_isotonic_when_validation_sample_is_large_enough():
    labels = pd.Series([0, 1] * 250)
    probabilities = np.linspace(0.05, 0.95, len(labels))
    method, _, calibrated, details = layer4._fit_calibrator(probabilities, labels)

    assert method in {"sigmoid", "isotonic"}
    assert "isotonic" in details["validation_brier_by_method"]
    assert np.all((0 <= calibrated) & (calibrated <= 1))


def test_observational_audit_labels_nonrandomized_and_reports_score_groups():
    frame = pd.DataFrame({
        "predicted_kept": np.linspace(0.01, 0.99, 100),
        "label_kept_30d": [0, 1] * 50,
        "action_outcome": ["answered", "no_answer"] * 50,
        "action_channel": ["sms", "call"] * 50,
        "recommended_action": ["AGENT_REVIEW"] * 100,
    })

    result = layer4._observational_action_audit(frame)

    assert "OBSERVATIONAL / NON-RANDOMIZED" in result["interpretation"]
    assert len(result["score_deciles"]) == 10
    assert sum(bucket["count"] for bucket in result["score_deciles"]) == 100


def test_explanation_includes_confidence_reason_fallback_and_safety_override():
    action, rationale = layer4.recommend_action(pd.Series({
        "hardship_flag": True, "consent_phone": True,
    }))
    explanation = layer4._recommendation_details(
        pd.Series({"hardship_flag": True, "consent_phone": True}),
        0.9,
        [{"feature": "gender", "contribution": 100}],
    )

    assert action == "HUMAN_HARDSHIP_REVIEW"
    assert rationale
    assert explanation["action"] == action
    assert 0 <= explanation["confidence"] <= 1
    assert explanation["reason_codes"][0]["type"] == "safety_rule"
    assert "gender" not in json.dumps(explanation).lower()
    assert explanation["next_best_option"]


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"case_status": "disputed", "consent_phone": True}, "HOLD_FOR_CASE_REVIEW"),
        ({"outbound_calls_7d": 3, "consent_phone": True}, "NO_FURTHER_CALLS_7D"),
        ({"consent_sms": False, "consent_phone": False}, "NO_CONSENTING_CHANNEL"),
    ],
)
def test_explanations_preserve_non_hardship_safety_overrides(row, expected):
    explanation = layer4._recommendation_details(pd.Series(row), 0.99)

    assert explanation["action"] == expected
    assert explanation["reason_codes"][0]["type"] == "safety_rule"


def test_low_confidence_uses_human_review_and_missing_consent_never_selects_channel():
    low = layer4._recommendation_details(pd.Series({
        "consent_sms": True, "best_channel": "sms",
    }), 0.51)
    no_consent = layer4._recommendation_details(pd.Series({
        "consent_sms": False, "consent_phone": False, "consent_email": False,
    }), 0.9)

    assert low["action"] == "AGENT_REVIEW"
    assert no_consent["action"] == "NO_CONSENTING_CHANNEL"


def test_protected_and_proxy_fields_are_rejected_and_do_not_change_safety_decision():
    baseline = layer4._recommendation_details(pd.Series({
        "consent_sms": True, "best_channel": "sms",
    }), 0.8)
    with_protected_value = layer4._recommendation_details(pd.Series({
        "consent_sms": True, "best_channel": "sms", "gender_code": "protected-value",
    }), 0.8)

    assert baseline["action"] == with_protected_value["action"]
    with pytest.raises(ValueError, match="Restricted columns"):
        layer4._model_matrix(pd.DataFrame({
            "accessibility_needs_flag": [True],
        }))


def test_fairness_audit_suppresses_small_groups_and_keeps_protected_fields_audit_only(tmp_path, monkeypatch):
    db_path = tmp_path / "restricted.duckdb"
    with duckdb.connect(str(db_path)) as con:
        con.execute("""
            CREATE TABLE c360_protected_restricted(
                golden_id VARCHAR, gender VARCHAR, newcomer BOOLEAN,
                accessibility BOOLEAN, vulnerability BOOLEAN
            )
        """)
        con.execute("""
            INSERT INTO c360_protected_restricted
            SELECT 'G' || i, CASE WHEN i < 50 THEN 'A' ELSE 'B' END,
                   false, false, false FROM range(100) t(i)
        """)
    monkeypatch.setattr(layer4, "C360DB", db_path)
    preds = pd.DataFrame({
        "golden_id": [f"G{i}" for i in range(100)],
        "label_kept_30d": [i % 2 for i in range(100)],
        "predicted_kept": [0.4 + (i % 2) * 0.2 for i in range(100)],
        "recommended_action": ["AGENT_REVIEW"] * 100,
    })
    override_log = tmp_path / "audit.jsonl"
    override_log.write_text(json.dumps({
        "golden_id": "G1", "suggested_action": "AGENT_REVIEW", "final_action": "HOLD",
    }) + "\n", encoding="utf-8")
    result = layer4._fairness_audit(preds, override_log)

    assert result["restricted_data_used_for_audit_only"] is True
    assert all(not group["suppressed"] for group in result["groups"]["gender"])
    assert result["groups"]["gender"][0]["override_rate"] > 0
    matrix = layer4._model_matrix(pd.DataFrame({
        "golden_id": ["G1", "G2"], "total_balance": [1, 2],
    }))
    assert "gender" not in matrix.columns
    with pytest.raises(ValueError, match="Restricted columns"):
        layer4._model_matrix(pd.DataFrame({"gender": ["A"]}))
