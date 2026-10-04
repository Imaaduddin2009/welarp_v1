"""Layer 4: temporal promise-outcome scoring with safety-gated human decision support."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import layer3 as F

L2DB = Path(os.getenv("C360_L2_DB", ROOT / "out" / "l2_maple.duckdb"))
C360DB = Path(os.getenv("C360_C360_DB", ROOT / "out" / "c360_maple.duckdb"))
MIN_FAIRNESS_GROUP = 50
FORBIDDEN_MODEL_FIELDS = {
    "gender", "marital_status", "citizenship_status", "newcomer",
    "accessibility", "vulnerability", "location", "province_code",
    "postal_code", "postal", "accent", "first_name", "last_name",
    "date_of_birth", "dob", "email", "phone", "national_id_hash",
}

F.SRC = str(L2DB)


def _truthy(value: object) -> bool:
    return False if pd.isna(value) else bool(value)


def action_panel(as_of_dates: list[str], horizon: int = 30) -> pd.DataFrame:
    """Join each customer's recent observed contact action to a strictly future label."""
    if not as_of_dates:
        raise ValueError("At least one as_of date is required")
    con = duckdb.connect(str(L2DB), read_only=True)
    panels = []
    try:
        for as_of in sorted(set(as_of_dates)):
            features = F.training_set(as_of, horizon)
            actions = con.execute(f"""
                SELECT golden_id,
                       arg_max(channel, contact_date) AS action_channel,
                       arg_max(outcome, contact_date) AS action_outcome,
                       max(contact_date) AS action_date,
                       count(*) FILTER (
                         WHERE direction='outbound' AND channel IN ('call', 'phone')
                           AND contact_date >= DATE '{as_of}' - INTERVAL 7 DAY
                           AND contact_date < DATE '{as_of}'
                       ) AS outbound_calls_7d
                FROM contacts
                WHERE contact_date >= DATE '{as_of}' - INTERVAL 7 DAY
                  AND contact_date < DATE '{as_of}'
                GROUP BY golden_id
            """).fetchdf()
            panel = features.merge(actions, on="golden_id", how="inner", validate="one_to_one")
            panel["as_of"] = pd.Timestamp(as_of)
            panels.append(panel)
    finally:
        con.close()
    result = pd.concat(panels, ignore_index=True)
    if result.empty:
        raise ValueError("No contact actions with an observed future promise outcome were found")
    return result


def recommend_action(row: pd.Series) -> tuple[str, str]:
    """Return a conservative workflow route; never initiate contact or offer terms."""
    if _truthy(row.get("hardship_flag", False)) or _truthy(row.get("hardship_review_needed", False)):
        return "HUMAN_HARDSHIP_REVIEW", "Hardship is flagged or suggested by masked text; a trained person must review."
    if str(row.get("case_status", "")).lower() in {"disputed", "on_hold", "closed", "cured"}:
        return "HOLD_FOR_CASE_REVIEW", "Case status requires review before any contact."
    attempts = row.get("outbound_calls_7d", 0)
    if not pd.isna(attempts) and int(attempts) >= 3:
        return "NO_FURTHER_CALLS_7D", "Three or more outbound calls are already recorded in the rolling seven-day window."
    channel = str(row.get("best_channel", "")).lower()
    consent = {
        "call": row.get("consent_phone"),
        "phone": row.get("consent_phone"),
        "sms": row.get("consent_sms"),
        "email": row.get("consent_email"),
    }
    if not any(_truthy(value) for value in consent.values()):
        return "NO_CONSENTING_CHANNEL", "No recorded contact channel has consent; do not initiate outbound contact."
    if channel in consent and _truthy(consent[channel]):
        return f"AGENT_REVIEW_{channel.upper()}", "Suggested channel is consented; the agent must verify policy and contact history."
    return "AGENT_REVIEW", "No safe preferred channel is established; supervisor or agent review is required."


def _model_matrix(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    excluded = {"golden_id", "as_of", "action_date", "label_kept_30d"}
    inputs = frame.drop(columns=[c for c in excluded if c in frame.columns])
    prohibited = FORBIDDEN_MODEL_FIELDS.intersection(map(str.lower, inputs.columns))
    if prohibited:
        raise ValueError(f"Restricted columns reached the model matrix: {sorted(prohibited)}")
    matrix = pd.get_dummies(inputs, dummy_na=True, dtype=float)
    matrix = matrix.astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if columns is not None:
        matrix = matrix.reindex(columns=columns, fill_value=0.0)
    return matrix


def _metrics(y: pd.Series, probability: np.ndarray) -> dict[str, float | None]:
    bins = np.digitize(np.clip(probability, 0, 1), np.linspace(0, 1, 11)[1:-1], right=True)
    calibration = []
    ece = 0.0
    for index in range(10):
        mask = bins == index
        if not mask.any():
            continue
        predicted_mean = float(probability[mask].mean())
        observed_mean = float(y.iloc[np.flatnonzero(mask)].mean())
        ece += float(mask.mean()) * abs(predicted_mean - observed_mean)
        calibration.append({
            "bin": index,
            "count": int(mask.sum()),
            "mean_predicted": predicted_mean,
            "observed_kept_rate": observed_mean,
        })
    return {
        "roc_auc": float(roc_auc_score(y, probability)) if y.nunique() > 1 else None,
        "brier_score": float(brier_score_loss(y, probability)),
        "expected_calibration_error": ece,
        "observed_kept_rate": float(y.mean()),
        "predicted_kept_rate": float(probability.mean()),
        "calibration_bins": calibration,
    }


def _fairness_audit(predictions: pd.DataFrame) -> dict[str, object]:
    if not C360DB.is_file():
        return {"available": False, "reason": f"Restricted audit database not found: {C360DB}"}
    con = duckdb.connect(str(C360DB), read_only=True)
    try:
        restricted = con.execute("""
            SELECT golden_id, gender, newcomer, accessibility, vulnerability
            FROM c360_protected_restricted
        """).fetchdf()
    finally:
        con.close()
    merged = predictions.merge(restricted, on="golden_id", how="left", validate="many_to_one")
    groups: dict[str, object] = {}
    for field in ("gender", "newcomer", "accessibility", "vulnerability"):
        field_results = []
        for value, rows in merged.groupby(field, dropna=False):
            count = len(rows)
            if count < MIN_FAIRNESS_GROUP:
                field_results.append({"group": str(value), "count": count, "suppressed": True})
                continue
            positive = rows.label_kept_30d.astype(int)
            prediction = rows.predicted_kept >= 0.5
            tp = int(((prediction) & (positive == 1)).sum())
            fp = int(((prediction) & (positive == 0)).sum())
            fn = int(((~prediction) & (positive == 1)).sum())
            tn = int(((~prediction) & (positive == 0)).sum())
            field_results.append({
                "group": str(value),
                "count": count,
                "suppressed": False,
                "observed_kept_rate": float(positive.mean()),
                "predicted_positive_rate": float(prediction.mean()),
                "true_positive_rate": tp / (tp + fn) if tp + fn else None,
                "false_positive_rate": fp / (fp + tn) if fp + tn else None,
            })
        groups[field] = field_results
    return {
        "available": True,
        "restricted_data_used_for_audit_only": True,
        "minimum_group_size": MIN_FAIRNESS_GROUP,
        "groups": groups,
    }


def run_model(as_of_dates: list[str], train_through: str, output_dir: Path) -> dict[str, object]:
    """Train a LightGBM ranker against a logistic baseline using a strict temporal split."""
    from lightgbm import LGBMClassifier

    output_dir.mkdir(parents=True, exist_ok=True)
    panel = action_panel(as_of_dates)
    train_mask = panel.as_of <= pd.Timestamp(train_through)
    test_mask = panel.as_of > pd.Timestamp(train_through)
    train, test = panel.loc[train_mask].copy(), panel.loc[test_mask].copy()
    if train.empty or test.empty:
        raise ValueError("Temporal split must contain both training and later holdout rows")
    if train.as_of.max() >= test.as_of.min():
        raise ValueError("Temporal evaluation overlaps training and holdout dates")
    if train.label_kept_30d.nunique() < 2 or test.label_kept_30d.nunique() < 2:
        raise ValueError("Training and holdout must each contain both promise outcomes")

    x_train = _model_matrix(train)
    x_test = _model_matrix(test, list(x_train.columns))
    y_train = train.label_kept_30d.astype(int)
    y_test = test.label_kept_30d.astype(int)

    baseline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    baseline.fit(x_train, y_train)
    baseline_probability = baseline.predict_proba(x_test)[:, 1]

    model = LGBMClassifier(
        n_estimators=250,
        learning_rate=0.04,
        num_leaves=31,
        class_weight="balanced",
        random_state=42,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    scored = test[["golden_id", "as_of", "label_kept_30d"]].copy()
    scored["predicted_kept"] = probability
    scored["baseline_predicted_kept"] = baseline_probability
    scored[["recommended_action", "action_rationale"]] = test.apply(
        lambda row: pd.Series(recommend_action(row)), axis=1
    )
    scored.to_csv(output_dir / "layer4_scored_actions.csv", index=False)

    contributions = model.booster_.predict(x_test.iloc[: min(200, len(x_test))], pred_contrib=True)
    feature_names = list(x_test.columns)
    explanation_rows = []
    sample = scored.iloc[: len(contributions)]
    for row_index, values in enumerate(contributions):
        top = np.argsort(np.abs(values[:-1]))[::-1][:5]
        explanation_rows.append({
            "golden_id": sample.iloc[row_index].golden_id,
            "as_of": str(sample.iloc[row_index].as_of),
            "baseline_logit": float(values[-1]),
            "top_contributions": json.dumps([
                {"feature": feature_names[index], "contribution": float(values[index])}
                for index in top
            ]),
        })
    pd.DataFrame(explanation_rows).to_csv(output_dir / "layer4_explanations.csv", index=False)
    model.booster_.save_model(str(output_dir / "layer4_lightgbm.txt"))

    top_decile = scored.predicted_kept >= scored.predicted_kept.quantile(0.9)
    metrics = {
        "model": _metrics(y_test, probability),
        "logistic_baseline": _metrics(y_test, baseline_probability),
        "holdout_rows": len(test),
        "train_rows": len(train),
        "train_as_of_max": str(train.as_of.max().date()),
        "holdout_as_of_min": str(test.as_of.min().date()),
        "top_decile_observed_kept_rate": float(y_test[top_decile].mean()),
        "overall_observed_kept_rate": float(y_test.mean()),
        "top_decile_is_ranking_yield_not_causal_uplift": True,
        "fairness_audit": _fairness_audit(scored),
    }
    (output_dir / "layer4_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    submission_dir = ROOT / "submission"
    submission_dir.mkdir(parents=True, exist_ok=True)
    (submission_dir / "fairness_report.json").write_text(
        json.dumps(metrics["fairness_audit"], indent=2), encoding="utf-8"
    )
    report = [
        "# Layer 4 temporal model and decision-support report",
        "",
        f"- Label: first promise outcome recorded within 30 days after an as-of date (kept=1, broken=0).",
        f"- Training rows: {len(train):,}; latest training as-of: {metrics['train_as_of_max']}.",
        f"- Holdout rows: {len(test):,}; earliest holdout as-of: {metrics['holdout_as_of_min']}.",
        f"- LightGBM ROC AUC: {metrics['model']['roc_auc']}; Brier: {metrics['model']['brier_score']:.4f}.",
        f"- Logistic baseline ROC AUC: {metrics['logistic_baseline']['roc_auc']}; Brier: {metrics['logistic_baseline']['brier_score']:.4f}.",
        f"- Top-decile observed kept rate: {metrics['top_decile_observed_kept_rate']:.3f}; this is ranking yield, not causal uplift or policy value.",
        "- The training target is conditional on observed collections/contact activity and cannot estimate counterfactual action effects.",
        "- The LightGBM model uses Layer 2 operational features only. Restricted demographics are joined only in the separate group fairness audit.",
        "- Recommendations are deterministic safety gates, not autonomous contact or offer actions. Hardship, disputes, consent, and rolling call limits route to human review or suppression.",
        "- Per-record explanations are model contribution summaries and do not establish causal reasons.",
        "- Fairness groups below the configured minimum size are suppressed; group gaps require human compliance review.",
        "",
        "Artifacts: layer4_scored_actions.csv, layer4_explanations.csv, layer4_lightgbm.txt, layer4_metrics.json, fairness_report.json.",
    ]
    (submission_dir / "layer4_model_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return metrics


def record_override(
    log_path: Path,
    golden_id: str,
    suggested_action: str,
    final_action: str,
    reviewer: str,
    reason: str,
) -> None:
    """Append a reviewer decision with a required rationale to the immutable audit trail."""
    if not all(value.strip() for value in (golden_id, suggested_action, final_action, reviewer, reason)):
        raise ValueError("Override entries require customer, actions, reviewer, and reason")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": time.time(),
        "golden_id": golden_id,
        "suggested_action": suggested_action,
        "final_action": final_action,
        "reviewer": reviewer,
        "reason": reason,
    }
    with log_path.open("a", encoding="utf-8") as audit:
        audit.write(json.dumps(event, ensure_ascii=False) + "\n")
