"""Layer 4: temporal promise-outcome scoring with safety-gated human decision support."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

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


def discover_as_of_dates(horizon: int = 30, cadence_days: int = 14) -> list[str]:
    """Create a 14-day schedule bounded by observed case, contact, and label data."""
    if not L2DB.is_file():
        raise FileNotFoundError(f"Layer 2 feature database not found: {L2DB}")
    with duckdb.connect(str(L2DB), read_only=True) as con:
        start = con.execute("""
            SELECT greatest(min(case_date), min(promise_date))
            FROM (SELECT min(opened_date) AS case_date FROM cases),
                 (SELECT min(promised_date) AS promise_date FROM promises)
        """).fetchone()[0]
        latest_outcome = con.execute("""
            SELECT max(status_date) FROM promises
            WHERE status IN ('KEPT', 'BROKEN')
        """).fetchone()[0]
        has_run_table = con.execute("""
            SELECT count(*) FROM information_schema.tables
            WHERE table_name='pipeline_run'
        """).fetchone()[0] > 0
        cutoff = (
            con.execute("SELECT max(data_snapshot_date) FROM pipeline_run").fetchone()[0]
            if has_run_table else None
        )
    if start is None or latest_outcome is None:
        raise ValueError("No case/promise history is available to discover usable snapshots")
    start_date = pd.Timestamp(start).normalize()
    last_date = pd.Timestamp(cutoff or latest_outcome).normalize() - pd.Timedelta(days=horizon)
    if last_date < start_date:
        raise ValueError("No as-of date has a fully observed label horizon")
    return [date.strftime("%Y-%m-%d") for date in pd.date_range(
        start=start_date, end=last_date, freq=f"{cadence_days}D"
    )]


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


def _recommendation_details(
    row: pd.Series,
    probability: float,
    contributions: list[dict[str, object]] | None = None,
    low_confidence_threshold: float = 0.55,
) -> dict[str, object]:
    """Explain model confidence separately from deterministic safety routing."""
    action, rationale = recommend_action(row)
    safety_action = action != "AGENT_REVIEW" and not action.startswith("AGENT_REVIEW_")
    confidence = max(float(probability), 1.0 - float(probability))
    if not safety_action and confidence < low_confidence_threshold:
        action = "AGENT_REVIEW"
        rationale = "Model confidence is low; use human review instead of a model-led route."

    safe_names = {
        "balance_change_90d": "recent balance change",
        "max_days_past_due_change_90d": "recent days-past-due change",
        "bureau_score_change_90d": "recent bureau-score change",
        "max_days_past_due": "days past due",
        "total_overdue_amount": "overdue balance",
        "promises_broken_90d": "recent broken promises",
        "promise_kept_rate_90d": "recent promise-kept rate",
        "contact_attempts_14d": "recent contact attempts",
        "days_since_last_promise": "time since last promise",
    }
    model_reasons = []
    for item in contributions or []:
        feature = str(item.get("feature", ""))
        if feature not in safe_names:
            continue
        contribution = float(item.get("contribution", 0))
        direction = "increased" if contribution > 0 else "decreased"
        model_reasons.append(
            f"{safe_names[feature]} {direction} the model's kept-probability estimate"
        )
    if safety_action:
        reason_codes = [{"type": "safety_rule", "reason": rationale}]
    else:
        reason_codes = [
            {"type": "model_reason", "reason": reason}
            for reason in model_reasons[:3]
        ]
        if not reason_codes:
            reason_codes = [{
                "type": "model_reason",
                "reason": "The model probability is based on operational features; no safe top contribution was available.",
            }]
    return {
        "action": action,
        "confidence": confidence,
        "probability_kept": float(probability),
        "reason_codes": reason_codes,
        "next_best_option": "AGENT_REVIEW" if action != "AGENT_REVIEW" else "SUPERVISOR_REVIEW",
        "explanation_type": "safety_rule" if safety_action else "observational_model_reason_not_causal",
    }


def _model_matrix(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    import numpy as np

    excluded = {"golden_id", "as_of", "action_date", "label_kept_30d"}
    inputs = frame.drop(columns=[c for c in excluded if c in frame.columns])
    prohibited = {
        str(column) for column in inputs.columns
        if any(token in str(column).lower() for token in FORBIDDEN_MODEL_FIELDS)
    }
    if prohibited:
        raise ValueError(f"Restricted columns reached the model matrix: {sorted(prohibited)}")
    matrix = pd.get_dummies(inputs, dummy_na=True, dtype=float)
    matrix = matrix.astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if columns is not None:
        matrix = matrix.reindex(columns=columns, fill_value=0.0)
    return matrix


def _metrics(y: pd.Series, probability: np.ndarray) -> dict[str, float | None]:
    import numpy as np
    from sklearn.metrics import brier_score_loss, roc_auc_score

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


def _recency_weights(as_of: pd.Series, half_life_days: int = 90) -> np.ndarray:
    """Exponential recency weights: 0.5 ** (age in days / 90), normalized to mean one."""
    import numpy as np

    dates = pd.to_datetime(as_of)
    age = (dates.max() - dates).dt.total_seconds().to_numpy() / 86400
    weights = np.power(0.5, age / half_life_days)
    return weights / weights.mean()


def _temporal_split(
    panel: pd.DataFrame,
    cutoff: str,
    horizon: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Use last pre-cutoff snapshot for validation and later snapshots for untouched holdout."""
    dates = sorted(pd.to_datetime(panel.as_of).dt.normalize().unique())
    cutoff_date = pd.Timestamp(cutoff)
    validation_dates = [date for date in dates if date <= cutoff_date]
    if len(validation_dates) < 2:
        raise ValueError("Temporal split needs two pre-cutoff snapshots and a later holdout")
    validation_date = pd.Timestamp(validation_dates[-1])
    holdout_dates = [
        date for date in dates
        if date > cutoff_date and date >= validation_date + pd.Timedelta(days=horizon)
    ]
    if not holdout_dates:
        raise ValueError("Temporal split needs two pre-cutoff snapshots and a later holdout")
    train_end = validation_date - pd.Timedelta(days=horizon)
    train = panel.loc[pd.to_datetime(panel.as_of).lt(validation_date - pd.Timedelta(days=horizon - 1))].copy()
    validation = panel.loc[pd.to_datetime(panel.as_of).dt.normalize().eq(validation_date)].copy()
    holdout = panel.loc[pd.to_datetime(panel.as_of).dt.normalize().isin(holdout_dates)].copy()
    if train.empty or validation.empty or holdout.empty:
        raise ValueError("Temporal train/validation/holdout split produced an empty partition")
    return train, validation, holdout, {
        "train_as_of_max": str(pd.to_datetime(train.as_of).max().date()),
        "validation_as_of": str(validation_date.date()),
        "holdout_as_of_min": str(pd.to_datetime(holdout.as_of).min().date()),
        "train_validation_gap_days": str((validation_date - pd.to_datetime(train.as_of).max()).days),
        "label_horizon_days": str(horizon),
    }


def _rolling_origin_validation(
    panel: pd.DataFrame,
    min_train_days: int = 60,
    horizon: int = 30,
) -> list[dict[str, object]]:
    """Evaluate expanding-origin logistic and small LightGBM candidates on past origins."""
    import numpy as np
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    dates = sorted(pd.to_datetime(panel.as_of).dt.normalize().unique())
    results = []
    for origin in dates[-5:]:
        train = panel.loc[pd.to_datetime(panel.as_of).lt(origin - pd.Timedelta(days=horizon - 1))]
        valid = panel.loc[pd.to_datetime(panel.as_of).dt.normalize().eq(origin)]
        if train.empty or valid.empty or (origin - pd.to_datetime(train.as_of).max()).days < horizon:
            continue
        if train.label_kept_30d.nunique() < 2 or valid.label_kept_30d.nunique() < 2:
            continue
        if (origin - pd.to_datetime(train.as_of).min()).days < min_train_days:
            continue
        x_train = _model_matrix(train)
        x_valid = _model_matrix(valid, list(x_train.columns))
        y_train, y_valid = train.label_kept_30d.astype(int), valid.label_kept_30d.astype(int)
        weights = _recency_weights(train.as_of)
        models = {
            "logistic": make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            ),
            "lightgbm_15": LGBMClassifier(
                n_estimators=120, learning_rate=0.04, num_leaves=15,
                class_weight="balanced", random_state=42, verbosity=-1,
            ),
        }
        for name, model in models.items():
            fit_params = {"logisticregression__sample_weight": weights} if name == "logistic" else {"sample_weight": weights}
            model.fit(x_train, y_train, **fit_params)
            probability = model.predict_proba(x_valid)[:, 1]
            results.append({"origin": str(pd.Timestamp(origin).date()), "model": name,
                            "metrics": _metrics(y_valid, probability)})
    return results


def _fit_calibrator(
    validation_probability: np.ndarray,
    validation_labels: pd.Series,
) -> tuple[str, object, np.ndarray, dict[str, object]]:
    """Fit/select calibration on a deterministic validation-only split."""
    import numpy as np

    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(validation_labels, dtype=int)
    p = np.clip(np.asarray(validation_probability, dtype=float), 1e-6, 1 - 1e-6)
    if len(y) < 40 or len(np.unique(y)) < 2:
        return "none_insufficient_validation", None, p, {"selected_on_validation_only": True}
    order = np.random.default_rng(42).permutation(len(y))
    fit_idx, eval_idx = order[::2], order[1::2]
    if len(np.unique(y[fit_idx])) < 2 or len(np.unique(y[eval_idx])) < 2:
        fit_idx, eval_idx = order[: len(order) // 2], order[len(order) // 2 :]
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    methods: dict[str, tuple[object, np.ndarray]] = {}
    sigmoid = LogisticRegression(max_iter=1000, random_state=42)
    sigmoid.fit(logit[fit_idx], y[fit_idx])
    methods["sigmoid"] = (sigmoid, sigmoid.predict_proba(logit[eval_idx])[:, 1])
    if len(fit_idx) >= 200 and len(np.unique(y[fit_idx])) > 1:
        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(p[fit_idx], y[fit_idx])
        methods["isotonic"] = (isotonic, isotonic.predict(p[eval_idx]))
    from sklearn.metrics import brier_score_loss
    selected = min(methods, key=lambda name: brier_score_loss(y[eval_idx], methods[name][1]))
    if selected == "sigmoid":
        calibrated = methods[selected][0].predict_proba(logit)[:, 1]
    else:
        calibrated = methods[selected][0].predict(p)
    return selected, methods[selected][0], np.clip(calibrated, 0, 1), {
        "selected_on_validation_only": True,
        "validation_fit_rows": int(len(fit_idx)),
        "validation_selection_rows": int(len(eval_idx)),
        "validation_selection_indices": eval_idx.tolist(),
        "validation_brier_by_method": {
            name: float(brier_score_loss(y[eval_idx], values[1]))
            for name, values in methods.items()
        },
    }


def _apply_calibrator(method: str, calibrator: object, probability: np.ndarray) -> np.ndarray:
    import numpy as np
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    if method == "sigmoid":
        logits = np.log(p / (1 - p)).reshape(-1, 1)
        return np.clip(calibrator.predict_proba(logits)[:, 1], 0, 1)
    if method == "isotonic":
        return np.clip(calibrator.predict(p), 0, 1)
    return p


def _observational_action_audit(scored: pd.DataFrame) -> dict[str, object]:
    """Describe score-bucket outcomes; this is observational, not causal."""
    rows = scored.copy()
    rows["score_bucket"] = pd.qcut(
        rows["predicted_kept"].rank(method="first"),
        q=min(10, len(rows)),
        labels=False,
        duplicates="drop",
    )
    buckets = []
    for bucket, group in rows.groupby("score_bucket", dropna=False):
        buckets.append({
            "score_decile": int(bucket) + 1,
            "count": int(len(group)),
            "observed_kept_rate": float(group.label_kept_30d.mean()),
            "mean_predicted_kept": float(group.predicted_kept.mean()),
            "contact_outcomes": group.action_outcome.value_counts(dropna=False).astype(int).to_dict(),
            "channels": group.action_channel.value_counts(dropna=False).astype(int).to_dict(),
            "recommendations": group.recommended_action.value_counts(dropna=False).astype(int).to_dict(),
        })
    return {"interpretation": "OBSERVATIONAL / NON-RANDOMIZED; no causal uplift or treatment effect inferred",
            "score_deciles": buckets}


def _existing_metrics_comparison(path: Path) -> dict[str, object]:
    """Compare with previously persisted Layer 4 metrics without using them for fitting."""
    if not path.is_file():
        return {"available": False, "reason": "No prior Layer 4 metrics artifact found"}
    previous = json.loads(path.read_text(encoding="utf-8"))
    model = previous.get("model", {})
    baseline = previous.get("logistic_baseline", {})
    return {
        "available": bool(model or baseline),
        "source": str(path),
        "previous_model": {key: model.get(key) for key in (
            "roc_auc", "brier_score", "expected_calibration_error",
            "observed_kept_rate", "predicted_kept_rate",
        )},
        "previous_logistic_baseline": {key: baseline.get(key) for key in (
            "roc_auc", "brier_score", "expected_calibration_error",
            "observed_kept_rate", "predicted_kept_rate",
        )},
        "note": "Prior artifact comparison only; not a controlled comparison unless cohort and split match.",
    }


def _fairness_audit(
    predictions: pd.DataFrame,
    override_log_path: Path | None = None,
) -> dict[str, object]:
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
    override_ids: set[str] = set()
    if override_log_path is not None and override_log_path.is_file():
        with override_log_path.open(encoding="utf-8") as source:
            for line in source:
                event = json.loads(line)
                if event.get("golden_id") and event.get("suggested_action") != event.get("final_action"):
                    override_ids.add(str(event["golden_id"]))
    predictions = predictions.copy()
    predictions["overridden"] = predictions.golden_id.astype(str).isin(override_ids)
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
                "recommendation_rate": (
                    rows.recommended_action.value_counts(normalize=True).to_dict()
                    if "recommended_action" in rows else {}
                ),
                "override_rate": float(rows.get("overridden", pd.Series(False, index=rows.index)).fillna(False).astype(bool).mean()),
                "brier_score": float(((rows.predicted_kept - positive) ** 2).mean()),
                "expected_calibration_error": _metrics(positive, rows.predicted_kept.to_numpy())["expected_calibration_error"],
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
    """Run chronological model comparison, validation-only calibration, and holdout audit."""
    import numpy as np
    from lightgbm import LGBMClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    output_dir.mkdir(parents=True, exist_ok=True)
    previous_metrics = _existing_metrics_comparison(output_dir / "layer4_metrics.json")
    if len(set(as_of_dates)) < 5:
        raise ValueError("At least five distinct usable as-of dates are required for temporal validation")
    panel = action_panel(as_of_dates)
    train, validation, test, split = _temporal_split(panel, train_through)
    if train.label_kept_30d.nunique() < 2 or validation.label_kept_30d.nunique() < 2 or test.label_kept_30d.nunique() < 2:
        raise ValueError("Training, validation and holdout must each contain both promise outcomes")

    x_train = _model_matrix(train)
    x_validation = _model_matrix(validation, list(x_train.columns))
    x_test = _model_matrix(test, list(x_train.columns))
    y_train = train.label_kept_30d.astype(int)
    y_validation = validation.label_kept_30d.astype(int)
    y_test = test.label_kept_30d.astype(int)
    train_weights = _recency_weights(train.as_of)

    baseline = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    )
    baseline.fit(x_train, y_train, logisticregression__sample_weight=train_weights)
    baseline_validation_probability = baseline.predict_proba(x_validation)[:, 1]
    baseline_probability = baseline.predict_proba(x_test)[:, 1]

    candidates = [
        {"n_estimators": 120, "learning_rate": 0.04, "num_leaves": 15},
        {"n_estimators": 250, "learning_rate": 0.04, "num_leaves": 31},
    ]
    tuned = []
    for params in candidates:
        candidate = LGBMClassifier(**params, class_weight="balanced", random_state=42, verbosity=-1)
        candidate.fit(x_train, y_train, sample_weight=train_weights)
        valid_probability = candidate.predict_proba(x_validation)[:, 1]
        tuned.append((float(_metrics(y_validation, valid_probability)["brier_score"]), params, candidate, valid_probability))
    _, selected_params, model, validation_probability = min(tuned, key=lambda item: item[0])
    probability = model.predict_proba(x_test)[:, 1]
    calibration_method, calibrator, validation_calibrated, calibration_selection = _fit_calibrator(
        validation_probability, y_validation
    )
    calibrated_probability = _apply_calibrator(calibration_method, calibrator, probability)
    scored = test[["golden_id", "as_of", "label_kept_30d"]].copy()
    scored["predicted_kept"] = calibrated_probability
    scored["predicted_kept_uncalibrated"] = probability
    scored["baseline_predicted_kept"] = baseline_probability
    contribution_values = model.booster_.predict(x_test, pred_contrib=True)
    feature_names = list(x_test.columns)
    explanations = []
    for index, values in enumerate(contribution_values):
        top = np.argsort(np.abs(values[:-1]))[::-1][:5]
        contributions = [{"feature": feature_names[position], "contribution": float(values[position])} for position in top]
        details = _recommendation_details(test.iloc[index], float(calibrated_probability[index]), contributions)
        explanations.append(details)
    scored["recommended_action"] = [item["action"] for item in explanations]
    scored["action_rationale"] = [
        "; ".join(item["reason_codes"][0]["reason"] for _ in [0]) for item in explanations
    ]
    scored["confidence"] = [item["confidence"] for item in explanations]
    scored["reason_codes"] = [json.dumps(item["reason_codes"]) for item in explanations]
    scored["next_best_option"] = [item["next_best_option"] for item in explanations]
    scored["explanation_type"] = [item["explanation_type"] for item in explanations]
    scored["action_channel"] = test.action_channel.to_numpy()
    scored["action_outcome"] = test.action_outcome.to_numpy()
    scored.to_csv(output_dir / "layer4_scored_actions.csv", index=False)

    explanation_rows = []
    for row_index, values in enumerate(contribution_values[: min(200, len(contribution_values))]):
        top = np.argsort(np.abs(values[:-1]))[::-1][:5]
        explanation_rows.append({
            "golden_id": scored.iloc[row_index].golden_id,
            "as_of": str(scored.iloc[row_index].as_of),
            "baseline_logit": float(values[-1]),
            "top_contributions": json.dumps([
                {"feature": feature_names[index], "contribution": float(values[index])}
                for index in top
            ]),
            "reason_codes": scored.iloc[row_index].reason_codes,
        })
    pd.DataFrame(explanation_rows).to_csv(output_dir / "layer4_explanations.csv", index=False)
    model.booster_.save_model(str(output_dir / "layer4_lightgbm.txt"))

    calibration_eval_idx = calibration_selection.get("validation_selection_indices", [])
    val_calibrated_metrics = _metrics(
        y_validation.reset_index(drop=True).iloc[calibration_eval_idx],
        validation_calibrated[calibration_eval_idx],
    ) if calibration_eval_idx else _metrics(y_validation.reset_index(drop=True), validation_calibrated)
    calibration_selection.pop("validation_selection_indices", None)
    holdout_before = _metrics(y_test.reset_index(drop=True), probability)
    holdout_after = _metrics(y_test.reset_index(drop=True), calibrated_probability)
    development_panel = pd.concat([train, validation], ignore_index=True)
    rolling = _rolling_origin_validation(development_panel)
    top_decile = scored.predicted_kept >= scored.predicted_kept.quantile(0.9)
    metrics = {
        "model": holdout_after,
        "model_uncalibrated": holdout_before,
        "logistic_baseline": _metrics(y_test, baseline_probability),
        "validation": {
            "logistic_baseline": _metrics(y_validation.reset_index(drop=True), baseline_validation_probability),
            "selected_lightgbm": _metrics(y_validation.reset_index(drop=True), validation_probability),
            "selected_calibrated": val_calibrated_metrics,
            "lightgbm_search": [
                {"parameters": params, "validation_brier": brier}
                for brier, params, _, _ in tuned
            ],
            "calibration": {"method": calibration_method, **calibration_selection},
        },
        "rolling_origin_validation": rolling,
        "recency_weighting": {"formula": "0.5 ** (age_days / 90), normalized to mean 1", "half_life_days": 90},
        "temporal_split": split,
        "holdout_rows": len(test),
        "train_rows": len(train),
        "train_as_of_max": str(train.as_of.max().date()),
        "holdout_as_of_min": str(test.as_of.min().date()),
        "top_decile_observed_kept_rate": float(y_test[top_decile].mean()),
        "overall_observed_kept_rate": float(y_test.mean()),
        "top_decile_is_ranking_yield_not_causal_uplift": True,
        "observational_action_audit": _observational_action_audit(scored),
        "existing_metrics_comparison": previous_metrics,
        "fairness_audit": _fairness_audit(scored, ROOT / "out" / "audit_log.jsonl"),
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
        f"- Validation rows: {len(validation):,}; validation as-of: {split['validation_as_of']}.",
        f"- Holdout rows: {len(test):,}; earliest holdout as-of: {metrics['holdout_as_of_min']}.",
        f"- Recency weight: 0.5 ** (age_days / 90), normalized to mean 1.",
        f"- Selected calibration (validation only): {calibration_method}.",
        f"- LightGBM ROC AUC: {metrics['model']['roc_auc']}; Brier: {metrics['model']['brier_score']:.4f}.",
        f"- Logistic baseline ROC AUC: {metrics['logistic_baseline']['roc_auc']}; Brier: {metrics['logistic_baseline']['brier_score']:.4f}.",
        f"- Top-decile observed kept rate: {metrics['top_decile_observed_kept_rate']:.3f}; this is ranking yield, not causal uplift or policy value.",
        "- The training target is conditional on observed collections/contact activity and cannot estimate counterfactual action effects.",
        "- Score-bucket/contact summaries are OBSERVATIONAL / NON-RANDOMIZED and do not establish causal effects.",
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
