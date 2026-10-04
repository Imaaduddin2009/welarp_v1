"""Offline Streamlit demonstration UI for guarded collections decision support."""
from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
default_l2 = ROOT / "out" / "l2_maple.duckdb"
os.environ.setdefault(
    "C360_L2_DB",
    str(default_l2 if default_l2.is_file() else ROOT / "out" / "l2.duckdb"),
)
os.environ.setdefault("C360_AUDIT_LOG", str(ROOT / "out" / "audit_log.jsonl"))
import layer2
import layer4
from ui_components import reset_role_state, show_qa_response, show_table

LOGGER = logging.getLogger(__name__)
L2DB = Path(layer2.L2DB)
FEATURE_DB = Path(os.getenv("C360_FEATURE_DB", ROOT / "out" / "feature_store.duckdb"))
C360DB = Path(os.getenv("C360_C360_DB", ROOT / "out" / "c360_maple.duckdb"))
DECISION_AUDIT = ROOT / "out" / "decision_audit.jsonl"
SNAPSHOT_DATE = "2026-09-28"
ROLE_OPTIONS = ["Agent", "Team lead", "Analyst", "Data steward", "Supervisor"]
ROLE_KEYS = {
    "Agent": "agent",
    "Team lead": "lead",
    "Analyst": "analyst",
    "Data steward": "steward",
    "Supervisor": "supervisor",
}
ROLE_CAPABILITIES = {
    "Agent": (
        "Sees only the selected agent portfolio and masked collection context.",
        "Other portfolios, protected attributes, and direct contact details are blocked.",
    ),
    "Team lead": (
        "Sees group-level collection summaries with small groups suppressed.",
        "Customer, account, and record-level views are blocked.",
    ),
    "Analyst": (
        "Sees approved masked collection data and aggregate fairness audit results.",
        "Protected attributes and direct personal details are blocked.",
    ),
    "Data steward": (
        "Reviews pseudonymous identity-match evidence and records a review decision.",
        "Source identities and automatic crosswalk changes are blocked.",
    ),
    "Supervisor": (
        "Sees masked case context, pending escalations, and audit-only summaries.",
        "Direct personal details and autonomous contact or payment actions are blocked.",
    ),
}

st.set_page_config(page_title="Collections C360", layout="wide")


@st.cache_data(ttl=60, show_spinner=False)
def query_l2(sql: str, params: tuple, role_name: str, agent_id: str) -> pd.DataFrame:
    """Guard and execute a read-only Layer 2 query for the active role."""
    safe_sql = layer2.guard(sql, {"role": role_name, "agent_id": agent_id})
    with duckdb.connect(str(L2DB), read_only=True) as con:
        con.execute("SET enable_external_access=false")
        return con.execute(safe_sql, params).fetchdf()


@st.cache_data(ttl=60, show_spinner=False)
def query_features(
    golden_ids: tuple[str, ...], role_name: str, agent_id: str
) -> pd.DataFrame:
    """Read live features only for customer IDs already authorized by Layer 2."""
    if not golden_ids or not FEATURE_DB.is_file():
        return pd.DataFrame()
    if role_name not in {"agent", "analyst", "supervisor"}:
        raise PermissionError("This role cannot access customer-level feature data.")
    if role_name == "agent":
        placeholders = ", ".join("?" for _ in golden_ids)
        allowed = query_l2(
            f"""
            SELECT DISTINCT golden_id
            FROM cases
            WHERE golden_id IN ({placeholders})
            """,
            golden_ids,
            role_name,
            agent_id,
        )
        golden_ids = tuple(allowed["golden_id"].dropna().astype(str).unique())
        if not golden_ids:
            return pd.DataFrame()
    placeholders = ", ".join("?" for _ in golden_ids)
    columns = """
        golden_id, max_days_past_due, total_overdue_amount,
        promises_broken_90d, promise_kept_rate_90d, hardship_flag,
        hardship_review_needed, best_channel, consent_phone, consent_sms,
        consent_email, case_status, case_stage, as_of
    """
    with duckdb.connect(str(FEATURE_DB), read_only=True) as con:
        return con.execute(
            f"SELECT {columns} FROM features_live WHERE golden_id IN ({placeholders})",
            golden_ids,
        ).fetchdf()


@st.cache_data(ttl=60, show_spinner=False)
def query_outbound_calls(golden_ids: tuple[str, ...], role_name: str, agent_id: str) -> pd.DataFrame:
    """Count recent calls only for the authorized IDs supplied by the page."""
    if not golden_ids:
        return pd.DataFrame(columns=["golden_id", "outbound_calls_7d"])
    placeholders = ", ".join("?" for _ in golden_ids)
    sql = f"""
        SELECT golden_id,
               count(*) FILTER (
                   WHERE direction='outbound' AND channel IN ('call', 'phone')
                     AND contact_date >= DATE '{SNAPSHOT_DATE}' - INTERVAL 7 DAY
               ) AS outbound_calls_7d
        FROM contacts
        WHERE golden_id IN ({placeholders})
        GROUP BY golden_id
    """
    return query_l2(sql, golden_ids, role_name, agent_id)


@st.cache_data(ttl=60, show_spinner=False)
def query_scored_actions(golden_ids: tuple[str, ...]) -> pd.DataFrame:
    path = ROOT / "out" / "layer4_scored_actions.csv"
    if not path.is_file() or not golden_ids:
        return pd.DataFrame()
    scored = pd.read_csv(path, usecols=lambda name: name in {
        "golden_id", "predicted_kept", "recommended_action", "action_rationale",
    })
    return scored.loc[scored["golden_id"].isin(golden_ids)]


@st.cache_data(ttl=60, show_spinner=False)
def query_pipeline_refresh() -> str:
    """Read pipeline metadata separately; pipeline_run is intentionally not an NLQ table."""
    with duckdb.connect(str(L2DB), read_only=True) as con:
        row = con.execute(
            "SELECT run_ts::VARCHAR FROM pipeline_run LIMIT 1"
        ).fetchone()
    return str(row[0])[:19] if row else "unavailable"


@st.cache_resource
def qa_client(role_name: str, identity: str, portfolio: str) -> layer2.QA:
    """Create an offline-only Layer 2 QA client; never invoke the optional LLM."""
    client = layer2.QA({"id": identity, "role": role_name, "agent_id": portfolio})
    client.use_llm = False
    return client


def _role_changed() -> None:
    reset_role_state(st.session_state)
    st.session_state.pop("agent_portfolio", None)
    query_l2.clear()
    query_features.clear()
    query_outbound_calls.clear()
    query_scored_actions.clear()
    qa_client.clear()


def _portfolio_changed() -> None:
    reset_role_state(st.session_state)
    query_l2.clear()
    query_features.clear()
    query_outbound_calls.clear()
    query_scored_actions.clear()
    qa_client.clear()


def _safe_l2_query(
    sql: str,
    params: tuple,
    role_name: str,
    agent_id: str,
    *,
    purpose: str,
) -> pd.DataFrame:
    try:
        with st.spinner(purpose):
            return query_l2(sql, params, role_name, agent_id)
    except Exception:
        LOGGER.exception("Layer 2 query failed in app: %s", purpose)
        st.error("This view could not be loaded. Please retry or contact the demo operator.")
        return pd.DataFrame()


def _load_kpis(role_name: str, agent_id: str) -> pd.DataFrame:
    sql = f"""
        SELECT
            (
                SELECT count(DISTINCT golden_id)
                FROM cases
                WHERE status IN ('open', 'on_hold', 'disputed')
            ) AS customers_in_queue,
            (
                SELECT count(DISTINCT golden_id)
                FROM cases
                WHERE status IN ('open', 'on_hold', 'disputed')
                  AND hardship_flag
            ) AS hardship_alerts,
            (
                SELECT count(DISTINCT p.ptp_id)
                FROM promises p
                WHERE p.status='BROKEN'
                  AND p.promised_date >= DATE '{SNAPSHOT_DATE}' - INTERVAL 30 DAY
                  AND p.golden_id IN (
                      SELECT golden_id FROM cases
                      WHERE status IN ('open', 'on_hold', 'disputed')
                  )
            ) AS broken_promises_30d,
            (
                SELECT round(
                    100.0 * count(*) FILTER (WHERE p.status='KEPT')
                    / nullif(count(*) FILTER (WHERE p.status IN ('KEPT', 'BROKEN')), 0), 1
                )
                FROM promises p
                WHERE p.golden_id IN (
                    SELECT golden_id FROM cases
                    WHERE status IN ('open', 'on_hold', 'disputed')
                )
            ) AS promise_kept_rate_pct,
            (
                SELECT round(avg(days_past_due), 1)
                FROM accounts a
                WHERE a.snapshot_date = (
                    SELECT max(snapshot_date) FROM accounts
                    WHERE snapshot_date <= DATE '{SNAPSHOT_DATE}'
                )
                  AND a.golden_id IN (
                      SELECT golden_id FROM cases
                      WHERE status IN ('open', 'on_hold', 'disputed')
                  )
            ) AS average_days_past_due
    """
    return _safe_l2_query(
        sql, (), role_name, agent_id, purpose="Loading guarded portfolio metrics…"
    )


def _format_short_id(value: object) -> str:
    text = str(value)
    return text if len(text) <= 15 else f"{text[:12]}…"


def _series_or_default(frame: pd.DataFrame, name: str, default: object) -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series(default, index=frame.index)


def _is_truthy(value: object) -> bool:
    return False if pd.isna(value) else bool(value)


def _load_queue(role_name: str, agent_id: str) -> pd.DataFrame:
    sql = """
        SELECT c.case_id, c.golden_id, c.product_type, c.stage, c.status, c.opened_date
        FROM cases c
        WHERE c.status IN ('open', 'on_hold', 'disputed')
        ORDER BY c.opened_date DESC
        LIMIT 100
    """
    params: tuple = ()
    if role_name == "agent":
        sql = sql.replace(
            "WHERE c.status IN ('open', 'on_hold', 'disputed')",
            "WHERE c.status IN ('open', 'on_hold', 'disputed') AND c.agent_id = ?",
        )
        params = (agent_id,)
    queue = _safe_l2_query(
        sql, params, role_name, agent_id, purpose="Loading the guarded queue…"
    )
    if queue.empty:
        return queue

    allowed_ids = tuple(queue["golden_id"].dropna().astype(str).unique())
    try:
        features = query_features(allowed_ids, role_name, agent_id)
        if not features.empty:
            queue = queue.merge(features, on="golden_id", how="left", validate="many_to_one")
        calls = query_outbound_calls(allowed_ids, role_name, agent_id)
        if not calls.empty:
            queue = queue.merge(calls, on="golden_id", how="left", validate="many_to_one")
    except Exception:
        LOGGER.exception("Queue feature enrichment failed")
        st.error("Some queue context could not be loaded. Retry the view or contact the demo operator.")

    queue["days_past_due"] = pd.to_numeric(
        _series_or_default(queue, "max_days_past_due", 0), errors="coerce"
    ).fillna(0)
    queue["total_overdue_amount"] = pd.to_numeric(
        _series_or_default(queue, "total_overdue_amount", 0), errors="coerce"
    ).fillna(0)
    queue["hardship_flag"] = _series_or_default(
        queue, "hardship_flag", False
    ).fillna(False).astype(bool)
    review_needed = _series_or_default(
        queue, "hardship_review_needed", False
    ).fillna(False).astype(bool)
    queue["hardship_alert"] = queue["hardship_flag"] | review_needed
    queue["outbound_calls_7d"] = pd.to_numeric(
        _series_or_default(queue, "outbound_calls_7d", 0), errors="coerce"
    ).fillna(0)

    recommendations: list[str] = []
    confidences: list[float | None] = []
    scored = query_scored_actions(allowed_ids)
    scored_by_id = (
        scored.drop_duplicates("golden_id").set_index("golden_id")
        if not scored.empty else pd.DataFrame()
    )
    for _, row in queue.iterrows():
        action, _ = layer4.recommend_action(row)
        recommendations.append(action)
        if not scored_by_id.empty and row["golden_id"] in scored_by_id.index:
            scored_row = scored_by_id.loc[row["golden_id"]]
            confidence = scored_row.get("predicted_kept")
            confidences.append(float(confidence) if pd.notna(confidence) else None)
        else:
            rate = row.get("promise_kept_rate_90d")
            confidences.append(float(rate) if pd.notna(rate) else None)
    queue["recommended_action"] = recommendations
    queue["confidence_pct"] = [
        round(confidence * 100, 1) if confidence is not None else None
        for confidence in confidences
    ]
    queue["hardship_badge"] = queue.apply(
        lambda row: (
            "🔴 HARDSHIP" if bool(row["hardship_flag"])
            else "🟠 REVIEW" if bool(row["hardship_alert"])
            else "🟢 SAFE"
        ),
        axis=1,
    )
    queue["golden_id_short"] = queue["golden_id"].map(_format_short_id)
    return queue


def _demo_examples(
    queue: pd.DataFrame, role_name: str, agent_id: str
) -> list[tuple[str, str]]:
    examples: list[tuple[str, str]] = []
    if queue.empty:
        return examples
    specs = [
        ("Hardship alert", queue["hardship_alert"]),
        (
            "Broken promise",
            pd.to_numeric(
                _series_or_default(queue, "promises_broken_90d", 0), errors="coerce"
            ).fillna(0) > 0,
        ),
        (
            "Clean account",
            (~queue["hardship_alert"])
            & (queue["days_past_due"] <= 0)
            & (queue["total_overdue_amount"] <= 0)
            & (pd.to_numeric(
                _series_or_default(queue, "promises_broken_90d", 0), errors="coerce"
            ).fillna(0) <= 0),
        ),
    ]
    for label, condition in specs:
        matched = queue.loc[condition]
        if not matched.empty:
            examples.append((label, str(matched.iloc[0]["golden_id"])))
            continue
        if label == "Hardship alert":
            sql = """
                SELECT golden_id FROM cases
                WHERE status IN ('open', 'on_hold', 'disputed') AND hardship_flag
                ORDER BY opened_date DESC LIMIT 1
            """
        elif label == "Broken promise":
            sql = f"""
                SELECT DISTINCT p.golden_id
                FROM promises p
                WHERE p.status='BROKEN'
                  AND p.promised_date >= DATE '{SNAPSHOT_DATE}' - INTERVAL 90 DAY
                  AND p.golden_id IN (
                      SELECT golden_id FROM cases
                      WHERE status IN ('open', 'on_hold', 'disputed')
                  )
                ORDER BY p.golden_id LIMIT 1
            """
        else:
            sql = f"""
                SELECT a.golden_id
                FROM accounts a
                WHERE a.snapshot_date = (
                    SELECT max(snapshot_date) FROM accounts
                    WHERE snapshot_date <= DATE '{SNAPSHOT_DATE}'
                )
                  AND a.days_past_due=0 AND a.overdue_amount=0
                  AND a.golden_id IN (
                      SELECT golden_id FROM customers WHERE NOT hardship_flag
                  )
                  AND a.golden_id IN (
                      SELECT golden_id FROM cases
                      WHERE status IN ('open', 'on_hold', 'disputed')
                  )
                ORDER BY a.golden_id LIMIT 1
            """
        found = _safe_l2_query(
            sql, (), role_name, agent_id, purpose=f"Finding a real {label.lower()} example…"
        )
        if not found.empty:
            examples.append((label, str(found.iloc[0]["golden_id"])))
    return examples


def _render_queue(role_name: str, agent_id: str) -> pd.DataFrame:
    if role_name == "lead":
        summary = _safe_l2_query(
            """
            SELECT stage, count(DISTINCT golden_id) AS customers
            FROM cases
            WHERE status IN ('open', 'on_hold', 'disputed')
            GROUP BY stage ORDER BY stage
            """,
            (), role_name, agent_id, purpose="Loading aggregate queue summary…",
        )
        if summary.empty:
            st.info("No eligible aggregate queue groups are available.")
        else:
            show_table(summary)
        return pd.DataFrame()

    queue = _load_queue(role_name, agent_id)
    if queue.empty:
        st.info("There are no cases in this role's current queue.")
        return queue

    left, right = st.columns([2, 3])
    examples = _demo_examples(queue, role_name, agent_id)
    with left:
        if examples:
            selected_label = st.selectbox(
                "Pick a demo customer",
                options=[label for label, _ in examples],
                key="selected_demo_customer",
            )
            selected_id = dict(examples)[selected_label]
            st.caption(f"Example selected: {_format_short_id(selected_id)}")
            if st.button("Open demo customer", key="open_demo_customer"):
                st.session_state["selected_customer_id"] = selected_id
        else:
            st.caption("No hardship, broken-promise, or clean example is present in this queue page.")
    with right:
        st.caption("Filters apply to the displayed, role-scoped queue.")

    filter_cols = st.columns(5)
    stages = sorted(queue["stage"].dropna().astype(str).unique())
    products = sorted(queue["product_type"].dropna().astype(str).unique())
    actions = sorted(queue["recommended_action"].dropna().astype(str).unique())
    with filter_cols[0]:
        hardship_only = st.checkbox("Hardship only", key="queue_hardship_only")
    with filter_cols[1]:
        stage_filter = st.multiselect("Case stage", stages, key="queue_stage")
    with filter_cols[2]:
        product_filter = st.multiselect("Product type", products, key="queue_product")
    with filter_cols[3]:
        max_days = max(1, int(queue["days_past_due"].max()))
        minimum_dpd = st.slider(
            "Minimum days past due", min_value=0, max_value=max_days,
            value=0, key="queue_minimum_dpd",
        )
    with filter_cols[4]:
        action_filter = st.multiselect("Recommended action", actions, key="queue_action")

    filtered = queue.copy()
    if hardship_only:
        filtered = filtered.loc[filtered["hardship_alert"]]
    if stage_filter:
        filtered = filtered.loc[filtered["stage"].astype(str).isin(stage_filter)]
    if product_filter:
        filtered = filtered.loc[filtered["product_type"].astype(str).isin(product_filter)]
    filtered = filtered.loc[filtered["days_past_due"] >= minimum_dpd]
    if action_filter:
        filtered = filtered.loc[filtered["recommended_action"].isin(action_filter)]
    display = filtered[
        [
            "golden_id_short", "days_past_due", "total_overdue_amount",
            "hardship_badge", "recommended_action", "confidence_pct",
            "stage", "product_type",
        ]
    ].rename(columns={
        "days_past_due": "days_past_due",
        "total_overdue_amount": "total_overdue",
        "hardship_badge": "hardship_badge",
        "recommended_action": "recommended_action",
        "confidence_pct": "confidence_pct",
    })
    event = show_table(
        display,
        column_config={
            "Golden Id Short": st.column_config.TextColumn("Customer"),
            "Days Past Due": st.column_config.NumberColumn("Days past due", format="%d"),
            "Total Overdue": st.column_config.NumberColumn("Total overdue", format="$%.2f"),
            "Hardship Badge": st.column_config.TextColumn("Hardship alert"),
            "Recommended Action": st.column_config.TextColumn("Recommended action"),
            "Confidence Pct": st.column_config.ProgressColumn(
                "Confidence", min_value=0, max_value=100, format="%.1f%%"
            ),
        },
        key="queue_table",
        on_select="rerun",
        selection_mode="single-row",
    )
    selected_rows = getattr(getattr(event, "selection", None), "rows", [])
    if selected_rows:
        row_index = int(selected_rows[0])
        if 0 <= row_index < len(filtered):
            st.session_state["selected_customer_id"] = str(
                filtered.iloc[row_index]["golden_id"]
            )
            st.info("Customer selected. Open the Customer 360 tab to review the record.")
    selected_id = st.session_state.get("selected_customer_id")
    if selected_id:
        st.caption(f"Selected customer: {_format_short_id(selected_id)}")
    return queue


def _customer_access(
    golden_id: str, role_name: str, agent_id: str
) -> bool:
    sql = "SELECT count(*) AS access_count FROM cases WHERE golden_id = ?"
    params: tuple = (golden_id,)
    if role_name == "agent":
        sql += " AND agent_id = ?"
        params = (golden_id, agent_id)
    result = _safe_l2_query(
        sql, params, role_name, agent_id, purpose="Verifying customer portfolio access…"
    )
    return not result.empty and int(result.iloc[0]["access_count"]) > 0


def _customer_rows(
    sql: str, golden_id: str, role_name: str, agent_id: str, purpose: str
) -> pd.DataFrame:
    return _safe_l2_query(
        sql, (golden_id,), role_name, agent_id, purpose=purpose
    )


def _shap_reasons(golden_id: str, rationale: str) -> list[str]:
    path = ROOT / "out" / "layer4_explanations.csv"
    if path.is_file():
        try:
            explanations = pd.read_csv(path, usecols=["golden_id", "top_contributions"])
            match = explanations.loc[explanations["golden_id"] == golden_id]
            if not match.empty:
                contributions = json.loads(match.iloc[0]["top_contributions"])
                labels = {
                    "max_days_past_due": "days past due",
                    "total_overdue_amount": "overdue balance",
                    "promises_broken_90d": "recent broken promises",
                    "promise_kept_rate_90d": "recent promise history",
                    "contact_attempts_14d": "recent contact history",
                    "days_since_last_promise": "time since a promise",
                    "consent_phone": "phone consent status",
                    "consent_sms": "SMS consent status",
                    "consent_email": "email consent status",
                    "case_stage": "case stage",
                    "case_status": "case status",
                    "hardship_flag": "recorded hardship status",
                    "hardship_review_needed": "masked hardship review signal",
                    "txt_job_loss": "masked job-loss cue",
                    "txt_medical": "masked medical cue",
                    "txt_struggling": "masked financial-strain cue",
                }
                reasons = []
                for item in contributions[:3]:
                    feature = str(item.get("feature", ""))
                    label = labels.get(feature)
                    contribution = float(item.get("contribution", 0))
                    if label:
                        effect = "supports" if contribution >= 0 else "lowers"
                        reasons.append(f"{label.capitalize()} {effect} the historical kept-outcome score.")
                if reasons:
                    return reasons
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            LOGGER.exception("Could not load explanation artifact for selected record")
    return [rationale]


def _safety_explanation(feature: dict[str, object], action: str) -> tuple[list[str], str]:
    rules: list[str] = []
    hardship = _is_truthy(feature.get("hardship_flag")) or _is_truthy(
        feature.get("hardship_review_needed")
    )
    status = str(feature.get("case_status", "")).lower()
    raw_attempts = feature.get("outbound_calls_7d", 0)
    attempts = 0 if pd.isna(raw_attempts) else int(raw_attempts)
    consents = {
        "phone": _is_truthy(feature.get("consent_phone")),
        "sms": _is_truthy(feature.get("consent_sms")),
        "email": _is_truthy(feature.get("consent_email")),
    }
    raw_channel = feature.get("best_channel", "")
    best_channel = "" if pd.isna(raw_channel) else str(raw_channel).lower()
    if best_channel == "call":
        best_channel = "phone"
    if hardship:
        rules.append("Hardship flag or masked hardship cue routes this case to human review.")
    if status in {"disputed", "on_hold", "closed", "cured"}:
        rules.append("Case status requires review before any contact.")
    if attempts >= 3:
        rules.append("Three or more outbound calls in seven days block another call.")
    if not any(consents.values()):
        rules.append("No recorded contact channel has consent; outbound contact is blocked.")
    why_not = ""
    if best_channel in consents and not consents[best_channel]:
        why_not = f"Why not {best_channel.upper()}? No recorded {best_channel.upper()} consent."
    if not rules:
        rules.append("No listed safety blocker is active; verify policy and contact history before acting.")
    allowed_alternatives = [
        channel for channel in ("phone", "sms", "email")
        if consents[channel] and channel != best_channel
    ]
    if hardship or status in {"disputed", "on_hold", "closed", "cured"} or attempts >= 3:
        alternative = "Human review; do not initiate contact."
    elif allowed_alternatives:
        alternative = f"Agent review using consented {allowed_alternatives[0].upper()} channel."
    elif not any(consents.values()):
        alternative = "No consented channel; retain for human review."
    else:
        alternative = "Agent review after verifying policy and contact limits."
    if action == "NO_CONSENTING_CHANNEL":
        why_not = "Why not contact? No recorded consent for any channel."
    return rules, f"Next-best alternative: {alternative}" + (f"\n{why_not}" if why_not else "")


def _append_decision_event(event: dict[str, object]) -> None:
    DECISION_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with DECISION_AUDIT.open("a", encoding="utf-8") as audit:
        audit.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
    _read_audit_events.clear()


def _render_customer(golden_id: str, role_name: str, agent_id: str) -> None:
    if role_name == "lead":
        st.error("Team leads receive aggregate-only views.")
        return
    if not _customer_access(golden_id, role_name, agent_id):
        st.error("This customer is not available in the active role's portfolio.")
        return

    profile_frame = query_features((golden_id,), role_name, agent_id)
    profile = profile_frame.iloc[0].to_dict() if not profile_frame.empty else {}
    hardship_flag = _is_truthy(profile.get("hardship_flag", False))
    hardship_review = _is_truthy(profile.get("hardship_review_needed", False))
    with st.container(border=True):
        st.subheader(f"Customer {_format_short_id(golden_id)}")
        if hardship_flag:
            st.error("Hardship flag: human review required")
        elif hardship_review:
            st.warning("Possible hardship cue in masked notes: human review required")
        else:
            st.success("No hardship alert is present in the current feature snapshot.")

    accounts = _customer_rows(
        """
        SELECT product_type, balance, overdue_amount, days_past_due, snapshot_date
        FROM accounts
        WHERE golden_id = ?
        QUALIFY row_number() OVER (
            PARTITION BY account_id ORDER BY snapshot_date DESC
        ) = 1
        ORDER BY product_type
        """,
        golden_id, role_name, agent_id, "Loading masked account context…",
    )
    cases = _customer_rows(
        """
        SELECT product_type, stage, status, opened_date, status_date, hardship_flag
        FROM cases WHERE golden_id=? ORDER BY opened_date DESC LIMIT 100
        """,
        golden_id, role_name, agent_id, "Loading current case details…",
    )
    promises = _customer_rows(
        """
        SELECT promised_date, status_date, promised_amount, status
        FROM promises WHERE golden_id=? ORDER BY promised_date DESC LIMIT 100
        """,
        golden_id, role_name, agent_id, "Loading promise history…",
    )
    contacts = _customer_rows(
        """
        SELECT c.contact_date, c.channel, c.outcome, c.note_id,
               n.note_text_masked
        FROM contacts c
        LEFT JOIN notes n USING (note_id)
        WHERE c.golden_id=?
        ORDER BY c.contact_date DESC
        LIMIT 50
        """,
        golden_id, role_name, agent_id, "Loading masked contact history…",
    )
    consents = _customer_rows(
        """
        SELECT consent_phone, consent_sms, consent_email, consent_as_of_date
        FROM customers WHERE golden_id=?
        """,
        golden_id, role_name, agent_id, "Loading channel consent…",
    )

    detail_left, detail_right = st.columns(2)
    with detail_left:
        st.subheader("Accounts")
        if accounts.empty:
            st.info("No account snapshots are available.")
        else:
            show_table(accounts, column_config={
                "Balance": st.column_config.NumberColumn(format="$%.2f"),
                "Overdue Amount": st.column_config.NumberColumn("Overdue", format="$%.2f"),
                "Days Past Due": st.column_config.NumberColumn(format="%d"),
            })
    with detail_right:
        st.subheader("Cases")
        if cases.empty:
            st.info("No case rows are available.")
        else:
            show_table(cases)

    promise_col, contact_col = st.columns(2)
    with promise_col:
        st.subheader("Promises")
        if promises.empty:
            st.info("No promise history is available.")
        else:
            for _, promise in promises.head(12).iterrows():
                amount = promise["promised_amount"]
                amount = 0.0 if pd.isna(amount) else float(amount)
                label = f"{promise['promised_date']} · {promise['status']} · ${amount:,.2f}"
                if str(promise["status"]).upper() == "KEPT":
                    st.success(label)
                elif str(promise["status"]).upper() == "BROKEN":
                    st.error(label)
                else:
                    st.warning(label)
    with contact_col:
        st.subheader("Contacts")
        if contacts.empty:
            st.info("No contact history is available.")
        else:
            show_table(contacts.drop(columns=["note_text_masked"], errors="ignore"))
    st.subheader("Notes (masked text and source IDs)")
    if contacts.empty or "note_text_masked" not in contacts:
        st.info("No masked notes are available.")
    else:
        notes = contacts.loc[
            contacts["note_id"].notna() & contacts["note_text_masked"].notna(),
            ["note_id", "note_text_masked"],
        ].drop_duplicates()
        if notes.empty:
            st.info("No masked notes are available.")
        else:
            show_table(notes.rename(columns={
                "note_id": "source_note_id", "note_text_masked": "masked_text",
            }))

    st.subheader("Consent by channel")
    if consents.empty:
        st.info("Consent snapshot is unavailable.")
    else:
        consent_row = consents.iloc[0]
        consent_cols = st.columns(3)
        for column, channel, field in zip(
            consent_cols, ("Phone", "SMS", "Email"),
            ("consent_phone", "consent_sms", "consent_email"),
        ):
            column.metric(
                channel,
                "Granted" if _is_truthy(consent_row[field]) else "Not recorded",
            )
        st.caption(f"Consent as of {consent_row['consent_as_of_date']}")

    freshness = query_pipeline_refresh()
    st.subheader("Data lineage")
    show_table(pd.DataFrame([
        {"Source table": table, "Freshness": freshness}
        for table in ("cases", "accounts", "promises", "contacts", "notes", "customers")
    ]))
    st.caption(f"Feature snapshot as of {profile.get('as_of', 'unavailable')}")

    call_rows = query_outbound_calls((golden_id,), role_name, agent_id)
    profile["outbound_calls_7d"] = (
        int(call_rows.iloc[0]["outbound_calls_7d"])
        if not call_rows.empty else 0
    )
    action, rationale = layer4.recommend_action(pd.Series(profile))
    scored = query_scored_actions((golden_id,))
    score = None
    if not scored.empty:
        match = scored.loc[scored["golden_id"] == golden_id]
        if not match.empty and pd.notna(match.iloc[0].get("predicted_kept")):
            score = float(match.iloc[0]["predicted_kept"])
    safety_rules, alternative = _safety_explanation(profile, action)
    with st.container(border=True):
        st.subheader("Recommendation")
        st.markdown(f"**Recommended action:** `{action}`")
        if score is not None:
            st.caption("Historical promise-kept model score; not causal uplift or certainty.")
            st.progress(max(0.0, min(1.0, score)), text=f"Model score: {score:.1%}")
        else:
            st.caption("Model confidence is unavailable for this customer.")
        st.markdown("**Why this recommendation**")
        for reason in _shap_reasons(golden_id, rationale):
            st.markdown(f"- {reason}")
        st.markdown("**Safety rules applied**")
        for rule in safety_rules:
            st.markdown(f"- {rule}")
        st.markdown(alternative.replace("\n", "  \n"))

    st.subheader("Human review decision")
    reason_category = st.selectbox(
        "Required reason", ["Select a reason", "Policy verified", "Consent verified",
                            "Hardship reviewed", "Supervisor direction", "Other"],
        key=f"decision_reason_{golden_id}",
    )
    reason_text = st.text_input(
        "Reason details", key=f"decision_reason_text_{golden_id}"
    )
    actions = sorted(set(scored["recommended_action"].dropna().astype(str))) if not scored.empty else []
    if action not in actions:
        actions.append(action)
    override_action = st.selectbox(
        "Override recommendation", actions, key=f"override_action_{golden_id}"
    )
    can_record = reason_category != "Select a reason" and bool(reason_text.strip())
    accept_col, override_col, escalate_col = st.columns(3)
    if accept_col.button(
        "Accept recommendation", key=f"accept_{golden_id}", disabled=not can_record
    ):
        try:
            layer4.record_override(
                DECISION_AUDIT, golden_id, action, action, role_name,
                f"{reason_category}: {reason_text.strip()}",
            )
            _read_audit_events.clear()
            st.toast("Decision recorded in the audit log.")
        except Exception:
            LOGGER.exception("Could not record accepted recommendation")
            st.error("The decision could not be recorded. Please retry.")
    if override_col.button(
        "Record override", key=f"override_{golden_id}", disabled=not can_record
    ):
        try:
            layer4.record_override(
                DECISION_AUDIT, golden_id, action, override_action, role_name,
                f"{reason_category}: {reason_text.strip()}",
            )
            _read_audit_events.clear()
            st.toast("Override recorded in the audit log.")
        except Exception:
            LOGGER.exception("Could not record recommendation override")
            st.error("The override could not be recorded. Please retry.")
    if escalate_col.button(
        "Escalate for supervisor approval",
        key=f"escalate_{golden_id}",
        disabled=not can_record,
    ):
        try:
            _append_decision_event({
                "event_type": "supervisor_escalation",
                "escalation_id": str(uuid.uuid4()),
                "timestamp": datetime.now().timestamp(),
                "golden_id": golden_id,
                "suggested_action": action,
                "requester": role_name,
                "reason": f"{reason_category}: {reason_text.strip()}",
                "status": "pending",
            })
            st.toast("Escalation sent to Supervisor approvals; no action was taken.")
        except OSError:
            LOGGER.exception("Could not create supervisor escalation")
            st.error("The escalation could not be recorded. Please retry.")


@st.cache_data(ttl=30, show_spinner=False)
def _read_audit_events() -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for path in (ROOT / "out" / "audit_log.jsonl", DECISION_AUDIT):
        if not path.is_file():
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            LOGGER.exception("Could not read audit file %s", path.name)
            continue
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                LOGGER.warning("Skipping malformed audit event in %s at line %d", path.name, line_number)
                continue
            if path.name == "audit_log.jsonl":
                user = raw.get("user") or {}
                timestamp = raw.get("ts")
                event_type = "QA refusal" if raw.get("refused") else "QA answer"
                try:
                    parsed_time = datetime.fromtimestamp(float(timestamp)) if timestamp else None
                except (ValueError, TypeError, OSError):
                    parsed_time = None
                events.append({
                    "timestamp": parsed_time,
                    "event_type": event_type,
                    "user_role": user.get("role", "unknown"),
                    "actor": user.get("id", "unknown"),
                    "route": raw.get("route", ""),
                    "decision": "",
                    "status": "",
                })
            else:
                timestamp = raw.get("timestamp", raw.get("ts"))
                try:
                    parsed_time = datetime.fromtimestamp(float(timestamp)) if timestamp else None
                except (ValueError, TypeError, OSError):
                    parsed_time = None
                suggested = raw.get("suggested_action", "")
                final = raw.get("final_action", "")
                inferred_type = (
                    "accept" if suggested and suggested == final
                    else "override" if suggested or final
                    else "decision"
                )
                events.append({
                    "timestamp": parsed_time,
                    "event_type": raw.get("event_type", inferred_type),
                    "user_role": raw.get("reviewer", raw.get("requester", "unknown")),
                    "actor": raw.get("reviewer", raw.get("requester", "unknown")),
                    "route": "",
                    "decision": raw.get("decision", final),
                    "status": raw.get("status", ""),
                    "escalation_id": raw.get("escalation_id", ""),
                    "golden_id": raw.get("golden_id", ""),
                    "suggested_action": suggested,
                    "reason": raw.get("reason", ""),
                })
    return events


def _render_steward_review() -> None:
    if not C360DB.is_file():
        st.info("The local identity-review database is unavailable.")
        return
    try:
        with duckdb.connect(str(C360DB), read_only=True) as con:
            con.execute("SET enable_external_access=false")
            candidates = con.execute("""
                SELECT source_system, source_id, candidate_golden_id,
                       match_confidence, review_status
                FROM steward_queue
                WHERE review_status='pending'
                ORDER BY match_confidence DESC
                LIMIT 100
            """).fetchdf()
    except Exception:
        LOGGER.exception("Could not load identity review queue")
        st.error("The identity review queue could not be loaded.")
        return
    if candidates.empty:
        st.success("No uncertain identity matches are waiting for review.")
        return
    st.caption("Match score is not a calibrated probability. Review decisions never merge records automatically.")
    labels = [
        f"{row.source_system} · {_format_short_id(row.source_id)} · "
        f"score {float(row.match_confidence):.3f}"
        for row in candidates.itertuples()
    ]
    selected = st.selectbox("Uncertain match", range(len(candidates)), format_func=lambda i: labels[i])
    candidate = candidates.iloc[int(selected)]
    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("**Source record (masked)**")
            st.write(f"System: {candidate.source_system}")
            st.write(f"Source ID: {_format_short_id(candidate.source_id)}")
    with right:
        with st.container(border=True):
            st.markdown("**Candidate record (masked)**")
            st.write(f"Candidate ID: {_format_short_id(candidate.candidate_golden_id)}")
            st.write(f"Score: {float(candidate.match_confidence):.3f}")
    st.caption("Rule used: deterministic composite identity match; no source identity fields are shown.")
    review_reason = st.text_input("Required review rationale", key="steward_reason")
    approve_col, separate_col = st.columns(2)
    for column, label, decision in (
        (approve_col, "Approve merge", "approve"),
        (separate_col, "Keep separate", "keep_separate"),
    ):
        if column.button(label, key=f"steward_{decision}", disabled=not review_reason.strip()):
            try:
                _append_decision_event({
                    "event_type": "steward_review",
                    "timestamp": datetime.now().timestamp(),
                    "source_system": candidate.source_system,
                    "source_id": candidate.source_id,
                    "candidate_golden_id": candidate.candidate_golden_id,
                    "decision": decision,
                    "reviewer": st.session_state["role_label"],
                    "reason": review_reason.strip(),
                    "review_status": "recorded_not_merged",
                })
                st.toast("Steward review recorded; no identity crosswalk was changed.")
            except OSError:
                LOGGER.exception("Could not record identity review")
                st.error("The review could not be recorded. Please retry.")


def _pending_escalations(events: list[dict[str, object]]) -> list[dict[str, object]]:
    resolutions = {
        str(event.get("escalation_id"))
        for event in events
        if event.get("event_type") == "supervisor_decision"
        and event.get("escalation_id")
    }
    return [
        event for event in events
        if event.get("event_type") == "supervisor_escalation"
        and event.get("status") == "pending"
        and str(event.get("escalation_id")) not in resolutions
    ]


def _render_supervisor_approvals(role_name: str) -> None:
    events = _read_audit_events()
    pending = _pending_escalations(events)
    if not pending:
        st.info("No pending supervisor escalations.")
        return
    for index, event in enumerate(pending):
        with st.container(border=True):
            st.markdown(f"**Escalation {index + 1}**")
            st.write(f"Customer: {_format_short_id(event.get('golden_id', ''))}")
            st.write(f"Requested route: {event.get('suggested_action', 'Review')}")
            st.write(f"Requested by: {event.get('user_role', 'unknown')}")
            reason_category = str(event.get("reason", "")).split(":", 1)[0]
            st.write(f"Reason category: {reason_category or 'Not supplied'}")
            reason = st.text_input(
                "Supervisor rationale",
                key=f"supervisor_reason_{event.get('escalation_id')}",
            )
            approve_col, reject_col = st.columns(2)
            for column, label, decision in (
                (approve_col, "Approve", "approved"),
                (reject_col, "Reject", "rejected"),
            ):
                if column.button(
                    label, key=f"supervisor_{decision}_{event.get('escalation_id')}",
                    disabled=not reason.strip(),
                ):
                    try:
                        golden_id = str(event.get("golden_id", ""))
                        suggested = str(event.get("suggested_action", "HUMAN_REVIEW"))
                        final_action = suggested if decision == "approved" else "REJECTED"
                        layer4.record_override(
                            DECISION_AUDIT, golden_id, suggested, final_action,
                            role_name, reason.strip(),
                        )
                        _append_decision_event({
                            "event_type": "supervisor_decision",
                            "timestamp": datetime.now().timestamp(),
                            "escalation_id": event.get("escalation_id"),
                            "decision": decision,
                            "reviewer": role_name,
                            "reason": reason.strip(),
                            "status": "resolved",
                        })
                        st.toast(f"Escalation {decision}.")
                    except Exception:
                        LOGGER.exception("Could not record supervisor decision")
                        st.error("The supervisor decision could not be recorded.")


def _render_audit_log() -> None:
    events = _read_audit_events()
    frame = pd.DataFrame(events)
    if frame.empty:
        st.info("No audit events have been recorded.")
        return
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    today = date.today()
    start_col, end_col, event_col, role_col = st.columns(4)
    start_date = start_col.date_input("From", today - timedelta(days=30), key="audit_start")
    end_date = end_col.date_input("To", today, key="audit_end")
    event_options = sorted(frame["event_type"].fillna("unknown").astype(str).unique())
    role_options = sorted(frame["user_role"].fillna("unknown").astype(str).unique())
    selected_events = event_col.multiselect("Event type", event_options, key="audit_events")
    selected_roles = role_col.multiselect("User role", role_options, key="audit_roles")
    if start_date > end_date:
        st.warning("The audit start date must be on or before the end date.")
        return
    filtered = frame.loc[
        frame["timestamp"].dt.date.between(start_date, end_date, inclusive="both")
    ]
    if selected_events:
        filtered = filtered.loc[filtered["event_type"].isin(selected_events)]
    if selected_roles:
        filtered = filtered.loc[filtered["user_role"].isin(selected_roles)]
    safe_columns = ["timestamp", "event_type", "user_role", "actor", "route", "decision", "status"]
    safe = filtered.reindex(columns=safe_columns).sort_values("timestamp", ascending=False)
    show_table(safe)
    st.download_button(
        "Download audit CSV",
        data=safe.to_csv(index=False),
        file_name="collections_audit_log.csv",
        mime="text/csv",
    )


def _render_fairness() -> None:
    metrics_path = ROOT / "out" / "layer4_metrics.json"
    st.warning("Fairness view: audit only, not used in decisions.")
    if not metrics_path.is_file():
        st.info("The fairness audit artifact is unavailable. Run the local Layer 4 pipeline first.")
        return
    try:
        audit = json.loads(metrics_path.read_text(encoding="utf-8")).get("fairness_audit", {})
    except (OSError, json.JSONDecodeError):
        LOGGER.exception("Could not read fairness audit output")
        st.error("The fairness audit could not be loaded.")
        return
    if not audit.get("available"):
        st.info(str(audit.get("reason", "No fairness audit output is available.")))
        return
    override_rates = _override_rates_by_group()
    for attribute, rows in audit.get("groups", {}).items():
        st.subheader(attribute.replace("_", " ").title())
        visible = [row for row in rows if not row.get("suppressed")]
        suppressed = [row for row in rows if row.get("suppressed")]
        if visible:
            attribute_overrides = override_rates.get(attribute, {})
            for row in visible:
                values = attribute_overrides.get(str(row.get("group")), {})
                row["override_rate"] = values.get("override_rate")
                row["override_decisions"] = values.get("decisions", 0)
            chart = pd.DataFrame([
                {
                    "Group": row["group"],
                    "Recommendation rate": 100 * float(row.get("predicted_positive_rate", 0)),
                    "Override rate": (
                        100 * float(row["override_rate"])
                        if row.get("override_rate") is not None else None
                    ),
                }
                for row in visible
            ]).set_index("Group")
            st.bar_chart(chart)
            show_table(pd.DataFrame(visible).rename(columns={
                "group": "group",
                "count": "audit_rows",
                "observed_kept_rate": "observed_kept_rate",
                "predicted_positive_rate": "recommendation_rate",
            }))
            if not any(row.get("override_rate") is not None for row in visible):
                st.caption("No group-level override rate is available yet; no audit decisions meet the minimum-size rule.")
        else:
            st.info("No groups meet the audit minimum size.")
        if suppressed:
            st.caption(f"{len(suppressed)} group(s) suppressed under the minimum-size rule.")


def _override_rates_by_group() -> dict[str, dict[str, dict[str, float | int]]]:
    decisions = [
        event for event in _read_audit_events()
        if event.get("event_type") in {"accept", "override"}
        and event.get("golden_id")
    ]
    if not decisions or not C360DB.is_file():
        return {}
    decision_frame = pd.DataFrame(decisions)
    decision_frame["overridden"] = (
        decision_frame["suggested_action"].astype(str)
        != decision_frame["decision"].astype(str)
    )
    ids = tuple(decision_frame["golden_id"].dropna().astype(str).unique())
    placeholders = ", ".join("?" for _ in ids)
    try:
        with duckdb.connect(str(C360DB), read_only=True) as con:
            con.execute("SET enable_external_access=false")
            restricted = con.execute(
                f"""
                SELECT golden_id, gender, newcomer, accessibility, vulnerability
                FROM c360_protected_restricted
                WHERE golden_id IN ({placeholders})
                """,
                ids,
            ).fetchdf()
    except Exception:
        LOGGER.exception("Could not compute audit-only group override rates")
        return {}
    if restricted.empty:
        return {}
    audited = decision_frame[["golden_id", "overridden"]].merge(
        restricted, on="golden_id", how="inner", validate="many_to_one"
    )
    result: dict[str, dict[str, dict[str, float | int]]] = {}
    minimum = layer4.MIN_FAIRNESS_GROUP
    for attribute in ("gender", "newcomer", "accessibility", "vulnerability"):
        attribute_result: dict[str, dict[str, float | int]] = {}
        for value, group in audited.groupby(attribute, dropna=False):
            name = "Unknown" if pd.isna(value) else str(value)
            count = len(group)
            if count >= minimum:
                attribute_result[name] = {
                    "decisions": count,
                    "override_rate": float(group["overridden"].mean()),
                }
        result[attribute] = attribute_result
    return result


if not L2DB.is_file():
    st.error("The masked Layer 2 database is unavailable. Run the local data pipeline first.")
    st.stop()

if "role_label" not in st.session_state:
    st.session_state["role_label"] = "Analyst"

st.title("Collections C360")
st.caption("Local, offline decision support for reviewing masked collections data.")
header_left, header_right = st.columns([5, 1])
with header_left:
    st.warning(
        "Decision support only — verify consent, policy, hardship, disputes, and contact limits before acting."
    )
with header_right:
    st.markdown(f"**Role badge**  \n`{st.session_state['role_label']}`")
st.caption(f"Data snapshot: {SNAPSHOT_DATE}")

selected_role = st.radio(
    "Role",
    ROLE_OPTIONS,
    horizontal=True,
    key="role_label",
    on_change=_role_changed,
    label_visibility="collapsed",
)
with st.container(border=True):
    st.markdown("**What this role can do**")
    st.caption(" · ".join(ROLE_CAPABILITIES[selected_role]))
role = ROLE_KEYS[selected_role]
agent_id = ""
if role == "agent":
    agents = _safe_l2_query(
        "SELECT DISTINCT agent_id FROM cases WHERE agent_id IS NOT NULL ORDER BY agent_id",
        (), "analyst", "", purpose="Loading agent portfolios…",
    )
    agent_options = agents["agent_id"].dropna().astype(str).tolist() if not agents.empty else []
    if not agent_options:
        st.error("No agent portfolios are available in the masked data.")
        st.stop()
    agent_id = st.selectbox(
        "Agent portfolio", agent_options, key="agent_portfolio", on_change=_portfolio_changed
    )
    st.caption(f"Viewing portfolio of {agent_id}")
user = {"id": agent_id or role, "role": role, "agent_id": agent_id}

try:
    st.caption(f"Pipeline refreshed: {query_pipeline_refresh()}")
except Exception:
    LOGGER.exception("Could not read pipeline refresh metadata")
    st.caption("Pipeline refresh time unavailable.")

with st.spinner("Loading role-scoped portfolio metrics…"):
    kpis = _load_kpis(role, agent_id)
if not kpis.empty:
    metrics = kpis.iloc[0]
    metric_columns = st.columns(5)
    metric_columns[0].metric("Customers in queue", f"{int(metrics['customers_in_queue'] or 0):,}")
    metric_columns[1].metric("Hardship alerts", f"{int(metrics['hardship_alerts'] or 0):,}")
    metric_columns[2].metric("Broken promises (30 days)", f"{int(metrics['broken_promises_30d'] or 0):,}")
    rate = metrics["promise_kept_rate_pct"]
    metric_columns[3].metric("Promise-kept rate", f"{float(rate):.1f}%" if pd.notna(rate) else "—")
    dpd = metrics["average_days_past_due"]
    metric_columns[4].metric("Average days past due", f"{float(dpd):.1f}" if pd.notna(dpd) else "—")

if role == "agent":
    tab_labels = ["Queue", "Customer 360", "Ask"]
elif role == "lead":
    tab_labels = ["Queue", "Ask"]
elif role == "analyst":
    tab_labels = ["Queue", "Customer 360", "Ask", "Audit log", "Fairness"]
elif role == "steward":
    tab_labels = ["Steward review", "Audit log"]
else:
    tab_labels = [
        "Queue", "Customer 360", "Ask", "Steward review",
        "Supervisor approvals", "Audit log", "Fairness",
    ]
tabs = st.tabs(tab_labels)
tab_map = dict(zip(tab_labels, tabs))

if "Queue" in tab_map:
    with tab_map["Queue"]:
        st.header("Assigned collections queue" if role == "agent" else "Collections queue")
        with st.spinner("Preparing role-scoped queue…"):
            queue_data = _render_queue(role, agent_id)
        st.caption("Displayed queue rows are bounded by the Layer 2 guard. No contact or payment is initiated.")

if "Customer 360" in tab_map:
    with tab_map["Customer 360"]:
        st.header("Customer 360")
        selected_id = st.session_state.get("selected_customer_id")
        if selected_id:
            with st.spinner("Loading masked customer portfolio…"):
                try:
                    _render_customer(str(selected_id), role, agent_id)
                except Exception:
                    LOGGER.exception("Customer 360 failed to load")
                    st.error("This customer view could not be loaded. Please retry.")
        else:
            st.info("Select a queue row or a demo customer to open its masked portfolio.")

if "Ask" in tab_map:
    with tab_map["Ask"]:
        st.header("Ask about approved collection data or policy")
        examples = [
            ("Count", "How many customers broke a promise in the last 30 days?"),
            ("Rate by group", "What is the promise kept rate by product type?"),
            ("Search notes", "What did customers say about losing their job?"),
            ("Policy", "What does policy say about disputes?"),
        ]
        example_columns = st.columns(len(examples))
        for column, (label, question) in zip(example_columns, examples):
            if column.button(label, key=f"ask_example_{label}"):
                st.session_state["pending_question"] = question

        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        for index, message in enumerate(st.session_state["messages"]):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and message.get("response"):
                    show_qa_response(message["response"], f"refusal_{index}")
                else:
                    st.markdown(message["content"])

        pending = st.session_state.pop("pending_question", None)
        question = st.chat_input("Ask a question about approved data or policy")
        question = question or pending
        if question:
            st.session_state["messages"].append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    with st.spinner("Checking scope and approved sources…"):
                        response = qa_client(role, user["id"], agent_id).ask(question)
                    st.session_state["last_response"] = response
                    show_qa_response(response, "refusal_current")
                    st.session_state["messages"].append({
                        "role": "assistant", "content": response["answer"], "response": response,
                    })
                except Exception:
                    LOGGER.exception("Offline QA request failed")
                    st.error("The question could not be answered. Please retry or contact the demo operator.")

if "Steward review" in tab_map:
    with tab_map["Steward review"]:
        st.header("Steward review")
        _render_steward_review()

if "Supervisor approvals" in tab_map:
    with tab_map["Supervisor approvals"]:
        st.header("Supervisor approvals")
        _render_supervisor_approvals(role)

if "Audit log" in tab_map:
    with tab_map["Audit log"]:
        st.header("Audit log")
        _render_audit_log()

if "Fairness" in tab_map:
    with tab_map["Fairness"]:
        st.header("Fairness")
        _render_fairness()
