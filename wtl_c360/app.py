"""Streamlit demonstration UI for agent, lead, and supervisor workflows."""
from __future__ import annotations

import json
import os
import sys
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

L2DB = Path(layer2.L2DB)
FEATURE_DB = Path(os.getenv("C360_FEATURE_DB", ROOT / "out" / "feature_store.duckdb"))
C360DB = Path(os.getenv("C360_C360_DB", ROOT / "out" / "c360_maple.duckdb"))
DECISION_AUDIT = ROOT / "out" / "decision_audit.jsonl"

st.set_page_config(page_title="Collections C360", layout="wide")
st.title("Collections C360")
st.caption("Decision support only. Verify consent, policy, hardship, disputes, and contact limits before acting.")


@st.cache_data(ttl=60)
def query_l2(sql: str, params: tuple = ()) -> pd.DataFrame:
    with duckdb.connect(str(L2DB), read_only=True) as con:
        return con.execute(sql, params).fetchdf()


@st.cache_data(ttl=60)
def query_features(sql: str, params: tuple = ()) -> pd.DataFrame:
    if not FEATURE_DB.is_file():
        return pd.DataFrame()
    with duckdb.connect(str(FEATURE_DB), read_only=True) as con:
        return con.execute(sql, params).fetchdf()


@st.cache_resource
def qa_client(role_name: str, identity: str, portfolio: str) -> layer2.QA:
    return layer2.QA({"id": identity, "role": role_name, "agent_id": portfolio})


def show_qa(user: dict[str, str]) -> None:
    question = st.text_input("Ask about approved collection data or policy")
    if st.button("Ask", disabled=not question):
        with st.spinner("Checking scope and approved sources…"):
            response = qa_client(user["role"], user["id"], user["agent_id"]).ask(question)
        st.write(response["answer"])
        st.caption(f"{response['route']} · {response['freshness']}")
        if response.get("sql_or_sources"):
            st.code(response["sql_or_sources"])
        if response.get("table") is not None:
            st.dataframe(response["table"], use_container_width=True)
        if response["refused"]:
            st.info(f"Refusal reason: {response['reason']}")


if not L2DB.is_file():
    st.error(f"Layer 2 database not found: {L2DB}. Run the full-data pipeline first.")
    st.stop()

roles = ["analyst", "agent", "lead", "supervisor"]
role = st.sidebar.selectbox("Role", roles)
agent_id = ""
if role == "agent":
    agents = query_l2("SELECT DISTINCT agent_id FROM cases WHERE agent_id IS NOT NULL ORDER BY 1")
    agent_id = st.sidebar.selectbox("Agent portfolio", agents.agent_id.tolist())
user = {"id": agent_id or role, "role": role, "agent_id": agent_id}

section = st.sidebar.radio(
    "Workspace",
    ["Queue", "Customer", "Ask", "Review & audit"] if role == "supervisor"
    else ["Queue", "Customer", "Ask"] if role in {"agent", "analyst"}
    else ["Queue summary", "Ask"],
)

if section == "Queue":
    st.header("Assigned collections queue")
    where = "WHERE c.status IN ('open', 'on_hold', 'disputed')"
    if role == "agent":
        where += " AND c.agent_id = ?"
    params = (agent_id,) if role == "agent" else ()
    queue = query_l2(f"""
        SELECT c.case_id, c.golden_id, c.agent_id, c.product_type, c.stage,
               c.status, c.opened_date
        FROM cases c {where}
        ORDER BY c.opened_date DESC
        LIMIT 500
    """, params)
    if not query_features("SELECT count(*) AS rows FROM features_live").empty:
        features = query_features("""
            SELECT golden_id, max_days_past_due, promises_broken_90d,
                   contact_attempts_14d, hardship_flag, hardship_review_needed,
                   best_channel, consent_phone, consent_sms, consent_email
            FROM features_live
        """)
        queue = queue.merge(features, on="golden_id", how="left", validate="many_to_one")
    st.dataframe(queue, use_container_width=True)
    st.caption("Queue rows are bounded to 500 for the demo; no action is initiated by the app.")

elif section == "Customer":
    if role == "lead":
        st.error("Team leads receive aggregate-only views.")
        st.stop()
    st.header("Customer and account context")
    golden_id = st.text_input("Golden customer ID")
    if golden_id:
        if role == "agent":
            access = query_l2(
                "SELECT count(*) AS n FROM cases WHERE golden_id=? AND agent_id=?",
                (golden_id, agent_id),
            ).iloc[0, 0]
            if not access:
                st.error("This customer is not in the selected agent's portfolio.")
                st.stop()
        account_rows = query_l2(
            "SELECT * FROM accounts WHERE golden_id=? ORDER BY snapshot_date DESC",
            (golden_id,),
        )
        case_rows = query_l2("SELECT * FROM cases WHERE golden_id=?", (golden_id,))
        contact_rows = query_l2("""
            SELECT c.contact_date, c.channel, c.direction, c.outcome,
                   n.note_id, n.note_text_masked
            FROM contacts c LEFT JOIN notes n USING (note_id)
            WHERE c.golden_id=? ORDER BY c.contact_date DESC LIMIT 50
        """, (golden_id,))
        profile = query_features("SELECT * FROM features_live WHERE golden_id=?", (golden_id,))
        st.subheader("Point-in-time features")
        st.dataframe(profile, use_container_width=True)
        st.subheader("Accounts")
        st.dataframe(account_rows, use_container_width=True)
        st.subheader("Cases")
        st.dataframe(case_rows, use_container_width=True)
        st.subheader("Recent masked contact evidence")
        st.dataframe(contact_rows, use_container_width=True)

elif section in {"Ask", "Queue summary"}:
    if section == "Queue summary":
        st.header("Team aggregate summary")
        summary = query_l2("""
            SELECT stage, count(*) AS cases,
                   count(DISTINCT golden_id) AS customers
            FROM cases GROUP BY stage
            HAVING count(DISTINCT golden_id) >= 10
            ORDER BY stage
        """)
        st.dataframe(summary, use_container_width=True)
    show_qa(user)

else:
    st.header("Supervisor review and audit")
    st.subheader("Identity candidates")
    if C360DB.is_file():
        with duckdb.connect(str(C360DB), read_only=True) as con:
            candidates = con.execute("""
                SELECT source_system, source_id, candidate_golden_id,
                       match_confidence, review_status
                FROM steward_queue
                ORDER BY match_confidence DESC
                LIMIT 500
            """).fetchdf()
        st.dataframe(candidates, use_container_width=True)
        st.caption("A recorded review decision is an audit event; it does not merge identities automatically.")
        if not candidates.empty:
            candidate_index = st.selectbox("Candidate row", range(len(candidates)))
            candidate = candidates.iloc[candidate_index]
            decision = st.selectbox("Review decision", ["accept", "reject", "defer"])
            reason = st.text_input("Required review rationale")
            if st.button("Record identity review", disabled=not reason.strip()):
                event = {
                    "source_system": candidate.source_system,
                    "source_id": candidate.source_id,
                    "candidate_golden_id": candidate.candidate_golden_id,
                    "decision": decision,
                    "reviewer": user["id"],
                    "reason": reason.strip(),
                }
                DECISION_AUDIT.parent.mkdir(parents=True, exist_ok=True)
                with DECISION_AUDIT.open("a", encoding="utf-8") as audit:
                    audit.write(json.dumps(event, ensure_ascii=False) + "\n")
                st.success("Review event recorded; the crosswalk was not changed.")

    st.subheader("Model action review")
    scored_path = ROOT / "out" / "layer4_scored_actions.csv"
    if scored_path.is_file():
        scored = pd.read_csv(scored_path)
        st.dataframe(scored.head(100), use_container_width=True)
        if not scored.empty:
            row_index = st.selectbox("Scored customer row", range(len(scored)))
            row = scored.iloc[row_index]
            action = st.selectbox(
                "Final human-reviewed action",
                sorted(set(scored.recommended_action.astype(str))),
            )
            rationale = st.text_input("Required action override / approval reason")
            if st.button("Record supervisor decision", disabled=not rationale.strip()):
                layer4.record_override(
                    DECISION_AUDIT, str(row.golden_id), str(row.recommended_action),
                    action, user["id"], rationale.strip(),
                )
                st.success("Supervisor decision appended to the audit log.")
    else:
        st.info("Layer 4 scoring artifact not found. Run run_layer4.py first.")

    if DECISION_AUDIT.is_file():
        st.subheader("Recent decision audit")
        events = [
            json.loads(line)
            for line in DECISION_AUDIT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        st.json(events[-100:])

    metrics_path = ROOT / "out" / "layer4_metrics.json"
    if metrics_path.is_file():
        st.subheader("Restricted fairness audit summary")
        st.json(json.loads(metrics_path.read_text(encoding="utf-8")).get("fairness_audit", {}))

    show_qa(user)
