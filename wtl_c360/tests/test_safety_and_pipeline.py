from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import duckdb
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import layer2
import layer4
from maple_layer1 import _flatten_policy_docs, _flatten_transcripts, _masked_text_sql
from score_benchmark import normalize_answer


@pytest.mark.parametrize("question", [
    "How many customers are male?",
    "List customers by gender",
    "Compare female and male borrowers",
    "How many customers have a marital status?",
    "Which citizens have overdue balances?",
    "Rank by newcomer status",
    "How many customers need accessibility support?",
    "Show vulnerable customers first",
    "Group cases by race",
    "Find customers by ethnicity",
    "List customers by religion",
    "Which nationality has the most debt?",
    "Sort by sexual orientation",
    "How many people are disabled?",
    "Show customers by household size",
    "Which accents get fewer calls?",
    "List immigration status",
    "How many dependants does each borrower have?",
    "Group customers by citizenship status",
    "What is the gender distribution in collections?",
    "Which postal areas should receive fewer hardship offers?",
])
def test_protected_attribute_questions_are_refused(question):
    assert not layer2.scope_check(question, {"role": "analyst"})["ok"]


@pytest.mark.parametrize("question", [
    "Give me the customer's phone number",
    "Show home addresses",
    "Find the date of birth",
    "Give me national ID numbers",
    "List full names of customers",
    "Show email addresses",
    "What is the customer's DOB?",
    "Return phone numbers for overdue customers",
    "Export every customer's data",
    "Dump customer records",
    "Delete all cases",
    "Update the overdue amount",
    "Drop the customer table",
    "Modify contact records",
    "Alter the approved schema",
    "Truncate the notes table",
    "Extract all customer details",
    "Download every customer record",
    "Overwrite promise history",
    "Insert a new customer",
])
def test_sensitive_or_write_questions_are_refused(question):
    assert not layer2.scope_check(question, {"role": "analyst"})["ok"]


@pytest.mark.parametrize("sql", [
    "DELETE FROM cases",
    "UPDATE cases SET status='closed'",
    "DROP TABLE cases",
    "SELECT * FROM read_csv('private.csv')",
    "SELECT * FROM secrets",
    "SELECT email FROM customers",
    "SELECT gender FROM customers",
    "SELECT * FROM cases; SELECT * FROM customers",
])
def test_sql_guard_blocks_unsafe_queries(sql):
    with pytest.raises(layer2.Blocked):
        layer2.guard(sql, {"role": "analyst"})


def test_lead_aggregate_is_cell_suppressed():
    sql = layer2.guard(
        "SELECT product_type, count(*) AS accounts FROM accounts GROUP BY 1",
        {"role": "lead"},
    )
    assert "HAVING COUNT(DISTINCT accounts.golden_id) >= 10" in sql


@pytest.mark.parametrize("sql", [
    "SELECT golden_id FROM accounts",
    "SELECT product_type FROM accounts",
    "SELECT golden_id, count(*) FROM accounts GROUP BY golden_id",
    "SELECT account_id, sum(balance) FROM accounts GROUP BY account_id",
])
def test_lead_cannot_view_customer_or_small_group_rows(sql):
    with pytest.raises(layer2.Blocked):
        layer2.guard(sql, {"role": "lead"})


def test_agent_queries_are_limited_to_assigned_portfolio():
    sql = layer2.guard(
        "SELECT golden_id FROM customers ORDER BY golden_id",
        {"role": "agent", "agent_id": "AG-17"},
    )
    assert "AG17" in sql
    assert "FROM customers" in sql


@pytest.mark.parametrize("question", [
    "How many call attempts can we make in a week?",
    "What does policy say about disputes?",
    "What must an agent do before taking payment?",
    "Can we sue after two missed payments?",
    "Summarise the most recent call transcript",
    "What did the customer say about lost work?",
])
def test_policy_and_call_questions_use_retrieval(question):
    assert layer2.route(question) == "text"


def test_benchmark_normalizes_case_and_whitespace_only():
    assert normalize_answer("  Hello\n WORLD ") == "hello world"


def test_tfidf_search_returns_notes_transcripts_and_policy_with_citations(tmp_path, monkeypatch):
    db_path = tmp_path / "l2.duckdb"
    with duckdb.connect(str(db_path)) as con:
        con.execute("CREATE TABLE notes(note_id VARCHAR, golden_id VARCHAR, note_text_masked VARCHAR)")
        con.execute("INSERT INTO notes VALUES ('N1','G1','hardship payment plan review'), ('N2','G2','hardship payment plan support')")
        con.execute("CREATE TABLE transcripts(transcript_id VARCHAR, golden_id VARCHAR, transcript_text_masked VARCHAR)")
        con.execute("INSERT INTO transcripts VALUES ('T1','G1','customer hardship payment plan')")
        con.execute("CREATE TABLE policy_chunks(chunk_id VARCHAR, doc_id VARCHAR, version VARCHAR, section VARCHAR, content VARCHAR)")
        con.execute("INSERT INTO policy_chunks VALUES ('P1','POL-X','2.0','§2','hardship payment plan policy')")
        con.execute("CREATE TABLE cases(golden_id VARCHAR, agent_id VARCHAR)")
        con.execute("INSERT INTO cases VALUES ('G1','AG1')")
    monkeypatch.setattr(layer2, "L2DB", str(db_path))
    search = layer2.NoteSearch()
    hits = search.search("hardship payment plan", user={"role": "analyst"})
    assert {hit["source_type"] for hit in hits} == {"note", "transcript", "policy"}
    assert all(hit["source_citation"] for hit in hits)


def test_agent_retrieval_keeps_policy_but_hides_other_customers(tmp_path, monkeypatch):
    db_path = tmp_path / "l2.duckdb"
    with duckdb.connect(str(db_path)) as con:
        con.execute("CREATE TABLE notes(note_id VARCHAR, golden_id VARCHAR, note_text_masked VARCHAR)")
        con.execute("INSERT INTO notes VALUES ('N1','G1','hardship payment plan review'), ('N2','G2','hardship payment plan support')")
        con.execute("CREATE TABLE transcripts(transcript_id VARCHAR, golden_id VARCHAR, transcript_text_masked VARCHAR)")
        con.execute("INSERT INTO transcripts VALUES ('T1','G2','customer hardship payment plan')")
        con.execute("CREATE TABLE policy_chunks(chunk_id VARCHAR, doc_id VARCHAR, version VARCHAR, section VARCHAR, content VARCHAR)")
        con.execute("INSERT INTO policy_chunks VALUES ('P1','POL-X','2.0','§2','hardship payment plan policy')")
        con.execute("CREATE TABLE cases(golden_id VARCHAR, agent_id VARCHAR)")
        con.execute("INSERT INTO cases VALUES ('G1','AG1')")
    monkeypatch.setattr(layer2, "L2DB", str(db_path))
    hits = layer2.NoteSearch().search(
        "hardship payment plan", user={"role": "agent", "agent_id": "AG1"}
    )
    assert all(hit["source_type"] == "policy" or hit["golden_id"] == "G1" for hit in hits)


def test_policy_chunks_exclude_superseded_versions(tmp_path):
    csv_path = ROOT / "data" / "data" / "reference_documents.csv"
    out_path = tmp_path / "chunks.csv"
    count = _flatten_policy_docs(csv_path, ROOT / "data" / "files", out_path)
    assert count > 0
    with out_path.open(encoding="utf-8", newline="") as source:
        documents = list(csv.DictReader(source))
    assert not any(row["doc_id"] == "SCR-MID-STD-EN-V1.0" for row in documents)
    assert any(row["doc_id"] == "SCR-MID-STD-EN" and row["version"] == "2.0" for row in documents)


def test_transcript_flattener_streams_turns(tmp_path):
    source = tmp_path / "transcripts" / "2026"
    source.mkdir(parents=True)
    (source / "one.json").write_text(json.dumps({
        "transcript_id": "T1",
        "contact_id": "C1",
        "case_id": "CS1",
        "crm_customer_id": "CRM1",
        "agent_id": "AG1",
        "call_start_ts": "2026-01-01 12:00:00",
        "turns": [{"turn_index": 1, "text": "hello"}, {"turn_index": 2, "text": "goodbye"}],
    }), encoding="utf-8")
    out_path = tmp_path / "turns.csv"
    assert _flatten_transcripts(tmp_path / "transcripts", out_path) == 2
    with out_path.open(encoding="utf-8", newline="") as source_file:
        rows = list(csv.DictReader(source_file))
    assert [row["turn_text"] for row in rows] == ["hello", "goodbye"]


def test_sql_text_mask_removes_names_email_and_phone():
    mask = _masked_text_sql(
        "x.txt",
        "p",
        r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
        r"\+?\d[\d\s().-]{8,}\d",
    )
    with duckdb.connect(":memory:") as con:
        result = con.execute(f"""
            SELECT {mask}
            FROM (SELECT 'Hi Jane Doe, call 416-555-0123 or jane@example.com.' AS txt) x
            CROSS JOIN (SELECT 'Jane' AS first_name, 'Doe' AS last_name) p
        """).fetchone()[0]
    assert "[NAME]" in result
    assert "[EMAIL]" in result
    assert "[PHONE]" in result
    assert "Jane" not in result and "Doe" not in result


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        ({"hardship_flag": True, "consent_phone": True}, "HUMAN_HARDSHIP_REVIEW"),
        ({"hardship_review_needed": True, "consent_phone": True}, "HUMAN_HARDSHIP_REVIEW"),
        ({"case_status": "disputed", "consent_phone": True}, "HOLD_FOR_CASE_REVIEW"),
        ({"outbound_calls_7d": 3, "consent_phone": True}, "NO_FURTHER_CALLS_7D"),
        ({"consent_phone": False, "consent_sms": False}, "NO_CONSENTING_CHANNEL"),
        ({"best_channel": "sms", "consent_sms": True}, "AGENT_REVIEW_SMS"),
    ],
)
def test_recommendations_obey_safety_gates(values, expected):
    action, rationale = layer4.recommend_action(pd.Series(values))
    assert action == expected
    assert rationale


def test_model_matrix_rejects_restricted_fields():
    with pytest.raises(ValueError, match="Restricted columns"):
        layer4._model_matrix(pd.DataFrame({"golden_id": ["G1"], "gender": ["F"]}))


def test_model_matrix_excludes_customer_identifier():
    matrix = layer4._model_matrix(pd.DataFrame({
        "golden_id": ["G1", "G2"],
        "as_of": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "total_balance": [1.0, 2.0],
    }))
    assert all("golden_id" not in column for column in matrix.columns)


def test_model_matrix_handles_nullable_boolean_values():
    matrix = layer4._model_matrix(pd.DataFrame({
        "golden_id": ["G1", "G2"],
        "hardship_flag": pd.Series([True, pd.NA], dtype="boolean"),
    }))
    assert matrix["hardship_flag"].tolist() == [1.0, 0.0]


def test_override_requires_rationale_and_reviewer(tmp_path):
    with pytest.raises(ValueError):
        layer4.record_override(tmp_path / "audit.jsonl", "G1", "A", "B", "reviewer", "")


def test_override_appends_auditable_event(tmp_path):
    audit = tmp_path / "audit.jsonl"
    layer4.record_override(audit, "G1", "A", "B", "reviewer", "manual check")
    event = json.loads(audit.read_text(encoding="utf-8").strip())
    assert event["golden_id"] == "G1"
    assert event["final_action"] == "B"
    assert event["reason"] == "manual check"


def test_temporal_training_rejects_empty_date_list():
    with pytest.raises(ValueError, match="At least one"):
        layer4.action_panel([])
