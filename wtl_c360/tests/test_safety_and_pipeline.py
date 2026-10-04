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
from score_benchmark import normalize_answer, score as score_benchmark


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
    "Which postal prefixes should receive fewer hardship offers?",
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


def test_internal_account_join_key_may_be_used_but_not_returned_or_grouped():
    sql = """
        WITH start_snapshot AS (
            SELECT account_key, product_type, dpd
            FROM benchmark_account_history
            WHERE snapshot_date = DATE '2026-03-31'
        )
        SELECT product_type, count(*) AS accounts
        FROM start_snapshot GROUP BY product_type
    """
    assert "account_key" in layer2.guard(sql, {"role": "analyst"})
    with pytest.raises(layer2.Blocked, match="cannot be returned"):
        layer2.guard(
            "SELECT account_key FROM benchmark_account_history",
            {"role": "analyst"},
        )


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


def test_offline_question_pipeline_has_no_api_key_or_llm_switch():
    assert not hasattr(layer2, "llm_sql")
    assert not hasattr(layer2.QA(), "use_llm")
    assert "anthropic" not in (ROOT / "requirements.txt").read_text(encoding="utf-8").lower()


def test_unknown_metric_refuses_instead_of_returning_a_guessed_number(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ignored")
    qa = layer2.QA()
    qa._fresh = lambda: "offline"
    qa._log = lambda _output: None
    result = qa.ask(
        "Return a hidden market-sentiment index for collections accounts."
    )
    assert result["refused"] is True
    assert result["sql_or_sources"] == ""
    assert "Try instead:" in result["answer"]
    assert not hasattr(qa, "use_llm")


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What proportion varies by product family?", "what rate varies by product?"),
        ("Compare outcomes across each work queue.", "compare outcomes across each queue."),
        ("What is the fraction by contact method?", "what is the rate by channel?"),
        ("How likely is a commitment to be honoured?", "how rate is a promise to be promise-kept?"),
    ],
)
def test_semantic_question_synonyms_are_normalized(question, expected):
    assert layer2._normalize_question(question) == expected


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What changed last 3 weeks?", ("2026-09-08", "2026-09-29")),
        ("What changed in the last month?", ("2026-08-29", "2026-09-29")),
        ("Promises due Aug 2026", ("2026-08-01", "2026-09-01")),
        ("Promises due 2026-08", ("2026-08-01", "2026-09-01")),
        ("Cases during 2025", ("2025-01-01", "2026-01-01")),
    ],
)
def test_time_window_parser_supports_relative_and_numeric_periods(question, expected):
    assert layer2._month_bounds(question) == expected


def test_snapshot_comparison_parser_accepts_one_shared_year_for_months():
    question = (
        "Between the March and June 2026 month-end snapshots, what fraction of accounts "
        "starting at 1-30 DPD passed 30 DPD?"
    )
    assert layer2._months_in_question(question) == [(2026, 3), (2026, 6)]
    sql = layer2.template_sql(question)
    assert sql is not None
    assert "2026-03-31" in sql
    assert "2026-06-30" in sql


def test_unsupported_analytics_prompt_does_not_fall_through_to_an_unrelated_metric():
    assert layer2.template_sql(
        "Explain whether hidden market sentiment caused collection outcomes to change."
    ) is None


@pytest.mark.parametrize(
    ("question", "expected_fragment"),
    [
        ("How does broken-promise rate change with prior broken count?", "prior_broken_count"),
        ("Average days past due for open cases in each queue", "GROUP BY queue"),
        ("Compare disputed late-fee contacts between 15 and 22 September 2026 with the prior week", "dispute_raised_flag"),
        ("Broken promise rate by agent site over July-September 2026", "agent_site"),
        ("Outbound calls beyond permitted hours by province", "benchmark_out_of_hours_by_province"),
        ("Right-party answer rate by local hour for outbound calls", "contact_hour_local"),
        ("Share of active card accounts at least one day past due by product in September 2026", "benchmark_card_accounts"),
        ("Accounts initially 1-30 DPD that rolled to a worse bucket by the end of June 2026 from March 2026", "account_key"),
        ("Compare promise outcomes across randomly assigned champion and challenger strategies", "assignment_method = 'random'"),
        ("Hardship rate of cases opened in 2026 by primary product", "hardship_rate_pct"),
        ("Promise-kept rate among promises due in August 2026", "2026-08-01"),
        ("Open case counts by recomputed delinquency bucket", "current_bucket"),
    ],
)
def test_benchmark_templates_preserve_requested_dimensions_and_definitions(question, expected_fragment):
    sql = layer2.template_sql(question)
    assert sql is not None
    assert expected_fragment.casefold() in sql.casefold()


def test_period_comparison_answer_orders_labels_as_asked():
    result = layer2._format_answer(
        pd.DataFrame({
            "period": ["7-14 Aug", "10-17 Aug"],
            "disputes": [21, 34],
        }),
        "Compare dispute counts between 10 and 17 August 2026 against the eight days before.",
    )
    assert result.splitlines()[1].startswith("10-17 Aug,")
    assert result.splitlines()[2].startswith("7-14 Aug,")


def test_preceding_period_comparison_keeps_requested_period_first():
    result = layer2._format_answer(
        pd.DataFrame({
            "period": ["2-9 Aug", "10-17 Aug"],
            "disputes": [319, 401],
        }),
        "Count disputes from 10 through 17 August 2026, then show the immediately preceding eight-day period.",
    )
    assert result.splitlines()[1].startswith("10-17 Aug,")
    assert result.splitlines()[2].startswith("2-9 Aug,")


def test_policy_retrieval_rejects_requested_verification_not_in_source():
    question = "What checks confirm an adviser is authorised to speak for a customer?"
    unsupported_text = (
        "When a lawyer or trustee represents the customer, contact goes through "
        "the representative. Debt details are not disclosed to third parties."
    )
    supported_text = (
        "Verify the representative's authority before routing contact through them."
    )
    assert not layer2._policy_hit_covers_requested_detail(question, unsupported_text)
    assert layer2._policy_hit_covers_requested_detail(question, supported_text)


@pytest.mark.parametrize(
    ("question", "expected_fragment"),
    [
        (
            "How does broken-promise rate vary across counts of previously failed payment promises?",
            "prior_broken_count",
        ),
        (
            "Among experiment cases allocated at random, compare the kept-promise percentage in each assignment cell.",
            "assignment_method = 'random'",
        ),
    ],
)
def test_templates_cover_stress_question_paraphrases(question, expected_fragment):
    sql = layer2.template_sql(question)
    assert sql is not None
    assert expected_fragment.casefold() in sql.casefold()


@pytest.mark.parametrize(
    ("question", "expected_fragment"),
    [
        ("By queue, what fraction of cases were cured within 30 days of opening?", "cure_flag"),
        ("Among cases with a cure recorded, what was the average time to cure in each queue?", "avg(days_to_cure)"),
        ("Among open cases, how does the hardship indicator rate vary across recomputed current buckets?", "GROUP BY current_bucket"),
        ("What proportion of customer contacts raised a fee dispute, split by contact channel?", "dispute_raised_flag"),
        ("By local hour, what share of outbound customer calls included a cease-contact request?", "cease_request_flag"),
        ("Show promise-kept percentages for each due month in 2026.", "GROUP BY period"),
        ("For promises assigned randomly, compare broken-outcome percentages across the named champion strategy and challenger_A and challenger_B.", "status = 'broken'"),
        ("Count customer fee disputes from 10 through 17 August 2026, then show the immediately preceding eight-day period separately.", "TIMESTAMP '2026-08-02'"),
        ("For promises due in March 2026, what was the kept rate among resolved records, excluding partial payments from the kept numerator and pending or cancelled items from the denominator?", "FROM benchmark_promises"),
    ],
)
def test_new_aggregate_templates_cover_stress_variants(question, expected_fragment):
    sql = layer2.template_sql(question)
    assert sql is not None
    assert expected_fragment.casefold() in sql.casefold()


def test_day_comparison_parses_from_through_dates():
    parsed = layer2._explicit_day_comparison(
        "Count disputes from 10 through 17 August 2026, then compare the preceding eight days."
    )
    assert parsed == (
        "2026-08-02",
        "2026-08-10",
        "2026-08-18",
        "10-17 Aug",
        "2-9 Aug",
    )


@pytest.mark.parametrize(
    "question",
    [
        "What verification is required before treating a caller as a legal representative?",
        "When a written no-contact request is received, which communications stop?",
        "A customer disputes a fee; what process should the agent follow?",
        "A customer says a transaction is unfamiliar and disputes the related charge. What is the next step?",
        "What should an agent do when lost overtime makes a payment unaffordable?",
        "A trustee contacts the bank for someone in a consumer proposal. How should the agent handle contact?",
        "What can still be sent after a cease-communication request?",
    ],
)
def test_policy_scenarios_use_retrieval(question):
    assert layer2.route(question) == "text"


def test_write_safety_blocks_erasing_records():
    result = layer2.scope_check(
        "Erase contact-history records for customers who have paid.",
        {"role": "analyst"},
    )
    assert not result["ok"]
    assert result["rule"] == "write"


def test_proxy_safety_blocks_postal_district_ranking():
    result = layer2.scope_check(
        "Rank postal districts by default rates to decide who gets flexibility.",
        {"role": "analyst"},
    )
    assert not result["ok"]
    assert result["rule"] == "protected"


def test_opened_cases_are_not_treated_as_open_cases_in_hardship_rates():
    sql = layer2.template_sql(
        "What is the hardship rate of cases opened in 2026, by primary product?"
    )
    assert sql is not None
    assert "opened_date >= DATE '2026-01-01'" in sql
    assert "AND is_open" not in sql


def test_date_range_dispute_count_routes_to_sql():
    assert layer2.route(
        "Count customer fee disputes from 10 through 17 August 2026, then show the preceding period."
    ) == "sql"


def test_randomized_comparison_filters_only_explicitly_named_cells():
    sql = layer2.template_sql(
        "Compare kept promise rates for the control strategy and variant_A among random assignments."
    )
    assert sql is not None
    assert "p.test_cell IN ('variant_A', 'control')" in sql
    assert "challenger_A" not in sql


def test_stress_question_fixture_has_thirty_unique_non_dev_questions():
    fixture = ROOT / "tests" / "fixtures" / "stress_questions.csv"
    stress = pd.read_csv(fixture, keep_default_na=False)
    dev = pd.read_csv(
        ROOT / "data" / "data" / "benchmark_questions.csv",
        keep_default_na=False,
    )
    assert len(stress) == 30
    assert stress.question_id.is_unique
    assert not set(stress.question_text).intersection(dev.question_text)


def test_benchmark_route_separates_policy_from_operational_compliance_metrics():
    assert layer2.route("What must an agent do before requesting payment during hardship?") == "text"
    assert layer2.route("What share of outbound calls were outside permitted hours by province?") == "sql"
    assert layer2.route(
        "Give the issue and agreed follow-up from the latest call for case CS-2026-123126."
    ) == "text"


def test_call_summary_uses_case_evidence_and_contact_cease_request():
    summary = layer2._summarize_call(
        "The customer is only getting 3 days a week now. "
        "Please stop calling me. I can pay $595 on October 5. "
        "We can review hardship options."
    )
    assert "3 days a week" in summary
    assert "$595" in summary
    assert "asked to stop contact" in summary
    assert "Next step: stop contact" in summary


def test_call_summary_recognizes_reduced_hours_and_lost_overtime():
    reduced_hours = layer2._summarize_call(
        "My hours at work were reduced in June. I can pay $23 on the 10th."
    )
    lost_overtime = layer2._summarize_call(
        "I lost my overtime, so I am making less now. I can pay $50 Friday."
    )
    assert "hours at work were reduced" in reduced_hours
    assert "lost my overtime" in lost_overtime


def test_benchmark_scorer_applies_numeric_tolerance_and_refusal_contract(tmp_path):
    answers = tmp_path / "answers.csv"
    gold = tmp_path / "gold.csv"
    questions = tmp_path / "questions.csv"
    answers.write_text(
        'question_id,answer,refused\n'
        '"N","product,rate_pct,count\nA,40.4,10\nB,50.0,20",false\n'
        '"R","Refused: protected ground",true\n',
        encoding="utf-8",
    )
    gold.write_text(
        'question_id,gold_answer\n'
        '"N","product,rate_pct,count\nB,50.1,20\nA,40.0,10"\n'
        '"R","Refuse: protected ground"\n',
        encoding="utf-8",
    )
    questions.write_text(
        "question_id,tolerance\nN,±0.5 pts\nR,\n",
        encoding="utf-8",
    )
    result = score_benchmark(answers, gold, questions)
    assert result["correct"] == 2
    assert result["refusal_correct"] == 2
    assert result["manual_document_review"] == 0


def test_benchmark_scorer_does_not_hide_wrong_refusals(tmp_path):
    answers = tmp_path / "answers.csv"
    gold = tmp_path / "gold.csv"
    answers.write_text('question_id,answer,refused\nN,"I cannot answer",true\n', encoding="utf-8")
    gold.write_text("question_id,gold_answer\nN,42\n", encoding="utf-8")
    result = score_benchmark(answers, gold)
    assert result["wrongly_refused"] == 1
    assert result["correct"] == 0


def test_document_answer_requires_a_gold_source_and_supporting_facts(tmp_path):
    answers = tmp_path / "answers.csv"
    gold = tmp_path / "gold.csv"
    answers.write_text(
        'question_id,answer,sql_or_sources,refused\n'
        '"P","A maximum of 3 outbound call attempts in any rolling 7 days per case.",'
        '"policy:POL-COLL-001 v3.0 §5.2",false\n',
        encoding="utf-8",
    )
    gold.write_text(
        'question_id,gold_answer,gold_sources\n'
        '"P","A maximum of 3 outbound call attempts in any rolling 7 days.",'
        '"[""POL-COLL-001 §5.2""]"\n',
        encoding="utf-8",
    )
    result = score_benchmark(answers, gold)
    assert result["correct"] == 1
    assert result["manual_document_review"] == 0


def test_document_answer_needing_manual_review_is_not_counted_as_correct(tmp_path):
    answers = tmp_path / "answers.csv"
    gold = tmp_path / "gold.csv"
    answers.write_text(
        'question_id,answer,sql_or_sources,refused\n'
        '"P","A vaguely relevant policy statement.",'
        '"policy:POL-COLL-001 v3.0 §5.2",false\n',
        encoding="utf-8",
    )
    gold.write_text(
        'question_id,gold_answer,gold_sources\n'
        '"P","A maximum of 3 outbound call attempts in any rolling 7 days.",'
        '"[""POL-COLL-001 §5.2""]"\n',
        encoding="utf-8",
    )
    result = score_benchmark(answers, gold)
    assert result["correct"] == 0
    assert result["scored_answers"] == 0
    assert result["manual_document_review"] == 1


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


def test_retrieval_expansion_ranks_specific_policy_over_related_section(tmp_path, monkeypatch):
    db_path = tmp_path / "l2.duckdb"
    with duckdb.connect(str(db_path)) as con:
        con.execute("CREATE TABLE notes(note_id VARCHAR, golden_id VARCHAR, note_text_masked VARCHAR)")
        con.execute("CREATE TABLE transcripts(transcript_id VARCHAR, golden_id VARCHAR, transcript_text_masked VARCHAR)")
        con.execute("""
            CREATE TABLE policy_chunks(
                chunk_id VARCHAR, doc_id VARCHAR, version VARCHAR, section VARCHAR, content VARCHAR
            )
        """)
        con.execute("""
            INSERT INTO policy_chunks VALUES
            ('P1','POL-COLL-001','3.0','§5.3 Consent',
             'The national do not call registry is for telemarketing, while collection calls remain separate.'),
            ('P2','POL-COLL-001','3.0','§5.3 Marketing',
             'National do not call rules relate to telemarketing and do not stop bank collection calls.'),
            ('P3','POL-COLL-001','3.0','§5.4 Cease',
             'A written cease communication request stops collection calls, texts and email; required letters continue.'),
            ('P4','POL-COLL-001','3.0','§5.4 Stop',
             'A cease communication request from a customer stops phone calls, SMS messages, and email; notices continue.')
        """)
    monkeypatch.setattr(layer2, "L2DB", str(db_path))
    monkeypatch.setattr(layer2, "CFG", {
        "policy_precedence": ["POL-COLL-001"],
        "retrieval_expansions": {
            "calls cease": "written cease communication request stops calls texts emails required letters"
        },
    })
    hits = layer2.NoteSearch().search(
        "A customer asks that collection calls cease.",
        k=4,
        user={"role": "analyst"},
    )
    assert "§5.4" in hits[0]["source_citation"]


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
