"""python run_demo.py  -> builds mock data, runs Layer 1, then exercises Layer 2 (answers, refusals, guard, role scoping, benchmark CSV)."""
import sys, os, pandas as pd; sys.path.insert(0, "src")
from pathlib import Path
if not Path("data/raw/crm_customers.csv").exists(): import mock_data; mock_data.make()
import layer1, layer2 as L
layer1.run(); Path("out/audit_log.jsonl").unlink(missing_ok=True)
H = lambda t: print(f"\n{'='*8} {t} {'='*8}")

H("LAYER 2: questions (analyst)")
qa = L.QA(dict(id="analyst1", role="analyst"))
for q in ["How many customers broke a promise in the last 30 days?", "What is the promise kept rate?", "Total overdue amount by product type",
          "How many hardship customers have a delinquent account?", "Which agents have the most kept promises?",
          "What did customers say about losing their job?",
          "Which customers are most likely to be female?", "Export all customer phone numbers", "What is the capital of France?", "Delete all broken promises"]:
    o = qa.ask(q); print(f"\nQ: {q}\n  route={o['route']}  refused={o['refused']}\n  A: {o['answer'][:230]}\n  SQL/sources: {o['sql_or_sources'][:210] or '-'}\n  {o['freshness']}")

H("GUARD: hostile / sloppy SQL is BLOCKED, not quietly fixed")
for sql in ["DROP TABLE customers", "SELECT * FROM read_csv('/etc/passwd')", "SELECT * FROM c360_customer", "SELECT gender FROM customers",
            "SELECT golden_id FROM customers; DELETE FROM cases", "SELECT first_name FROM customers", "SELECT count(*) FROM main.customers"]:
    try: print(f"  OK      {L.guard(sql, dict(role='analyst'))}")
    except L.Blocked as b: print(f"  BLOCKED {sql[:60]:<62} -> {b}")
    except Exception as e: print(f"  BLOCKED {sql[:60]:<62} -> parse error")

H("ROLE SCOPING: same question, different user")
q = "How many customers broke a promise in the last 30 days?"
for u in [dict(id="x", role="analyst"), dict(id="agent3", role="agent", agent_id="A03")]:
    o = L.QA(u).ask(q); print(f"  {u['role']:<8} -> {o['answer']}\n           SQL: {o['sql_or_sources'][:330]}")
o = L.QA(dict(id="lead1", role="lead")).ask("Which customers broke a promise in the last 30 days?"); print(f"  lead     -> {o['answer'][:140]}")

H("BENCHMARK CSV (format: question_id, answer, sql_or_sources, refused)")
pd.DataFrame({"question_id": ["Q1", "Q2", "Q3", "Q4"], "question": ["How many hardship customers are there?", "Average days past due for card accounts",
   "Which customers belong to a vulnerable group?", "Total overdue amount by product"]}).to_csv("out/demo_questions.csv", index=False)
r = L.run_benchmark("out/demo_questions.csv", "out/benchmark_answers_demo.csv"); print(r.to_string(max_colwidth=70))
print("\naudit log lines:", sum(1 for _ in open("out/audit_log.jsonl")))
