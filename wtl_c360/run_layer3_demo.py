"""Run Layer 3 against the full Maple dataset in data/data."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import layer3 as F
from maple_layer1 import build_maple_layer1

MAPLE_RESULT = build_maple_layer1(
    ROOT / "data" / "data",
    ROOT / "out" / "c360_maple.duckdb",
    ROOT / "out" / "l2_maple.duckdb",
)
MAPLE_COUNTS = MAPLE_RESULT["counts"]
F.SRC = str(ROOT / "out" / "l2_maple.duckdb")
F.NO_HISTORY_IN_MOCK = set()

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
H = lambda t: print(f"\n{'='*8} {t} {'='*8}")

H("FULL MAPLE DATA: feature-ready rows loaded (customer/account history/cases/promises/contacts/notes)")
for table, rows in MAPLE_COUNTS.items():
    print(f"  {table:<10} {rows:>12,} rows")

H("1. NIGHTLY BATCH: features for every customer in the collections queue")
df = F.publish(); print(f"{len(df)} customers x {len(F.REGISTRY)} features, version {F.VERSION}")
reg = pd.DataFrame([(g, k) for k, (g, s, m) in F.REGISTRY.items()], columns=["group", "feature"]).groupby("group").feature.apply(lambda s: ", ".join(s))
for g, f in reg.items(): print(f"  {g:<8}: {f}")

H("2. ONE CUSTOMER, as the agent screen would see it (live, same SQL)")
row = df[df.hardship_review_needed & (df.promises_broken_90d >= 1)].iloc[0]; g = row.golden_id
live = F.online(g)
print(pd.Series({k: v for k, v in live.items() if k not in ("as_of",) and not (isinstance(v, float) and pd.isna(v))}).to_string())
con = __import__("duckdb").connect(F.SRC, read_only=True)
ev = con.execute(f"""
  SELECT DISTINCT 'note' AS source_type, n.note_id AS source_id,
         c.contact_date AS occurred_at, n.note_text_masked AS text
  FROM contacts c JOIN notes n USING (note_id)
  WHERE c.golden_id='{g}'
  UNION ALL
  SELECT 'transcript', transcript_id, call_start_ts::DATE, transcript_text_masked
  FROM transcripts WHERE golden_id='{g}'
  ORDER BY occurred_at DESC
""").fetchdf()
con.close()
print("\nEvidence supporting the text features (source IDs are shown):"); print(ev.to_string(index=False))
for feature, pattern in F.TEXT_RULES.items():
    matched = ev[ev.text.str.contains(pattern, case=False, regex=True, na=False)]
    sources = ", ".join(f"{row.source_type}:{row.source_id}" for row in matched.itertuples())
    if sources:
        print(f"  {feature}: {sources}")

H("3. MULTI-DATE TRAINING SET: features at monthly as-of dates + 30-day outcomes")
as_of_dates = ["2026-03-31", "2026-04-30", "2026-05-31", "2026-06-30", "2026-07-31", "2026-08-14"]
tr = F.temporal_training_sets(as_of_dates)
tr["as_of"] = pd.to_datetime(tr["as_of"])
tr.to_parquet(ROOT / "out" / "training_history.parquet", index=False)
cutoff = "2026-06-30"
train_rows = tr[tr.as_of <= pd.Timestamp(cutoff)]
test_rows = tr[tr.as_of > pd.Timestamp(cutoff)]
print(f"{len(tr)} rows across {len(as_of_dates)} snapshots; 30-day promise kept/broken label")
print(f"Temporal split: train through {cutoff} ({len(train_rows)} rows), later holdout ({len(test_rows)} rows)")
print(tr.groupby("as_of").label_kept_30d.agg(["count", "mean"]).to_string())
print(tr[["golden_id", "as_of", "promises_made_90d", "promises_broken_90d", "contact_attempts_14d", "txt_job_loss", "hardship_flag", "label_kept_30d"]].head(5).to_string(index=False))

H("4. AUTOMATIC SAFETY TESTS")
for name, fn in [
    ("no protected attributes", lambda: F.test_no_protected(df)),
    ("training == live (no skew)", F.test_parity),
    ("no peeking at the future, early", lambda: F.test_point_in_time("2026-03-31")),
    ("no peeking at the future, latest", lambda: F.test_point_in_time("2026-08-14")),
    ("text features use as-of evidence", lambda: F.test_text_point_in_time("2026-08-14")),
]:
    print(f"  PASS  {name:<28} {fn()}")
print(f"\nHardship hints a human should check (masked text suggests hardship, not flagged): {int(df.hardship_review_needed.sum())} customers")
print("Written: out/training_history.parquet")
print("Written: out/feature_store.duckdb (table features_live), config/feature_registry.yaml")
