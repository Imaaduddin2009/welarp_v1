"""Run Layer 3 against the full Maple dataset in data/data."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import layer3 as F
from maple_layer3_data import build_maple_l2

MAPLE_DB, MAPLE_COUNTS = build_maple_l2(ROOT / "data" / "data", ROOT / "out" / "l2_maple.duckdb")
F.SRC = str(MAPLE_DB)
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
ev = con.execute(f"SELECT c.contact_date, n.note_id, n.note_text_masked FROM contacts c JOIN notes n USING (note_id) WHERE c.golden_id='{g}' ORDER BY 1 DESC").fetchdf()
print("\nEvidence behind the text features (every text feature traces to a note):"); print(ev.to_string(index=False))

H("3. TRAINING SET: same features, as they looked 45 days ago + what happened next")
as_of = "2026-08-14"; tr = F.training_set(as_of)
print(f"as_of={as_of}: {len(tr)} training rows; label = first promise in the next 30 days was kept (1) or broken (0)")
print(tr.label_kept_30d.value_counts().rename({1: "kept", 0: "broken"}).to_string())
print(tr[["golden_id", "promises_made_90d", "promises_broken_90d", "contact_attempts_14d", "txt_job_loss", "hardship_flag", "label_kept_30d"]].head(5).to_string(index=False))

H("4. AUTOMATIC SAFETY TESTS")
for name, fn in [("no protected attributes", lambda: F.test_no_protected(df)), ("training == live (no skew)", F.test_parity),
                 ("no peeking at the future", lambda: F.test_point_in_time(as_of))]:
    print(f"  PASS  {name:<28} {fn()}")
print(f"\nHardship hints a human should check (notes suggest hardship, not flagged): {int(df.hardship_review_needed.sum())} customers")
print("Written: out/feature_store.duckdb (table features_live), config/feature_registry.yaml")
