# Collections C360

Collections C360 is a local decision-support demo built on the complete Maple release under `data/data/`. It preserves the small mock-data demo, but the main path now loads the full release through Layer 1, publishes a physically separate masked Layer 2 database, builds point-in-time Layer 3 features, and evaluates an advisory Layer 4 model.

## Setup

Use Python 3.12 or newer:

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python run_demo.py                         # fast mock-data Layer 1 + Layer 2 demo
python run_layer3_demo.py                   # full Maple Layer 1 -> masked L2 -> L3
python run_layer4.py                        # temporal outcome model and fairness report
python scripts/run_benchmark.py             # 35 organiser questions; four-column CSV
python scripts/score_benchmark.py           # score the dev-answer subset
python -m streamlit run app.py              # agent / analyst / lead / supervisor UI
python run_all.py                           # tests, full pipeline, model, benchmark, scoring
```

The full-data pipeline expects the supplied release at `data/data`, documents at `data/files`, and the release manifest/schema under `data`. Outputs are generated under `out/` and `submission/`; do not edit the source release. The main databases are:

- `out/c360_maple.duckdb`: bronze source views, curated C360, identity candidates, restricted audit attributes, and DQ artifacts.
- `out/l2_maple.duckdb`: approved, masked Layer 2 tables only.
- `out/feature_store.duckdb`: latest nightly feature snapshot.
- `out/training_history.parquet`: multiple point-in-time feature/label snapshots.

To use another Layer 2 database with the UI or scripts, set `C360_L2_DB`. Optional LLM-backed NL-to-SQL is enabled only when `ANTHROPIC_API_KEY` is set. SQL validation, role scope, aggregation suppression, and read-only execution still apply.

## Pipeline

1. **Layer 1** exposes all 31 CSV/Parquet source tables through DuckDB bronze views, reconciles counts to `data/schema/schema.json`, and checks the release manifest file sizes. It creates identity candidates and a steward queue, curated C360 records, a contract, sensitivity inventory, and data-quality report.
2. **Layer 2** publishes an allowlisted masked database, with no direct names, email/phone, national-ID hashes, or protected attributes. It supports deterministic templates and optional guarded NL-to-SQL, policy/document retrieval, deduplicated note/transcript retrieval with citations, role-based portfolio scoping, lead aggregate-only queries, minimum group size 10, and audit logging.
3. **Layer 3** defines feature logic once for training, batch, and online paths. It supports monthly as-of snapshots, a chronological train/holdout split, temporal label windows, text features from masked notes and transcripts, and explicit evidence IDs.
4. **Layer 4** trains LightGBM against a logistic-regression baseline using observed recent contact actions and future promise outcomes. It reports ROC AUC, Brier score and calibration bins, offers policy-gated workflow routing, per-row feature contributions, a restricted-data-only group fairness audit, and human override logging.

The Streamlit UI provides bounded queue views, an assigned agent portfolio, customer/account context, guarded Q&A, identity candidate review, model action review, and audit views. It never initiates contact or executes a payment plan.

## Safety and limitations

- The account snapshot has no direct overdue-dollar field. `overdue_amount` is an estimate: `max(minimum_due - payment_made, 0)` when DPD is positive, capped at a nonnegative balance. Negative balances are retained as net credits with zero estimated overdue.
- System events, invalid contact timestamps, and empty customer references are excluded from curated agent contacts. These exclusions are counted separately in `submission/dq_report.md`.
- Cross-system identity matching uses DOB/name/phone/postal blocking with a custom weighted score. It is not Splink, and its score is not a calibrated match probability. Ambiguous candidates remain isolated pending review; the observed steward queue is large. The synthetic release has no labeled identity truth set, so precision/recall are not measured.
- Free-text PII masking uses regex and source-customer first/last names. It is not a guarantee against every indirect identifier; review the generated DQ report and do not send unmasked source data to an LLM.
- Retrieval currently uses sparse TF-IDF over masked notes, transcripts, and current non-superseded policy documents. It is not a sentence-transformer/FAISS index.
- Layer 4 labels reflect observed promise outcomes after recorded contacts. They do not identify causal action effects or counterfactual policy value. Recommendations are workflow gates for a human, not autonomous decisions. Small fairness groups are suppressed.
- Synthetic source data is for demonstration and must not be used for real collections decisions.

See [submission/pre_submission_checklist.md](submission/pre_submission_checklist.md), [submission/architecture.mmd](submission/architecture.mmd), and [submission/demo_script.md](submission/demo_script.md).
