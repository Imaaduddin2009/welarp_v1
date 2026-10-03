# Pre-submission checklist

- [ ] Install the pinned runtime ranges with `python -m pip install -r requirements.txt`.
- [x] Confirm the full Maple source, transcripts, policy documents, schema, and manifest are present under `data/`.
- [x] Run `python -m pytest -q`; all safety and helper tests pass (79 tests).
- [x] Run `python run_layer3_demo.py`; all 31 source tables reconcile, all 26,153 manifest entries match file size, all 52 DQ checks pass, Layer 2 privacy checks pass, and temporal/batch-live/PIT checks pass.
- [x] Review `submission/dq_report.md`, `submission/dq_results.json`, `submission/sensitivity_inventory.csv`, and the identity steward queue size (463,889 candidates).
- [x] Run `python run_layer4.py`; the temporal model, logistic baseline, fairness audit, and human decision gates execute. Holdout LightGBM ROC AUC is 0.601 and the logistic baseline ROC AUC is 0.613; these are weak predictive results, not a deployment signal.
- [x] Run `python scripts/run_benchmark.py` and `python scripts/score_benchmark.py`. The run produced 35 answers (14 refusals); on the 21-question dev subset it scored 0 exact matches, 7 refusals, and 0 missing answers. Benchmark answer quality is not submission-ready and needs improvement.
- [x] Launch `python -m streamlit run app.py`; startup health endpoint returned HTTP 200.
- [ ] Interactively verify agent portfolio scoping, lead aggregation suppression, and supervisor audit in the Streamlit UI.
- [ ] Verify no source data, secrets, or generated database files are included in any external submission.
- [ ] Review the demo script, architecture diagram, current policy versions, and all known limitations in the README.
