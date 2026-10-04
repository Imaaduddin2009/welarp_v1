# Layer 4 temporal model and decision-support report

- Label: first promise outcome recorded within 30 days after an as-of date (kept=1, broken=0).
- Training rows: 8,807; latest training as-of: 2026-06-30.
- Holdout rows: 13,539; earliest holdout as-of: 2026-07-31.
- LightGBM ROC AUC: 0.6007382017109636; Brier: 0.2508.
- Logistic baseline ROC AUC: 0.6127953716018266; Brier: 0.2377.
- Top-decile observed kept rate: 0.506; this is ranking yield, not causal uplift or policy value.
- The training target is conditional on observed collections/contact activity and cannot estimate counterfactual action effects.
- The LightGBM model uses Layer 2 operational features only. Restricted demographics are joined only in the separate group fairness audit.
- Recommendations are deterministic safety gates, not autonomous contact or offer actions. Hardship, disputes, consent, and rolling call limits route to human review or suppression.
- Per-record explanations are model contribution summaries and do not establish causal reasons.
- Fairness groups below the configured minimum size are suppressed; group gaps require human compliance review.

Artifacts: layer4_scored_actions.csv, layer4_explanations.csv, layer4_lightgbm.txt, layer4_metrics.json, fairness_report.json.
