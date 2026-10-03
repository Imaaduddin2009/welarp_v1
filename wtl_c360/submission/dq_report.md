# Full Maple Layer 1 quality report

Snapshot date: 2026-09-28. Counts are computed from the current DuckDB build; no source files were edited.

## Curated output counts

| Table | Rows |
|---|---:|
| crosswalk | 3,813,985 |
| customers | 1,261,197 |
| accounts | 12,569,692 |
| cases | 367,229 |
| promises | 238,815 |
| contacts | 2,164,693 |
| notes | 760,594 |
| transcripts | 25,000 |
| steward_queue | 463,889 |
| bronze_sources | 31 |

## Source row-count reconciliation

| Source | Schema rows | Observed rows | Result |
|---|---:|---:|---|
| customers | 1,020,000 | 1,020,000 | PASS |
| card_accounts | 660,000 | 660,000 | PASS |
| loan_accounts | 404,500 | 404,500 | PASS |
| deposit_accounts | 780,000 | 780,000 | PASS |
| collections_cases | 367,229 | 367,229 | PASS |
| contact_history | 2,688,337 | 2,688,337 | PASS |
| promises_to_pay | 238,815 | 238,815 | PASS |
| agent_notes | 760,594 | 760,594 | PASS |
| call_transcripts | 25,000 | 25,000 | PASS |
| voice_samples | 2,000 | 2,000 | PASS |
| external | 1,000,000 | 1,000,000 | PASS |
| loan_instalments | 14,832,472 | 14,832,472 | PASS |
| transactions | 12,401,924 | 12,401,924 | PASS |
| card_statements | 7,531,283 | 7,531,283 | PASS |
| account_monthly_snapshot | 12,569,692 | 12,569,692 | PASS |
| bureau_history | 4,679,562 | 4,679,562 | PASS |
| salary_credit_history | 6,000,000 | 6,000,000 | PASS |
| model_scores | 1,718,841 | 1,718,841 | PASS |
| offers | 40,767 | 40,767 | PASS |
| metric_definitions | 15 | 15 | PASS |
| benchmark_questions | 35 | 35 | PASS |
| reference_documents | 19 | 19 | PASS |
| hardship_programs | 18 | 18 | PASS |
| qa_checklist | 10 | 10 | PASS |
| agents | 1,000 | 1,000 | PASS |
| case_assignment_history | 423,830 | 423,830 | PASS |
| agent_shifts | 15,317 | 15,317 | PASS |
| channel_capacity | 287 | 287 | PASS |
| ops_events | 6 | 6 | PASS |
| geo_reference | 656 | 656 | PASS |
| code_lookups | 113 | 113 | PASS |

## Release manifest size reconciliation

Verified 26,153/26,153 entries by file presence and byte size. SHA-256 values are not recomputed.

## Automated checks

| Check | Description | Checked | Excluded | Failed | Pass rate | Result |
|---|---|---:|---:|---:|---:|---|
| unique_golden_id | golden_id unique | 1,261,197 | 0 | 0 | 100.00% | PASS |
| crosswalk_one_golden | one crosswalk row per source key | 3,813,985 | 0 | 0 | 100.00% | PASS |
| account_linked | accounts link to a customer | 12,569,692 | 0 | 0 | 100.00% | PASS |
| valid_dpd | days past due is nonnegative | 12,569,692 | 0 | 0 | 100.00% | PASS |
| valid_account_values | overdue estimates are nonnegative and do not exceed nonnegative balances; negative balances represent credits | 12,569,692 | 0 | 0 | 100.00% | PASS |
| negative_balance_overdue | credit/net-negative balances do not generate overdue estimates | 12,569,692 | 0 | 0 | 100.00% | PASS |
| valid_case_dates | case open dates parse | 367,229 | 0 | 0 | 100.00% | PASS |
| notes_linked | notes map to a customer | 760,594 | 0 | 0 | 100.00% | PASS |
| transcripts_linked | transcripts map to a customer | 25,000 | 0 | 0 | 100.00% | PASS |
| masked_note_contact_pii | notes have email and phone patterns masked | 760,594 | 0 | 0 | 100.00% | PASS |
| masked_note_customer_names | notes have the customer's first and last names masked | 760,594 | 0 | 0 | 100.00% | PASS |
| masked_transcript_contact_pii | transcripts have email and phone patterns masked | 25,000 | 0 | 0 | 100.00% | PASS |
| crosswalk_confidence_range | confidence in [0,1] | 3,813,985 | 0 | 0 | 100.00% | PASS |
| masked_transcript_customer_names | Transcripts have the customer's first and last names masked | 25,000 | 0 | 0 | 100.00% | PASS |
| source_row_count_customers | Source customers row count agrees with release schema | 1,020,000 | 0 | 0 | 100.00% | PASS |
| source_row_count_card_accounts | Source card_accounts row count agrees with release schema | 660,000 | 0 | 0 | 100.00% | PASS |
| source_row_count_loan_accounts | Source loan_accounts row count agrees with release schema | 404,500 | 0 | 0 | 100.00% | PASS |
| source_row_count_deposit_accounts | Source deposit_accounts row count agrees with release schema | 780,000 | 0 | 0 | 100.00% | PASS |
| source_row_count_collections_cases | Source collections_cases row count agrees with release schema | 367,229 | 0 | 0 | 100.00% | PASS |
| source_row_count_contact_history | Source contact_history row count agrees with release schema | 2,688,337 | 0 | 0 | 100.00% | PASS |
| source_row_count_promises_to_pay | Source promises_to_pay row count agrees with release schema | 238,815 | 0 | 0 | 100.00% | PASS |
| source_row_count_agent_notes | Source agent_notes row count agrees with release schema | 760,594 | 0 | 0 | 100.00% | PASS |
| source_row_count_call_transcripts | Source call_transcripts row count agrees with release schema | 25,000 | 0 | 0 | 100.00% | PASS |
| source_row_count_voice_samples | Source voice_samples row count agrees with release schema | 2,000 | 0 | 0 | 100.00% | PASS |
| source_row_count_external | Source external row count agrees with release schema | 1,000,000 | 0 | 0 | 100.00% | PASS |
| source_row_count_loan_instalments | Source loan_instalments row count agrees with release schema | 14,832,472 | 0 | 0 | 100.00% | PASS |
| source_row_count_transactions | Source transactions row count agrees with release schema | 12,401,924 | 0 | 0 | 100.00% | PASS |
| source_row_count_card_statements | Source card_statements row count agrees with release schema | 7,531,283 | 0 | 0 | 100.00% | PASS |
| source_row_count_account_monthly_snapshot | Source account_monthly_snapshot row count agrees with release schema | 12,569,692 | 0 | 0 | 100.00% | PASS |
| source_row_count_bureau_history | Source bureau_history row count agrees with release schema | 4,679,562 | 0 | 0 | 100.00% | PASS |
| source_row_count_salary_credit_history | Source salary_credit_history row count agrees with release schema | 6,000,000 | 0 | 0 | 100.00% | PASS |
| source_row_count_model_scores | Source model_scores row count agrees with release schema | 1,718,841 | 0 | 0 | 100.00% | PASS |
| source_row_count_offers | Source offers row count agrees with release schema | 40,767 | 0 | 0 | 100.00% | PASS |
| source_row_count_metric_definitions | Source metric_definitions row count agrees with release schema | 15 | 0 | 0 | 100.00% | PASS |
| source_row_count_benchmark_questions | Source benchmark_questions row count agrees with release schema | 35 | 0 | 0 | 100.00% | PASS |
| source_row_count_reference_documents | Source reference_documents row count agrees with release schema | 19 | 0 | 0 | 100.00% | PASS |
| source_row_count_hardship_programs | Source hardship_programs row count agrees with release schema | 18 | 0 | 0 | 100.00% | PASS |
| source_row_count_qa_checklist | Source qa_checklist row count agrees with release schema | 10 | 0 | 0 | 100.00% | PASS |
| source_row_count_agents | Source agents row count agrees with release schema | 1,000 | 0 | 0 | 100.00% | PASS |
| source_row_count_case_assignment_history | Source case_assignment_history row count agrees with release schema | 423,830 | 0 | 0 | 100.00% | PASS |
| source_row_count_agent_shifts | Source agent_shifts row count agrees with release schema | 15,317 | 0 | 0 | 100.00% | PASS |
| source_row_count_channel_capacity | Source channel_capacity row count agrees with release schema | 287 | 0 | 0 | 100.00% | PASS |
| source_row_count_ops_events | Source ops_events row count agrees with release schema | 6 | 0 | 0 | 100.00% | PASS |
| source_row_count_geo_reference | Source geo_reference row count agrees with release schema | 656 | 0 | 0 | 100.00% | PASS |
| source_row_count_code_lookups | Source code_lookups row count agrees with release schema | 113 | 0 | 0 | 100.00% | PASS |
| curated_coverage_contact_history | contact_history.csv: all eligible rows are curated and linked; expected exclusions are reported separately | 2,164,693 | 523,644 | 0 | 100.00% | PASS |
| curated_coverage_agent_notes | agent_notes.csv: all eligible rows are curated and linked; expected exclusions are reported separately | 760,594 | 0 | 0 | 100.00% | PASS |
| curated_coverage_collections_cases | collections_cases.csv: all eligible rows are curated and linked; expected exclusions are reported separately | 367,229 | 0 | 0 | 100.00% | PASS |
| curated_coverage_promises_to_pay | promises_to_pay.csv: all eligible rows are curated and linked; expected exclusions are reported separately | 238,815 | 0 | 0 | 100.00% | PASS |
| account_snapshot_dates_parsed | All account snapshot month-end dates parse using supported release formats | 12,569,692 | 0 | 0 | 100.00% | PASS |
| account_snapshot_linked | All valid-date account snapshot rows link through a CRM crosswalk key | 12,569,692 | 0 | 0 | 100.00% | PASS |
| l2_no_sensitive_columns | Masked Layer 2 contains no direct PII, national-ID hashes, or protected attributes | 66 | 0 | 0 | 100.00% | PASS |

Pass target: at least 99% of checked rows must satisfy each rule. Exclusions are rows outside a check's eligible input, not hidden failures.
Negative account balances are retained as valid net-credit balances; the estimated overdue amount is zero on those rows.

Identity scores are rule-based candidate scores, not calibrated match probabilities; no labeled Maple identity ground truth was supplied.

The steward queue preserves ambiguous candidates for review; unmatched identifiers remain isolated rather than being force-linked.
