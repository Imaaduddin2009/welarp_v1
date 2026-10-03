# Data quality report (auto-generated)

## Issues found in source data

| Issue | Records affected | Out of | Handling |
|---|---|---|---|
| CRM duplicate customer records merged by matching | 63 | 3,213 | merged under one golden_id |
| dob: present but invalid -> quarantined (raw kept) | 165 | 10,658 | raw kept; normalised value NULL; logged in quarantine |
| dob: missing | 534 | 10,658 | left NULL; matching uses remaining fields |
| phone: present but invalid -> quarantined (raw kept) | 200 | 10,658 | raw kept; normalised value NULL; logged in quarantine |
| phone: missing | 600 | 10,658 | left NULL; matching uses remaining fields |
| postal_code: present but invalid -> quarantined (raw kept) | 1,010 | 10,658 | raw kept; normalised value NULL; logged in quarantine |
| postal_code: missing | 517 | 10,658 | left NULL; matching uses remaining fields |
| dob stored in a non-ISO format (mixed formats standardised) | 3,690 | 10,658 | parsed with 4 formats into one DATE type |
| exact duplicate account rows removed | 57 | 5,789 | removed (exact duplicates) |
| accounts set aside: negative days_past_due | 20 | 5,789 | kept out of c360_account; row retained in quarantine |
| accounts set aside: overdue exceeds balance | 30 | 5,789 | kept out of c360_account; row retained in quarantine |
| accounts set aside: no golden_id (unlinked customer) | 251 | 5,789 | kept out of c360_account; row retained in quarantine |
| notes containing phone/email, masked before Layer 2 | 547 | 1,960 | regex-masked in l2.notes |
| pairs routed to data-steward review | 683 | 18,279 | queued for human review |

## Contract rules (run on every refresh)

| # | Rule | Checked | Failed | Status | If it fails |
|---|---|---|---|---|---|
| 1 | golden_id present and unique in c360_customer | 3,150 | 0 | PASS | block publish |
| 2 | every source record maps to exactly one golden_id | 10,658 | 0 | PASS | block publish; steward |
| 3 | c360_account: every row has a valid golden_id | 5,431 | 0 | PASS | set rows aside |
| 3 | c360_promise: every row has a valid golden_id | 1,228 | 0 | PASS | set rows aside |
| 3 | c360_contact: every row has a valid golden_id | 1,960 | 0 | PASS | set rows aside |
| 3 | c360_case: every row has a valid golden_id | 811 | 0 | PASS | set rows aside |
| 3 | c360_note: every row has a valid golden_id | 1,960 | 0 | PASS | set rows aside |
| 4 | days_past_due>=0, overdue>=0, overdue<=balance (published accounts) | 5,431 | 0 | PASS | set rows aside |
| 5 | dob / phone / postal valid in customer record (raw kept) | 3,150 | 802 | WARN | set aside, keep raw |
| 6 | hardship and consent flags never empty | 3,150 | 0 | PASS | block publish |
| 7 | freshness: snapshot date within 2 days of expected | 1 | 0 | PASS | mark stale; alert owner |