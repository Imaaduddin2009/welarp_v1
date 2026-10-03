"""Build the masked Layer-2 tables used by Layer 3 from the full Maple release."""
from __future__ import annotations

import os
from pathlib import Path

import duckdb


def build_maple_l2(data_dir: str | Path, destination: str | Path) -> tuple[Path, dict[str, int]]:
    """Load the full customer and collections history into a feature-safe DuckDB file."""
    data_dir = Path(data_dir).resolve()
    destination = Path(destination).resolve()
    required = (
        "customers.csv",
        "collections_cases.csv",
        "contact_history.csv",
        "promises_to_pay.csv",
        "agent_notes.csv",
        "external.csv",
        "bureau_history",
        "account_monthly_snapshot",
    )
    missing = [str(data_dir / name) for name in required if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError("Missing Maple data inputs: " + ", ".join(missing))

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.stem + ".building" + destination.suffix)
    staging.unlink(missing_ok=True)

    def sql_path(path: Path) -> str:
        return path.as_posix().replace("'", "''")

    def csv(path: Path) -> str:
        return (
            f"read_csv('{sql_path(path)}', header=true, all_varchar=true, "
            "sample_size=10000)"
        )

    con = duckdb.connect(str(staging))
    try:
        customers_csv = csv(data_dir / "customers.csv")
        cases_csv = csv(data_dir / "collections_cases.csv")

        con.execute(f"""
            CREATE TEMP TABLE _customer_keys AS
            SELECT crm_customer_id,
                   nullif(trim(national_id_hash), '') AS national_id_hash
            FROM {customers_csv}
            WHERE nullif(trim(crm_customer_id), '') IS NOT NULL
        """)
        con.execute("""
            CREATE TEMP TABLE _crm_id_map AS
            WITH keyed AS (
              SELECT crm_customer_id,
                     CASE WHEN national_id_hash IS NOT NULL
                          THEN min(crm_customer_id) OVER (PARTITION BY national_id_hash)
                          ELSE crm_customer_id END AS canonical_crm_id
              FROM _customer_keys
            )
            SELECT crm_customer_id,
                   'G' || sha256(canonical_crm_id) AS golden_id
            FROM keyed
        """)
        con.execute("DROP TABLE _customer_keys")

        con.execute(f"""
            CREATE TEMP TABLE _bureau_id_map AS
            WITH external AS (
              SELECT DISTINCT bank_subject_ref, subject_dob_reported,
                     lower(strip_accents(trim(split_part(subject_name_reported, ',', 1)))) AS last_name,
                     lower(strip_accents(trim(split_part(
                       trim(split_part(subject_name_reported, ',', 2)), ' ', 1
                     )))) AS first_name
              FROM {csv(data_dir / "external.csv")}
            ),
            candidates AS (
              SELECT DISTINCT e.bank_subject_ref, m.golden_id
              FROM external e
              JOIN {customers_csv} c
                ON try_cast(e.subject_dob_reported AS DATE) = try_cast(c.date_of_birth AS DATE)
               AND e.first_name = lower(strip_accents(trim(c.first_name)))
               AND e.last_name = lower(strip_accents(trim(c.last_name)))
              JOIN _crm_id_map m ON m.crm_customer_id = c.crm_customer_id
              WHERE nullif(trim(e.bank_subject_ref), '') IS NOT NULL
            ),
            unambiguous AS (
              SELECT bank_subject_ref, min(golden_id) AS golden_id
              FROM candidates
              GROUP BY bank_subject_ref
              HAVING count(DISTINCT golden_id) = 1
            )
            SELECT bank_subject_ref, golden_id FROM unambiguous
        """)
        con.execute(f"""
            CREATE TABLE bureau_scores AS
            SELECT m.golden_id,
                   try_cast(b.pull_date AS DATE) AS score_date,
                   try_cast(b.risk_score AS DOUBLE) AS bureau_score
            FROM read_parquet('{sql_path(data_dir / "bureau_history" / "year=*" / "*.parquet")}',
                              hive_partitioning=true) b
            JOIN _bureau_id_map m ON m.bank_subject_ref = b.bank_subject_ref
            WHERE try_cast(b.pull_date AS DATE) IS NOT NULL
              AND try_cast(b.risk_score AS DOUBLE) IS NOT NULL
        """)

        con.execute(f"""
            CREATE TABLE cases AS
            SELECT c.case_id,
                   m.golden_id,
                   lower(trim(c.case_status)) AS status,
                   CASE
                     WHEN try_cast(c.entry_dpd AS INTEGER) <= 30 THEN 'early'
                     WHEN try_cast(c.entry_dpd AS INTEGER) <= 90 THEN 'mid'
                     ELSE 'late'
                   END AS stage,
                   try_cast(c.case_open_date AS DATE) AS opened_date,
                   try_cast(c.case_status_date AS DATE) AS status_date,
                   lower(trim(c.primary_product)) AS product_type,
                   coalesce(try_cast(c.hardship_flag AS BOOLEAN), false) AS hardship_flag,
                   coalesce(try_cast(c.hardship_plan_start AS DATE),
                            try_cast(c.case_open_date AS DATE)) AS hardship_start_date
            FROM {cases_csv} c
            JOIN _crm_id_map m ON m.crm_customer_id = c.coll_customer_ref
            WHERE try_cast(c.case_open_date AS DATE) IS NOT NULL
        """)

        con.execute(f"""
            CREATE TABLE customers AS
            SELECT m.golden_id,
                   arg_min(c.preferred_language, c.crm_customer_id) AS language,
                   arg_min(c.segment, c.crm_customer_id) AS segment,
                   coalesce(bool_or(k.hardship_flag), false) AS hardship_flag,
                   max(try_cast(c.consent_last_updated_ts AS TIMESTAMP)::DATE) AS consent_as_of_date,
                   coalesce(arg_max(try_cast(c.consent_sms AS BOOLEAN),
                                    try_cast(c.consent_last_updated_ts AS TIMESTAMP)), false) AS consent_sms,
                   coalesce(arg_max(try_cast(c.consent_email AS BOOLEAN),
                                    try_cast(c.consent_last_updated_ts AS TIMESTAMP)), false) AS consent_email,
                   coalesce(arg_max(try_cast(c.consent_call AS BOOLEAN)
                                    OR try_cast(c.consent_call_mobile AS BOOLEAN),
                                    try_cast(c.consent_last_updated_ts AS TIMESTAMP)), false) AS consent_phone,
                   NULL::DOUBLE AS bureau_score
            FROM {customers_csv} c
            JOIN _crm_id_map m ON m.crm_customer_id = c.crm_customer_id
            LEFT JOIN cases k ON k.golden_id = m.golden_id
            GROUP BY m.golden_id
        """)

        snapshot_glob = sql_path(data_dir / "account_monthly_snapshot" / "year=*" / "*.parquet")
        # Maple snapshots provide minimum_due and payment_made, not an overdue_amount field.
        con.execute(f"""
            CREATE TABLE accounts AS
            WITH source AS (
              SELECT account_id, crm_customer_id, product_type, account_status,
                     balance, minimum_due, payment_made, dpd, month_end_date,
                     coalesce(
                       try_strptime(month_end_date, '%Y-%m-%d'),
                       try_strptime(month_end_date, '%m/%d/%Y'),
                       try_strptime(month_end_date, '%d/%m/%Y')
                     )::DATE AS parsed_snapshot_date
              FROM read_parquet('{snapshot_glob}', hive_partitioning=true)
            )
            SELECT s.account_id,
                   m.golden_id,
                   lower(s.product_type) AS product_type,
                   s.balance::DOUBLE AS balance,
                   CASE WHEN coalesce(s.dpd, 0) > 0
                        THEN greatest(coalesce(s.minimum_due, 0) - coalesce(s.payment_made, 0), 0)
                        ELSE 0 END::DOUBLE AS overdue_amount,
                   coalesce(s.dpd, 0)::INTEGER AS days_past_due,
                   s.parsed_snapshot_date AS snapshot_date
            FROM source s
            JOIN _crm_id_map m ON m.crm_customer_id = s.crm_customer_id
            WHERE s.parsed_snapshot_date IS NOT NULL
        """)

        promises_csv = csv(data_dir / "promises_to_pay.csv")
        con.execute(f"""
            CREATE TABLE promises AS
            SELECT p.ptp_id,
                   p.case_id,
                   m.golden_id,
                   try_cast(p.ptp_created_ts AS TIMESTAMP)::DATE AS promised_date,
                   try_cast(p.status_ts AS TIMESTAMP)::DATE AS status_date,
                   try_cast(p.ptp_amount AS DOUBLE) AS promised_amount,
                   upper(trim(p.ptp_status)) AS status
            FROM {promises_csv} p
            JOIN _crm_id_map m ON m.crm_customer_id = p.crm_customer_id
            WHERE try_cast(p.ptp_created_ts AS TIMESTAMP) IS NOT NULL
        """)

        contacts_csv = csv(data_dir / "contact_history.csv")
        con.execute(f"""
            CREATE TABLE contacts AS
            SELECT c.contact_id,
                   c.case_id,
                   m.golden_id,
                   coalesce(nullif(trim(c.channel_v2), ''), lower(trim(c.channel))) AS channel,
                   try_cast(c.contact_ts_local AS TIMESTAMP)::DATE AS contact_date,
                   CASE
                     WHEN coalesce(try_cast(c.ptp_made_flag AS BOOLEAN), false) THEN 'promise_made'
                     WHEN coalesce(try_cast(c.rpc_flag AS BOOLEAN), false) THEN 'answered'
                     ELSE lower(trim(c.outcome_code))
                   END AS outcome,
                   nullif(trim(c.note_id), '') AS note_id
            FROM {contacts_csv} c
            JOIN _crm_id_map m ON m.crm_customer_id = c.crm_customer_id
            WHERE lower(trim(c.channel)) <> 'system'
              AND try_cast(c.contact_ts_local AS TIMESTAMP) IS NOT NULL
        """)

        notes_csv = csv(data_dir / "agent_notes.csv")
        email_pattern = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        phone_pattern = r"\+?\d[\d\s().-]{8,}\d"
        con.execute(f"""
            CREATE TABLE notes AS
            SELECT n.note_id,
                   m.golden_id,
                   regexp_replace(
                     regexp_replace(coalesce(n.note_text, ''), '{email_pattern}', '[EMAIL]', 'g'),
                     '{phone_pattern}', '[PHONE]', 'g'
                   ) AS note_text_masked
            FROM {notes_csv} n
            JOIN _crm_id_map m ON m.crm_customer_id = n.crm_customer_id
        """)

        con.execute("DROP TABLE _crm_id_map")
        con.execute("DROP TABLE _bureau_id_map")
        counts = {
            table: int(con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            for table in ("customers", "accounts", "cases", "promises", "contacts", "notes", "bureau_scores")
        }
        con.close()
        os.replace(staging, destination)
        return destination, counts
    except Exception:
        con.close()
        staging.unlink(missing_ok=True)
        raise
