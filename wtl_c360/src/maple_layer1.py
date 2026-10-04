"""DuckDB-first Layer 1 build for the full Maple release.

Only identity keys needed downstream are materialized. Original release files stay
unchanged on disk and the crosswalk retains the source key and match provenance.
"""
from __future__ import annotations

import os
import csv
import datetime as dt
import json
import re
import tempfile
from pathlib import Path

import duckdb
import yaml


SNAPSHOT_DATE = "2026-09-28"
SYSTEMS = ("crm", "card", "loan", "deposit", "bureau")


def _sql_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "''")


def _csv(path: Path) -> str:
    return f"read_csv('{_sql_path(path)}', header=true, all_varchar=true, sample_size=10000)"


def _masked_text_sql(
    text_expression: str, person_alias: str, email_pattern: str, phone_pattern: str
) -> str:
    masked = (
        f"regexp_replace(regexp_replace(coalesce({text_expression}, ''), "
        f"'{email_pattern}', '[EMAIL]', 'g'), '{phone_pattern}', '[PHONE]', 'g')"
    )
    for column in ("first_name", "last_name"):
        name = f"trim({person_alias}.{column})"
        masked = (
            f"CASE WHEN nullif({name}, '') IS NULL THEN {masked} "
            f"ELSE regexp_replace({masked}, '(?i)\\b' || regexp_escape({name}) || "
            f"'\\b', '[NAME]', 'g') END"
        )
    return masked


def _flatten_transcripts(transcript_dir: Path, output_path: Path) -> int:
    """Stream transcript JSON files to a narrow CSV to avoid loading the corpus into memory."""
    rows = 0
    with output_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow([
            "transcript_id", "contact_id", "case_id", "crm_customer_id",
            "agent_id", "call_start_ts", "turn_index", "turn_text",
        ])
        for path in sorted(transcript_dir.rglob("*.json")):
            with path.open(encoding="utf-8") as source:
                transcript = json.load(source)
            for turn in transcript.get("turns", []):
                writer.writerow([
                    transcript.get("transcript_id"), transcript.get("contact_id"),
                    transcript.get("case_id"), transcript.get("crm_customer_id"),
                    transcript.get("agent_id"), transcript.get("call_start_ts"),
                    turn.get("turn_index"), turn.get("text", ""),
                ])
                rows += 1
    return rows


def _flatten_policy_docs(
    reference_path: Path, docs_dir: Path, output_path: Path
) -> int:
    """Copy current, non-superseded policy material into searchable chunks."""
    rows = 0
    with reference_path.open(newline="", encoding="utf-8") as references, \
            output_path.open("w", newline="", encoding="utf-8") as output:
        reader = csv.DictReader(references)
        writer = csv.writer(output)
        writer.writerow([
            "chunk_id", "doc_id", "title", "version", "effective_date",
            "section", "source_path", "content",
        ])
        for document in reader:
            if document.get("superseded_by", "").strip():
                continue
            source_path = docs_dir / Path(document["file_path"])
            if not source_path.is_file():
                raise FileNotFoundError(f"Referenced policy document does not exist: {source_path}")
            section = "document"
            chunk_lines: list[str] = []
            chunk_number = 0

            def write_chunk() -> None:
                nonlocal rows, chunk_number, chunk_lines
                text = "\n".join(chunk_lines).strip()
                if text:
                    chunk_number += 1
                    writer.writerow([
                        f"{document['doc_id']}:{document['version']}:{chunk_number}",
                        document["doc_id"], document["title"], document["version"],
                        document["effective_date"], section, document["file_path"], text,
                    ])
                    rows += 1
                chunk_lines = []

            for line in source_path.read_text(encoding="utf-8").splitlines():
                if re.match(r"^#{1,6}\s", line):
                    write_chunk()
                    section = line.lstrip("#").strip()
                else:
                    chunk_lines.append(line)
                    if sum(len(item) for item in chunk_lines) > 1400:
                        write_chunk()
            write_chunk()
    return rows


def _create_bronze_views(con: duckdb.DuckDBPyConnection, data_dir: Path) -> int:
    """Expose every release table unchanged with source and stable load metadata."""
    con.execute("CREATE SCHEMA raw")
    loaded_at = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat(sep=" ")
    table_count = 0
    csv_sources = sorted(data_dir.glob("*.csv"))
    for path in csv_sources:
        name = re.sub(r"[^A-Za-z0-9_]", "_", path.stem)
        con.execute(f"""
            CREATE VIEW raw.{name} AS
            SELECT *, TIMESTAMP '{loaded_at}' AS _loaded_at,
                   '{_sql_path(path)}' AS _source_file
            FROM {_csv(path)}
        """)
        table_count += 1
    for directory in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        parquet_glob = directory / "year=*" / "*.parquet"
        if not list(directory.glob("year=*/*.parquet")):
            continue
        name = re.sub(r"[^A-Za-z0-9_]", "_", directory.name)
        con.execute(f"""
            CREATE VIEW raw.{name} AS
            SELECT *, TIMESTAMP '{loaded_at}' AS _loaded_at,
                   '{_sql_path(directory)}' AS _source_file
            FROM read_parquet('{_sql_path(parquet_glob)}', hive_partitioning=true)
        """)
        table_count += 1
    con.execute("CREATE TABLE bronze_source_registry(source_name VARCHAR, source_path VARCHAR, row_format VARCHAR, loaded_at TIMESTAMP)")
    for path in csv_sources:
        con.execute("INSERT INTO bronze_source_registry VALUES (?, ?, 'csv', ?)",
                    [path.stem, str(path), loaded_at])
    for directory in sorted(p for p in data_dir.iterdir() if p.is_dir() and list(p.glob("year=*/*.parquet"))):
        con.execute("INSERT INTO bronze_source_registry VALUES (?, ?, 'parquet', ?)",
                    [directory.name, str(directory), loaded_at])
    return table_count


def build_maple_layer1(data_dir: str | Path, c360_path: str | Path, l2_path: str | Path) -> dict[str, object]:
    """Build a reversible five-system crosswalk, curated C360 tables, DQ, and masked L2."""
    data_dir, c360_path, l2_path = Path(data_dir).resolve(), Path(c360_path).resolve(), Path(l2_path).resolve()
    required = [
        "customers.csv", "card_accounts.csv", "loan_accounts.csv", "deposit_accounts.csv",
        "collections_cases.csv", "contact_history.csv", "promises_to_pay.csv",
        "agent_notes.csv", "external.csv", "model_scores.csv", "offers.csv",
    ]
    required_dirs = ["account_monthly_snapshot", "bureau_history"]
    missing = [str(data_dir / name) for name in required if not (data_dir / name).is_file()]
    missing.extend(
        str(data_dir / name) for name in required_dirs if not (data_dir / name).is_dir()
    )
    if not (data_dir.parent / "files" / "transcripts").is_dir():
        missing.append(str(data_dir.parent / "files" / "transcripts"))
    if missing:
        raise FileNotFoundError("Missing Maple inputs: " + ", ".join(missing))

    c360_path.parent.mkdir(parents=True, exist_ok=True)
    l2_path.parent.mkdir(parents=True, exist_ok=True)
    c360_build = c360_path.with_name(c360_path.stem + ".building" + c360_path.suffix)
    l2_build = l2_path.with_name(l2_path.stem + ".building" + l2_path.suffix)
    c360_build.unlink(missing_ok=True)
    l2_build.unlink(missing_ok=True)
    Path(str(c360_build) + ".wal").unlink(missing_ok=True)
    Path(str(l2_build) + ".wal").unlink(missing_ok=True)
    transcript_fd, transcript_name = tempfile.mkstemp(
        prefix="maple_transcript_turns_", suffix=".csv", dir=c360_path.parent
    )
    os.close(transcript_fd)
    transcript_csv = Path(transcript_name)
    transcript_csv.unlink()
    documents_fd, documents_name = tempfile.mkstemp(
        prefix="maple_policy_chunks_", suffix=".csv", dir=c360_path.parent
    )
    os.close(documents_fd)
    documents_csv = Path(documents_name)
    documents_csv.unlink()
    con = duckdb.connect(str(c360_build))
    try:
        _flatten_transcripts(data_dir.parent / "files" / "transcripts", transcript_csv)
        _flatten_policy_docs(
            data_dir / "reference_documents.csv",
            data_dir.parent / "files",
            documents_csv,
        )
        bronze_tables = _create_bronze_views(con, data_dir)
        customers = _csv(data_dir / "customers.csv")
        con.execute(f"""
            CREATE TEMP TABLE _crm AS
            SELECT crm_customer_id AS source_id,
                   nullif(trim(national_id_hash), '') AS national_id_hash,
                   first_name, last_name, try_cast(date_of_birth AS DATE) AS dob,
                   lower(trim(email)) AS email,
                   regexp_replace(coalesce(primary_phone_e164, primary_phone_raw, ''), '[^0-9]', '', 'g') AS phone,
                   upper(regexp_replace(coalesce(postal_code, postal_code_raw, ''), '[^A-Za-z0-9]', '', 'g')) AS postal
            FROM {customers}
            WHERE nullif(trim(crm_customer_id), '') IS NOT NULL
        """)
        con.execute("""
            CREATE TEMP TABLE _crm_golden AS
            WITH identified AS (
              SELECT *,
                CASE WHEN national_id_hash IS NOT NULL THEN 'NID:' || national_id_hash
                     WHEN email IS NOT NULL AND dob IS NOT NULL
                       THEN 'EMAIL_DOB:' || email || ':' || cast(dob AS VARCHAR)
                     ELSE 'CRM:' || source_id END AS identity_key
              FROM _crm
            )
            SELECT source_id, national_id_hash, identity_key,
                   'G' || sha256(identity_key) AS golden_id
            FROM identified
        """)
        con.execute(f"""
            INSERT INTO _crm_golden
            SELECT orphan.source_id, NULL, 'ORPHAN_CRM:' || orphan.source_id,
                   'G' || sha256('ORPHAN_CRM:' || orphan.source_id)
            FROM (
              SELECT coll_customer_ref AS source_id FROM {_csv(data_dir / 'collections_cases.csv')}
              UNION SELECT crm_customer_id FROM {_csv(data_dir / 'contact_history.csv')}
              UNION SELECT crm_customer_id FROM {_csv(data_dir / 'promises_to_pay.csv')}
              UNION SELECT crm_customer_id FROM {_csv(data_dir / 'agent_notes.csv')}
              UNION SELECT crm_customer_id
                FROM read_parquet('{_sql_path(data_dir / 'account_monthly_snapshot' / 'year=*' / '*.parquet')}',
                                  hive_partitioning=true)
            ) orphan
            WHERE nullif(trim(orphan.source_id), '') IS NOT NULL
              AND NOT EXISTS (SELECT 1 FROM _crm_golden g WHERE g.source_id=orphan.source_id)
        """)

        # A source record is one distinct customer reference, not one account row.
        identity_sources = [
            ("card", "card_accounts.csv", "src_customer_ref", "cardholder_name_raw",
             "cardholder_dob_raw", "cardholder_phone_raw", "cardholder_postal_raw"),
            ("loan", "loan_accounts.csv", "src_customer_ref", "borrower_name_raw",
             "borrower_dob_raw", "borrower_phone_raw", "borrower_postal_raw"),
            ("deposit", "deposit_accounts.csv", "src_customer_ref", "holder_name_raw",
             "holder_dob_raw", "holder_phone_raw", "holder_postal_raw"),
        ]
        identity_selects = []
        for system, filename, key, name, dob, phone, postal in identity_sources:
            identity_selects.append(f"""
                SELECT '{system}' AS source_system, {key} AS source_id, {name} AS raw_name,
                       coalesce(try_cast({dob} AS DATE),
                                try_strptime({dob}, '%Y%m%d')::DATE,
                                try_strptime({dob}, '%Y/%m/%d')::DATE,
                                try_strptime({dob}, '%d-%m-%Y')::DATE) AS dob,
                       regexp_replace(coalesce({phone}, ''), '[^0-9]', '', 'g') AS phone,
                       upper(regexp_replace(coalesce({postal}, ''), '[^A-Za-z0-9]', '', 'g')) AS postal
                FROM {_csv(data_dir / filename)}
                WHERE nullif(trim({key}), '') IS NOT NULL
            """)
        identity_selects.append(f"""
            SELECT 'bureau' AS source_system, bank_subject_ref AS source_id,
                   CASE WHEN strpos(subject_name_reported, ',') > 0
                        THEN trim(split_part(subject_name_reported, ',', 2)) || ' ' ||
                             trim(split_part(subject_name_reported, ',', 1))
                        ELSE subject_name_reported END AS raw_name,
                   try_cast(subject_dob_reported AS DATE) AS dob,
                   '' AS phone, '' AS postal
            FROM {_csv(data_dir / 'external.csv')}
            WHERE nullif(trim(bank_subject_ref), '') IS NOT NULL
        """)
        con.execute("""
            CREATE TEMP TABLE _other_identity AS
            SELECT DISTINCT source_system, source_id, raw_name, dob, phone, postal
            FROM (""" + " UNION ALL ".join(identity_selects) + ")"
        )
        con.execute("""
            CREATE TEMP TABLE _name_parts AS
            SELECT *,
              lower(strip_accents(regexp_replace(trim(raw_name), '[^A-Za-zÀ-ÿ ]', ' ', 'g'))) AS clean_name,
              lower(strip_accents(regexp_replace(trim(raw_name), '[^A-Za-zÀ-ÿ]', '', 'g'))) AS compact_name
            FROM _other_identity
        """)
        con.execute("""
            CREATE TEMP TABLE _crm_names AS
            SELECT source_id,
                   lower(strip_accents(regexp_replace(trim(first_name || last_name),
                                                       '[^A-Za-zÀ-ÿ]', '', 'g'))) AS compact_name,
                   lower(strip_accents(regexp_replace(trim(first_name), '[^A-Za-zÀ-ÿ]', '', 'g'))) AS first_name_clean,
                   lower(strip_accents(regexp_replace(trim(last_name), '[^A-Za-zÀ-ÿ]', '', 'g'))) AS last_name_clean,
                   left(lower(strip_accents(regexp_replace(trim(first_name),
                                                           '[^A-Za-zÀ-ÿ]', '', 'g'))), 1) AS first_initial,
                   dob, phone, postal
            FROM _crm
        """)
        con.execute("""
            CREATE TABLE silver_identity AS
            SELECT source_system, source_id, clean_name, compact_name, dob,
                   nullif(right(phone, 10), '') AS phone10,
                   nullif(postal, '') AS postal
            FROM _name_parts
            UNION ALL
            SELECT 'crm', source_id,
                   lower(strip_accents(trim(first_name || ' ' || last_name))),
                   lower(strip_accents(regexp_replace(first_name || last_name, '[^A-Za-zÀ-ÿ]', '', 'g'))),
                   dob, nullif(right(phone, 10), ''), nullif(postal, '')
            FROM _crm
        """)
        con.execute("""
            CREATE TABLE quarantine AS
            SELECT source_system AS source_table, source_id AS record_id, 'identity' AS field,
                   coalesce(raw_name, '') AS raw_value,
                   CASE WHEN dob IS NULL THEN 'unparseable or missing DOB'
                        WHEN clean_name IS NULL OR clean_name = '' THEN 'missing name'
                        ELSE 'identity key lacks matching evidence' END AS reason
            FROM _name_parts
            WHERE dob IS NULL OR clean_name IS NULL OR clean_name=''
        """)
        con.execute("""
            CREATE TEMP TABLE _candidate_scores AS
            WITH candidates AS (
              SELECT s.source_system, s.source_id, s.raw_name, s.compact_name, s.dob, s.phone, s.postal,
                     s.clean_name, c.source_id AS crm_id, g.golden_id, c.compact_name AS crm_compact_name,
                     c.first_name_clean, c.last_name_clean, c.first_initial,
                     (length(s.phone) >= 10 AND right(s.phone, 10) = right(c.phone, 10))::INT AS phone_match,
                     (s.postal <> '' AND s.postal = c.postal)::INT AS postal_match,
                     (s.dob = c.dob)::INT AS dob_match
              FROM _name_parts s
              JOIN _crm_names c ON s.dob = c.dob
                AND s.compact_name <> ''
                AND (left(s.compact_name, 1) = left(c.compact_name, 1)
                     OR left(s.clean_name, 1) = left(c.last_name_clean, 1))
              JOIN _crm_golden g ON g.source_id = c.source_id
              WHERE s.source_system <> 'bureau' OR s.raw_name IS NOT NULL
              UNION
              SELECT s.source_system, s.source_id, s.raw_name, s.compact_name, s.dob, s.phone, s.postal,
                     s.clean_name, c.source_id AS crm_id, g.golden_id, c.compact_name AS crm_compact_name,
                     c.first_name_clean, c.last_name_clean, c.first_initial,
                     1 AS phone_match, (s.postal <> '' AND s.postal = c.postal)::INT AS postal_match,
                     1 AS dob_match
              FROM _name_parts s
              JOIN _crm_names c ON s.dob = c.dob
                AND length(s.phone) >= 10 AND right(s.phone, 10) = right(c.phone, 10)
              JOIN _crm_golden g ON g.source_id = c.source_id
              WHERE s.source_system <> 'bureau' OR s.raw_name IS NOT NULL
            ),
            name_scored AS (
              SELECT *,
                 greatest(
                   jaro_winkler_similarity(compact_name, crm_compact_name),
                   CASE WHEN last_name_clean <> '' AND first_initial <> ''
                          AND regexp_matches(clean_name,
                              '(^|[^a-z])' || last_name_clean || '([^a-z]|$)')
                          AND regexp_matches(clean_name,
                              '(^|[^a-z])' || first_initial || '([^a-z]|$)')
                        THEN 0.95 ELSE 0.0 END
                 ) AS name_score
              FROM candidates
            ),
            scored AS (
              SELECT *,
                 (0.55 * name_score
                   + 0.25 * dob_match + 0.15 * phone_match + 0.05 * postal_match)
                   / nullif(0.55 + 0.25 + (CASE WHEN length(phone) >= 10 THEN 0.15 ELSE 0 END)
                            + (CASE WHEN postal <> '' THEN 0.05 ELSE 0 END), 0) AS score
              FROM name_scored s
            )
            SELECT * FROM scored
            WHERE dob_match = 1 AND (name_score >= 0.72 OR phone_match = 1)
        """)
        con.execute("""
            CREATE TEMP TABLE _best_matches AS
            SELECT source_system, source_id, golden_id, score,
                   count(DISTINCT golden_id) OVER (PARTITION BY source_system, source_id) AS n_candidates,
                   row_number() OVER (PARTITION BY source_system, source_id ORDER BY score DESC, golden_id) AS rank
            FROM _candidate_scores
        """)
        con.execute("""
            CREATE TEMP TABLE _unmatched_keys AS
            SELECT source_system, source_id
            FROM _other_identity
            EXCEPT SELECT source_system, source_id FROM _best_matches
        """)
        con.execute("""
            CREATE TABLE id_crosswalk AS
            SELECT 'crm' AS source_system, g.source_id,
                   g.golden_id, CASE WHEN g.identity_key LIKE 'ORPHAN_CRM:%' THEN 0.0 ELSE 1.0 END AS match_confidence,
                   CASE WHEN g.identity_key LIKE 'NID:%' THEN 'exact_national_id'
                        WHEN g.identity_key LIKE 'EMAIL_DOB:%' THEN 'exact_email_dob'
                        WHEN g.identity_key LIKE 'ORPHAN_CRM:%' THEN 'unmatched_crm_orphan'
                        ELSE 'crm_anchor' END AS rule_used
            FROM _crm_golden g
            UNION ALL
            SELECT b.source_system, b.source_id,
                   CASE WHEN b.n_candidates = 1 AND b.score >= 0.92 THEN b.golden_id
                        ELSE 'G' || sha256(b.source_system || ':' || b.source_id) END AS golden_id,
                   CASE WHEN b.n_candidates = 1 AND b.score >= 0.92 THEN least(b.score, 1.0)
                        WHEN b.n_candidates > 1 OR b.score >= 0.75 THEN least(b.score, 1.0)
                        ELSE 0.0 END AS match_confidence,
                   CASE WHEN b.n_candidates = 1 AND b.score >= 0.92 THEN 'unique_high_confidence_composite'
                        WHEN b.n_candidates > 1 OR b.score >= 0.75 THEN 'steward_review'
                        ELSE 'unmatched_isolated' END AS rule_used
            FROM _best_matches b WHERE b.rank = 1
            UNION ALL
            SELECT u.source_system, u.source_id,
                   'G' || sha256(u.source_system || ':' || u.source_id) AS golden_id,
                   0.0::DOUBLE AS match_confidence, 'unmatched_isolated' AS rule_used
            FROM _unmatched_keys u
        """)
        con.execute("""
            CREATE TABLE steward_queue AS
            SELECT b.source_system, b.source_id, b.golden_id AS candidate_golden_id,
                   b.score AS match_confidence, 'pending' AS review_status
            FROM _best_matches b
            WHERE b.rank = 1 AND (b.n_candidates > 1 OR b.score >= 0.75 AND b.score < 0.92)
        """)
        con.execute("""
            CREATE TABLE match_candidates AS
            SELECT source_system, source_id, crm_id AS candidate_crm_id, golden_id,
                   name_score, dob_match, phone_match, postal_match, score, 'candidate' AS status
            FROM _candidate_scores
        """)
        con.execute(f"""
            CREATE TEMP TABLE _latest_bureau AS
            WITH linked_scores AS (
              SELECT x.golden_id, try_cast(b.pull_date AS DATE) AS score_date,
                     try_cast(b.risk_score AS DOUBLE) AS bureau_score
              FROM read_parquet('{_sql_path(data_dir / 'bureau_history' / 'year=*' / '*.parquet')}',
                                hive_partitioning=true) b
              JOIN id_crosswalk x ON x.source_system='bureau' AND x.source_id=b.bank_subject_ref
              WHERE try_cast(b.pull_date AS DATE) <= DATE '{SNAPSHOT_DATE}'
                AND try_cast(b.risk_score AS DOUBLE) IS NOT NULL
            ),
            latest_dates AS (
              SELECT golden_id, max(score_date) AS score_date
              FROM linked_scores GROUP BY golden_id
            )
            SELECT s.golden_id, avg(s.bureau_score) AS bureau_score
            FROM linked_scores s JOIN latest_dates d USING (golden_id, score_date)
            GROUP BY s.golden_id
        """)
        con.execute("""
            CREATE TEMP TABLE _crosswalk_profile AS
            SELECT golden_id,
                   count(DISTINCT source_system)
                     FILTER (WHERE source_system IN ('card', 'loan', 'deposit')) AS n_product_systems,
                   count(*) AS n_source_records
            FROM id_crosswalk
            GROUP BY golden_id
        """)

        # Curated, feature-safe outputs. Restricted identity attributes stay outside L2.
        con.execute(f"""
            CREATE TABLE c360_customer AS
            WITH profiled AS (
            SELECT g.golden_id,
                   arg_min(c.preferred_language, c.crm_customer_id) AS language,
                   arg_min(c.segment, c.crm_customer_id) AS segment,
                   false AS hardship_flag,
                   coalesce(arg_max(try_cast(c.consent_sms AS BOOLEAN),
                                    try_cast(c.consent_last_updated_ts AS TIMESTAMP)), false) AS consent_sms,
                   coalesce(arg_max(try_cast(c.consent_email AS BOOLEAN),
                                    try_cast(c.consent_last_updated_ts AS TIMESTAMP)), false) AS consent_email,
                   coalesce(arg_max(try_cast(c.consent_call AS BOOLEAN)
                                    OR try_cast(c.consent_call_mobile AS BOOLEAN),
                                    try_cast(c.consent_last_updated_ts AS TIMESTAMP)), false) AS consent_phone,
                   max(try_cast(c.consent_last_updated_ts AS TIMESTAMP)::DATE) AS consent_as_of_date,
                   max(b.bureau_score) AS bureau_score,
                   max(p.n_product_systems) AS n_product_systems,
                   max(p.n_source_records) AS n_source_records
            FROM {_csv(data_dir / 'customers.csv')} c
            JOIN id_crosswalk x ON x.source_system='crm' AND x.source_id=c.crm_customer_id
            JOIN _crm_golden g ON g.source_id=c.crm_customer_id
            LEFT JOIN _latest_bureau b ON b.golden_id=g.golden_id
            LEFT JOIN _crosswalk_profile p ON p.golden_id=g.golden_id
            GROUP BY g.golden_id
            )
            SELECT * FROM profiled
            UNION ALL
            SELECT x.golden_id, NULL, NULL, false, false, false, false, NULL, NULL::DOUBLE, 0, 1
            FROM id_crosswalk x
            WHERE x.source_system='crm' AND x.rule_used='unmatched_crm_orphan'
        """)
        con.execute(f"""
            CREATE TABLE c360_case AS
            SELECT c.case_id, x.golden_id, lower(trim(c.case_status)) AS status,
                   CASE WHEN try_cast(c.entry_dpd AS INTEGER) <= 30 THEN 'early'
                        WHEN try_cast(c.entry_dpd AS INTEGER) <= 90 THEN 'mid' ELSE 'late' END AS stage,
                   try_cast(c.case_open_date AS DATE) AS opened_date,
                   try_cast(c.case_status_date AS DATE) AS status_date,
                   lower(trim(c.primary_product)) AS product_type, c.assigned_agent_id AS agent_id,
                   coalesce(try_cast(c.hardship_flag AS BOOLEAN), false) AS hardship_flag,
                   coalesce(try_cast(c.hardship_plan_start AS DATE),
                            try_cast(c.case_open_date AS DATE)) AS hardship_start_date
            FROM {_csv(data_dir / 'collections_cases.csv')} c
            JOIN id_crosswalk x ON x.source_system='crm' AND x.source_id=c.coll_customer_ref
            WHERE try_cast(c.case_open_date AS DATE) IS NOT NULL
        """)
        con.execute("""
            UPDATE c360_customer AS cu
            SET hardship_flag = (
              SELECT coalesce(bool_or(c.hardship_flag), false)
              FROM c360_case c WHERE c.golden_id=cu.golden_id
            )
        """)
        con.execute(f"""
            CREATE TABLE c360_promise AS
            SELECT p.ptp_id, p.case_id, x.golden_id,
                   try_cast(p.ptp_created_ts AS TIMESTAMP)::DATE AS promised_date,
                   try_cast(p.status_ts AS TIMESTAMP)::DATE AS status_date,
                   try_cast(p.ptp_amount AS DOUBLE) AS promised_amount,
                   upper(trim(p.ptp_status)) AS status
            FROM {_csv(data_dir / 'promises_to_pay.csv')} p
            JOIN id_crosswalk x ON x.source_system='crm' AND x.source_id=p.crm_customer_id
            WHERE try_cast(p.ptp_created_ts AS TIMESTAMP) IS NOT NULL
        """)
        con.execute(f"""
            CREATE TABLE c360_contact AS
            SELECT c.contact_id, c.case_id, x.golden_id, c.agent_id,
                   coalesce(nullif(trim(c.channel_v2), ''), lower(trim(c.channel))) AS channel,
                   lower(trim(c.direction)) AS direction,
                   try_cast(c.contact_ts_local AS TIMESTAMP)::DATE AS contact_date,
                   CASE WHEN coalesce(try_cast(c.ptp_made_flag AS BOOLEAN), false) THEN 'promise_made'
                        WHEN coalesce(try_cast(c.rpc_flag AS BOOLEAN), false) THEN 'answered'
                        WHEN lower(trim(c.outcome_desc)) = 'no answer' THEN 'no_answer'
                        ELSE lower(trim(c.outcome_code)) END AS outcome,
                   nullif(trim(c.note_id), '') AS note_id
            FROM {_csv(data_dir / 'contact_history.csv')} c
            JOIN id_crosswalk x ON x.source_system='crm' AND x.source_id=c.crm_customer_id
            WHERE lower(trim(c.channel)) <> 'system'
              AND try_cast(c.contact_ts_local AS TIMESTAMP) IS NOT NULL
        """)
        email_rx = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        phone_rx = r"\+?\d[\d\s().-]{8,}\d"
        note_mask = _masked_text_sql("n.note_text", "c", email_rx, phone_rx)
        con.execute(f"""
            CREATE TABLE c360_note AS
            SELECT n.note_id, x.golden_id, {note_mask} AS note_text_masked
            FROM {_csv(data_dir / 'agent_notes.csv')} n
            JOIN id_crosswalk x ON x.source_system='crm' AND x.source_id=n.crm_customer_id
            LEFT JOIN _crm c ON c.source_id=n.crm_customer_id
        """)
        transcript_mask = _masked_text_sql("s.turn_text", "c", email_rx, phone_rx)
        con.execute(f"""
            CREATE TABLE c360_transcript AS
            WITH source_turns AS (
              SELECT transcript_id, contact_id, case_id, crm_customer_id,
                     agent_id, try_cast(call_start_ts AS TIMESTAMP) AS call_start_ts,
                     try_cast(turn_index AS INTEGER) AS turn_index, coalesce(turn_text, '') AS turn_text
              FROM {_csv(transcript_csv)}
            ),
            masked_turns AS (
              SELECT s.transcript_id, s.contact_id, s.case_id, x.golden_id, s.agent_id,
                       s.call_start_ts, s.turn_index, {transcript_mask} AS text_masked
              FROM source_turns s
              JOIN id_crosswalk x ON x.source_system='crm' AND x.source_id=s.crm_customer_id
              LEFT JOIN _crm c ON c.source_id=s.crm_customer_id
            )
            SELECT transcript_id, contact_id, case_id, golden_id, agent_id, call_start_ts,
                   string_agg(text_masked, ' ' ORDER BY turn_index) AS transcript_text_masked
            FROM masked_turns GROUP BY 1, 2, 3, 4, 5, 6
        """)
        con.execute(f"""
            CREATE TABLE c360_account AS
            WITH snapshots AS (
              SELECT account_id, crm_customer_id, product_type, balance, minimum_due,
                     payment_made, dpd, month_end_date,
                     coalesce(try_strptime(month_end_date, '%Y-%m-%d'),
                              try_strptime(month_end_date, '%m/%d/%Y'),
                              try_strptime(month_end_date, '%d/%m/%Y'),
                              try_strptime(month_end_date, '%Y%m%d'))::DATE AS snapshot_date
              FROM read_parquet('{_sql_path(data_dir / 'account_monthly_snapshot' / 'year=*' / '*.parquet')}',
                                hive_partitioning=true)
            )
            SELECT s.account_id, x.golden_id, lower(s.product_type) AS product_type,
                   s.balance::DOUBLE AS balance,
                   CASE WHEN coalesce(s.dpd, 0) > 0
                        THEN least(greatest(s.balance, 0),
                                   greatest(coalesce(s.minimum_due, 0) - coalesce(s.payment_made, 0), 0))
                        ELSE 0 END::DOUBLE AS overdue_amount,
                   coalesce(s.dpd, 0)::INTEGER AS days_past_due, s.snapshot_date
            FROM snapshots s JOIN id_crosswalk x ON x.source_system='crm' AND x.source_id=s.crm_customer_id
            WHERE s.snapshot_date IS NOT NULL
        """)
        con.execute("""
            CREATE TABLE c360_protected_restricted AS
            SELECT x.golden_id,
                   arg_min(c.gender_code, c.crm_customer_id) AS gender,
                   arg_min(c.marital_status, c.crm_customer_id) AS marital_status,
                   arg_min(c.citizenship_status, c.crm_customer_id) AS citizenship_status,
                   arg_min(c.newcomer_program_flag, c.crm_customer_id) AS newcomer,
                   arg_min(c.accessibility_needs_flag, c.crm_customer_id) AS accessibility,
                   arg_min(c.vulnerability_flag, c.crm_customer_id) AS vulnerability,
                   arg_min(c.province_code, c.crm_customer_id) AS location
            FROM id_crosswalk x JOIN """ + customers + """ c ON c.crm_customer_id=x.source_id
            WHERE x.source_system='crm'
            GROUP BY x.golden_id
        """)
        con.execute(f"""
            CREATE TABLE pipeline_run AS
            SELECT current_timestamp AS run_ts, DATE '{SNAPSHOT_DATE}' AS data_snapshot_date
        """)
        con.execute("CREATE TABLE bronze_catalog(table_name VARCHAR, source_path VARCHAR, source_kind VARCHAR, load_ts TIMESTAMP)")
        for filename in required:
            con.execute("INSERT INTO bronze_catalog VALUES (?, ?, 'csv', current_timestamp)", [Path(filename).stem, str(data_dir / filename)])
        for dirname in ("account_monthly_snapshot", "bureau_history"):
            con.execute("INSERT INTO bronze_catalog VALUES (?, ?, 'parquet', current_timestamp)", [dirname, str(data_dir / dirname)])

        con.execute("""
            CREATE TABLE id_crosswalk_rules AS
            SELECT source_system, count(*) AS records,
                   count(*) FILTER (WHERE rule_used='exact_national_id') AS exact_national_id,
                   count(*) FILTER (WHERE rule_used='unique_high_confidence_composite') AS high_confidence_composite,
                   count(*) FILTER (WHERE rule_used='steward_review') AS steward_review,
                   count(*) FILTER (WHERE rule_used='unmatched_isolated') AS unmatched
            FROM id_crosswalk GROUP BY source_system
        """)
        counts = {
            name: int(con.execute(f"SELECT count(*) FROM {table}").fetchone()[0])
            for name, table in (
                ("crosswalk", "id_crosswalk"), ("customers", "c360_customer"),
                ("accounts", "c360_account"), ("cases", "c360_case"),
                ("promises", "c360_promise"), ("contacts", "c360_contact"),
                ("notes", "c360_note"), ("transcripts", "c360_transcript"),
                ("steward_queue", "steward_queue"),
            )
        }
        counts["bronze_sources"] = bronze_tables
        source_counts = _source_row_counts(con, data_dir)
        con.execute(f"ATTACH '{_sql_path(l2_build)}' AS l2")
        con.execute("""
            CREATE TABLE l2.customers AS SELECT golden_id, language, segment, hardship_flag,
              consent_sms, consent_email, consent_phone, consent_as_of_date, bureau_score, n_product_systems
              FROM c360_customer
        """)
        for src, dst in (("c360_account", "accounts"), ("c360_case", "cases"),
                         ("c360_promise", "promises"), ("c360_contact", "contacts"),
                         ("c360_note", "notes")):
            con.execute(f"CREATE TABLE l2.{dst} AS SELECT * FROM {src}")
        con.execute("CREATE TABLE l2.transcripts AS SELECT * FROM c360_transcript")
        con.execute(f"CREATE TABLE l2.policy_chunks AS SELECT * FROM {_csv(documents_csv)}")
        con.execute("""
            CREATE TABLE l2.bureau_scores AS
            SELECT golden_id, try_cast(pull_date AS DATE) AS score_date,
                   try_cast(risk_score AS DOUBLE) AS bureau_score
            FROM read_parquet(?) b
            JOIN id_crosswalk x ON x.source_system='bureau' AND x.source_id=b.bank_subject_ref
            WHERE try_cast(pull_date AS DATE) IS NOT NULL AND try_cast(risk_score AS DOUBLE) IS NOT NULL
        """, [str(data_dir / "bureau_history" / "year=*" / "*.parquet")])
        con.execute("""
            CREATE TABLE l2.benchmark_promises AS
            SELECT lower(trim(p.ptp_status)) AS status,
                   try_cast(p.ptp_due_date AS DATE) AS due_date,
                   try_cast(p.prior_broken_ptp_count AS INTEGER) AS prior_broken_count,
                   coalesce(a.site, 'self_serve') AS agent_site,
                   c.test_cell, lower(trim(c.assignment_method)) AS assignment_method
            FROM raw.promises_to_pay p
            LEFT JOIN raw.collections_cases c ON c.case_id = p.case_id
            LEFT JOIN raw.agents a ON a.agent_id = p.agent_id
        """)
        con.execute("""
            CREATE TABLE l2.benchmark_cases AS
            SELECT try_cast(case_open_date AS DATE) AS opened_date,
                   try_cast(current_dpd AS INTEGER) AS current_dpd,
                   nullif(trim(queue), '') AS queue,
                   outcome IS NULL AS is_open,
                   CASE WHEN try_cast(current_dpd AS INTEGER) <= 0 THEN 'current'
                        WHEN try_cast(current_dpd AS INTEGER) <= 30 THEN '1-30'
                        WHEN try_cast(current_dpd AS INTEGER) <= 60 THEN '31-60'
                        WHEN try_cast(current_dpd AS INTEGER) <= 90 THEN '61-90'
                        WHEN try_cast(current_dpd AS INTEGER) <= 120 THEN '91-120'
                        WHEN try_cast(current_dpd AS INTEGER) <= 150 THEN '121-150'
                        WHEN try_cast(current_dpd AS INTEGER) <= 180 THEN '151-180'
                        ELSE '180+' END AS current_bucket,
                   lower(trim(primary_product)) AS primary_product,
                   coalesce(try_cast(hardship_flag AS BOOLEAN), false) AS hardship_flag,
                   try_cast(cure_flag AS BOOLEAN) AS cure_flag,
                   try_cast(days_to_cure AS INTEGER) AS days_to_cure
            FROM raw.collections_cases
        """)
        con.execute("""
            CREATE TABLE l2.benchmark_contacts AS
            SELECT coalesce(nullif(trim(channel_v2), ''), lower(trim(channel))) AS channel,
                   lower(trim(direction)) AS direction,
                   try_cast(contact_ts_local AS TIMESTAMP) AS contact_ts,
                   try_cast(contact_hour_local AS INTEGER) AS contact_hour_local,
                   coalesce(try_cast(rpc_flag AS BOOLEAN), false) AS rpc_flag,
                   coalesce(try_cast(dispute_raised_flag AS BOOLEAN), false) AS dispute_raised_flag,
                   coalesce(try_cast(within_permitted_hours_flag AS BOOLEAN), true) AS within_permitted_hours,
                   coalesce(try_cast(cease_request_flag AS BOOLEAN), false) AS cease_request_flag,
                   coalesce(try_cast(system_event_type AS VARCHAR), '') AS system_event_type
            FROM raw.contact_history
            WHERE try_cast(contact_ts_local AS TIMESTAMP) IS NOT NULL
        """)
        con.execute("""
            CREATE TABLE l2.benchmark_out_of_hours_by_province AS
            SELECT date_trunc('month', try_cast(h.contact_ts_local AS TIMESTAMP))::DATE AS month,
                   c.province_code,
                   round(100.0 * avg(CASE WHEN NOT coalesce(
                       try_cast(h.within_permitted_hours_flag AS BOOLEAN), true
                   ) THEN 1.0 ELSE 0.0 END), 2) AS out_of_hours_pct,
                   count(*) AS calls
            FROM raw.contact_history h
            JOIN raw.collections_cases c ON c.case_id = h.case_id
            WHERE coalesce(nullif(trim(h.channel_v2), ''), lower(trim(h.channel))) = 'call'
              AND lower(trim(h.direction)) = 'outbound'
              AND try_cast(h.contact_ts_local AS TIMESTAMP) IS NOT NULL
              AND coalesce(trim(h.system_event_type), '') = ''
            GROUP BY 1, c.province_code
        """)
        con.execute("""
            CREATE TABLE l2.benchmark_card_accounts AS
            SELECT try_cast(snapshot_date AS DATE) AS snapshot_date,
                   lower(trim(account_status)) AS account_status,
                   nullif(trim(product_code), '') AS product_code,
                   try_cast(dpd AS INTEGER) AS dpd
            FROM raw.card_accounts
            WHERE try_cast(snapshot_date AS DATE) IS NOT NULL
        """)
        con.execute("""
            CREATE TABLE l2.benchmark_account_history AS
            SELECT sha256(CAST(account_id AS VARCHAR)) AS account_key,
                   lower(trim(product_type)) AS product_type,
                   coalesce(try_cast(month_end_date AS DATE),
                            try_strptime(month_end_date, '%Y-%m-%d')::DATE,
                            try_strptime(month_end_date, '%d/%m/%Y')::DATE,
                            try_strptime(month_end_date, '%Y%m%d')::DATE) AS snapshot_date,
                   try_cast(dpd AS INTEGER) AS dpd
            FROM raw.account_monthly_snapshot
            WHERE coalesce(try_cast(month_end_date AS DATE),
                           try_strptime(month_end_date, '%Y-%m-%d')::DATE,
                           try_strptime(month_end_date, '%d/%m/%Y')::DATE,
                           try_strptime(month_end_date, '%Y%m%d')::DATE) IS NOT NULL
        """)
        con.execute("CREATE TABLE l2.pipeline_run AS SELECT * FROM pipeline_run")
        dq_rules = _run_dq(con, transcript_csv)
        dq_rules.extend(_run_data_quality_checks(con, data_dir))
        dq_rules.append(_run_l2_surface_check(con))
        con.execute("DETACH l2")
        con.close()
        transcript_csv.unlink(missing_ok=True)
        documents_csv.unlink(missing_ok=True)
        os.replace(c360_build, c360_path)
        os.replace(l2_build, l2_path)
        _write_reports(counts, dq_rules, c360_path.parent.parent / "submission", data_dir, source_counts)
        return {"counts": counts, "rules": dq_rules, "source_counts": source_counts}
    except Exception:
        con.close()
        c360_build.unlink(missing_ok=True)
        l2_build.unlink(missing_ok=True)
        transcript_csv.unlink(missing_ok=True)
        documents_csv.unlink(missing_ok=True)
        raise


def _run_dq(
    con: duckdb.DuckDBPyConnection, transcript_source: Path
) -> list[dict[str, object]]:
    checks = [
        ("unique_golden_id", "c360_customer", "count(*) - count(DISTINCT golden_id)", "golden_id unique"),
        ("crosswalk_one_golden", "id_crosswalk", "count(*) - count(DISTINCT source_system || ':' || source_id)", "one crosswalk row per source key"),
        ("account_linked", "c360_account", "count(*) FILTER (WHERE golden_id IS NULL)", "accounts link to a customer"),
        ("valid_dpd", "c360_account", "count(*) FILTER (WHERE days_past_due < 0)", "days past due is nonnegative"),
        ("valid_account_values", "c360_account", "count(*) FILTER (WHERE overdue_amount < 0 OR (balance >= 0 AND overdue_amount > balance))", "overdue estimates are nonnegative and do not exceed nonnegative balances; negative balances represent credits"),
        ("negative_balance_overdue", "c360_account", "count(*) FILTER (WHERE balance < 0 AND overdue_amount <> 0)", "credit/net-negative balances do not generate overdue estimates"),
        ("valid_case_dates", "c360_case", "count(*) FILTER (WHERE opened_date IS NULL)", "case open dates parse"),
        ("notes_linked", "c360_note", "count(*) FILTER (WHERE golden_id IS NULL)", "notes map to a customer"),
        ("transcripts_linked", "c360_transcript", "count(*) FILTER (WHERE golden_id IS NULL)", "transcripts map to a customer"),
        ("masked_note_contact_pii", "c360_note",
         "count(*) FILTER (WHERE regexp_matches(note_text_masked, '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\\\.[A-Za-z]{2,}|\\\\+?\\\\d[\\\\d\\\\s().-]{8,}\\\\d'))",
         "notes have email and phone patterns masked"),
        ("masked_note_customer_names", "c360_note",
         "count(*) FILTER (WHERE EXISTS (SELECT 1 FROM raw.agent_notes r JOIN _crm p ON p.source_id=r.crm_customer_id WHERE r.note_id=c360_note.note_id AND ((nullif(trim(p.first_name), '') IS NOT NULL AND regexp_matches(note_text_masked, '(?i)\\\\b' || regexp_escape(trim(p.first_name)) || '\\\\b')) OR (nullif(trim(p.last_name), '') IS NOT NULL AND regexp_matches(note_text_masked, '(?i)\\\\b' || regexp_escape(trim(p.last_name)) || '\\\\b')))))",
         "notes have the customer's first and last names masked"),
        ("masked_transcript_contact_pii", "c360_transcript",
         "count(*) FILTER (WHERE regexp_matches(transcript_text_masked, '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\\\.[A-Za-z]{2,}|\\\\+?\\\\d[\\\\d\\\\s().-]{8,}\\\\d'))",
         "transcripts have email and phone patterns masked"),
        ("crosswalk_confidence_range", "id_crosswalk", "count(*) FILTER (WHERE match_confidence < 0 OR match_confidence > 1)", "confidence in [0,1]"),
    ]
    results = []
    for check_id, table, failed_expr, description in checks:
        checked, failed = con.execute(f"SELECT count(*), {failed_expr} FROM {table}").fetchone()
        results.append({"id": check_id, "description": description, "checked": int(checked), "failed": int(failed)})
    transcript_names = con.execute(f"""
        SELECT count(DISTINCT t.transcript_id) FILTER (WHERE
          (nullif(trim(p.first_name), '') IS NOT NULL AND
           regexp_matches(t.transcript_text_masked, '(?i)\\\\b' || regexp_escape(trim(p.first_name)) || '\\\\b'))
          OR
          (nullif(trim(p.last_name), '') IS NOT NULL AND
           regexp_matches(t.transcript_text_masked, '(?i)\\\\b' || regexp_escape(trim(p.last_name)) || '\\\\b'))
        )
        FROM c360_transcript t
        JOIN (SELECT DISTINCT transcript_id, crm_customer_id FROM {_csv(transcript_source)}) s
          USING (transcript_id)
        JOIN _crm p ON p.source_id=s.crm_customer_id
    """).fetchone()[0]
    results.append({
        "id": "masked_transcript_customer_names",
        "description": "Transcripts have the customer's first and last names masked",
        "checked": int(con.execute("SELECT count(*) FROM c360_transcript").fetchone()[0]),
        "failed": int(transcript_names),
    })
    return results


def _run_data_quality_checks(
    con: duckdb.DuckDBPyConnection, data_dir: Path
) -> list[dict[str, object]]:
    checks = []
    schema = json.loads((data_dir.parent / "schema" / "schema.json").read_text(encoding="utf-8"))
    for name, metadata in schema.items():
        source_table = re.sub(r"[^A-Za-z0-9_]", "_", name)
        actual = int(con.execute(f"SELECT count(*) FROM raw.{source_table}").fetchone()[0])
        expected = int(metadata["rows"])
        checks.append({
            "id": f"source_row_count_{source_table}",
            "description": f"Source {name} row count agrees with release schema",
            "checked": expected,
            "failed": abs(actual - expected),
        })
    for source_table, source_file in (
        ("contact_history", "contact_history.csv"),
        ("agent_notes", "agent_notes.csv"),
        ("collections_cases", "collections_cases.csv"),
        ("promises_to_pay", "promises_to_pay.csv"),
    ):
        bronze_count, curated_count = con.execute(
            f"SELECT (SELECT count(*) FROM raw.{source_table}), (SELECT count(*) FROM c360_{'contact' if source_table == 'contact_history' else 'note' if source_table == 'agent_notes' else 'case' if source_table == 'collections_cases' else 'promise'})"
        ).fetchone()
        bronze_count, curated_count = int(bronze_count), int(curated_count)
        eligible_count = bronze_count
        if source_table == "contact_history":
            eligible_count = int(con.execute("""
                SELECT count(*) FROM raw.contact_history
                WHERE lower(trim(channel)) <> 'system'
                  AND try_cast(contact_ts_local AS TIMESTAMP) IS NOT NULL
                  AND nullif(trim(crm_customer_id), '') IS NOT NULL
            """).fetchone()[0])
        checks.append({
            "id": f"curated_coverage_{source_table}",
            "description": f"{source_file}: all eligible rows are curated and linked; expected exclusions are reported separately",
            "checked": eligible_count,
            "failed": max(eligible_count - curated_count, 0),
            "excluded": bronze_count - eligible_count,
        })
    snapshot_total, snapshot_curated, bad_dates = con.execute("""
        SELECT (SELECT count(*) FROM raw.account_monthly_snapshot),
               (SELECT count(*) FROM c360_account),
               (SELECT count(*) FROM raw.account_monthly_snapshot
                WHERE coalesce(try_strptime(month_end_date, '%Y-%m-%d'),
                               try_strptime(month_end_date, '%m/%d/%Y'),
                               try_strptime(month_end_date, '%d/%m/%Y'),
                               try_strptime(month_end_date, '%Y%m%d')) IS NULL)
    """).fetchone()
    checks.extend([
        {
            "id": "account_snapshot_dates_parsed",
            "description": "All account snapshot month-end dates parse using supported release formats",
            "checked": int(snapshot_total),
            "failed": int(bad_dates),
        },
        {
            "id": "account_snapshot_linked",
            "description": "All valid-date account snapshot rows link through a CRM crosswalk key",
            "checked": int(snapshot_total) - int(bad_dates),
            "failed": int(snapshot_total) - int(bad_dates) - int(snapshot_curated),
        },
    ])
    return checks


def _run_l2_surface_check(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    columns = con.execute("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_catalog = 'l2'
    """).fetchall()
    restricted = {
        "first_name", "middle_name", "last_name", "preferred_name", "full_name",
        "email", "phone", "primary_phone_e164", "primary_phone_raw", "dob",
        "date_of_birth", "postal_code", "postal_code_raw", "national_id_hash",
        "gender", "gender_code", "marital_status", "citizenship_status",
        "newcomer", "newcomer_program_flag", "accessibility", "accessibility_needs_flag",
        "vulnerability", "vulnerability_flag", "accent", "province_code",
    }
    leaked = [
        f"{table}.{column}" for table, column in columns
        if column.lower() in restricted
        and not (table == "benchmark_out_of_hours_by_province" and column.lower() == "province_code")
    ]
    return {
        "id": "l2_no_sensitive_columns",
        "description": "Masked Layer 2 contains no direct PII, national-ID hashes, or protected attributes outside the pre-aggregated province compliance report",
        "checked": len(columns),
        "failed": len(leaked),
        "leaked_columns": leaked,
    }


def _source_row_counts(con: duckdb.DuckDBPyConnection, data_dir: Path) -> list[dict[str, object]]:
    schema_path = data_dir.parent / "schema" / "schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    results = []
    for name, metadata in schema.items():
        source_name = re.sub(r"[^A-Za-z0-9_]", "_", name)
        actual = int(con.execute(f"SELECT count(*) FROM raw.{source_name}").fetchone()[0])
        expected = int(metadata["rows"])
        results.append({
            "table": name, "expected": expected, "actual": actual,
            "matches": expected == actual,
        })
    return results


def _write_reports(
    counts: dict[str, int], rules: list[dict[str, object]], submission: Path,
    data_dir: Path, source_counts: list[dict[str, object]],
) -> None:
    submission.mkdir(parents=True, exist_ok=True)
    report = [
        "# Full Maple Layer 1 quality report", "",
        f"Snapshot date: {SNAPSHOT_DATE}. Counts are computed from the current DuckDB build; no source files were edited.", "",
        "## Curated output counts", "", "| Table | Rows |", "|---|---:|",
    ]
    report.extend(f"| {key} | {value:,} |" for key, value in counts.items())
    report.extend(["", "## Source row-count reconciliation", "",
                   "| Source | Schema rows | Observed rows | Result |", "|---|---:|---:|---|"])
    report.extend(
        f"| {r['table']} | {r['expected']:,} | {r['actual']:,} | {'PASS' if r['matches'] else 'MISMATCH'} |"
        for r in source_counts
    )
    manifest_rows = []
    manifest_path = data_dir.parent / "MANIFEST.csv"
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        for row in csv.DictReader(manifest_file):
            target = data_dir.parent / Path(row["path"])
            actual_bytes = target.stat().st_size if target.is_file() else None
            manifest_rows.append({
                "path": row["path"], "expected_bytes": int(row["bytes"]),
                "actual_bytes": actual_bytes,
                "matches": actual_bytes == int(row["bytes"]),
            })
    report.extend(["", "## Release manifest size reconciliation", "",
                   f"Verified {sum(r['matches'] for r in manifest_rows):,}/{len(manifest_rows):,} entries by file presence and byte size. SHA-256 values are not recomputed."])
    schema = json.loads((data_dir.parent / "schema" / "schema.json").read_text(encoding="utf-8"))
    sensitive_terms = (
        "name", "email", "phone", "address", "postal", "date_of_birth",
        "national_id", "sin", "dob", "gender", "marital", "citizenship",
        "newcomer", "accessibility", "vulnerability", "accent", "province",
    )
    with (submission / "sensitivity_inventory.csv").open(
        "w", newline="", encoding="utf-8"
    ) as sensitivity_file:
        writer = csv.DictWriter(sensitivity_file, fieldnames=[
            "source_table", "column", "source_sensitivity", "classification",
            "l2_policy", "description",
        ])
        writer.writeheader()
        for table_name, table in schema.items():
            for column in table.get("columns", []):
                name = column["name"]
                declared = column.get("sensitivity") or ""
                lowered = name.lower()
                if declared:
                    classification = declared
                elif any(term in lowered for term in sensitive_terms):
                    classification = "Direct PII or protected/quasi-identifier"
                elif lowered.endswith(("_id", "_ref", "_uuid", "_hash")):
                    classification = "Direct or pseudonymous identifier"
                else:
                    classification = "Business data; sensitivity not declared"
                writer.writerow({
                    "source_table": table_name,
                    "column": name,
                    "source_sensitivity": declared,
                    "classification": classification,
                    "l2_policy": "deny by default; only contract-listed curated columns are published",
                    "description": column.get("description", ""),
                })
    report.extend(["", "## Automated checks", "", "| Check | Description | Checked | Excluded | Failed | Pass rate | Result |", "|---|---|---:|---:|---:|---:|---|"])
    for r in rules:
        rate = (r["checked"] - r["failed"]) / r["checked"] if r["checked"] else 1.0
        report.append(f"| {r['id']} | {r['description']} | {r['checked']:,} | {r.get('excluded', 0):,} | {r['failed']:,} | {rate:.2%} | {'PASS' if rate >= 0.99 else 'FAIL'} |")
    report.extend([
        "", "Pass target: at least 99% of checked rows must satisfy each rule. Exclusions are rows outside a check's eligible input, not hidden failures.",
        "Negative account balances are retained as valid net-credit balances; the estimated overdue amount is zero on those rows.",
    ])
    report.extend(["", "Identity scores are rule-based candidate scores, not calibrated match probabilities; no labeled Maple identity ground truth was supplied."])
    report.extend(["", "The steward queue preserves ambiguous candidates for review; unmatched identifiers remain isolated rather than being force-linked."])
    (submission / "dq_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (submission / "dq_results.json").write_text(json.dumps({
        "minimum_pass_rate": 0.99,
        "checks": [
            {**r, "pass_rate": (r["checked"] - r["failed"]) / r["checked"] if r["checked"] else 1.0,
             "passed": (r["checked"] - r["failed"]) / r["checked"] >= 0.99 if r["checked"] else True}
            for r in rules
        ],
        "source_row_counts": source_counts,
        "manifest_size_reconciliation": {
            "checked": len(manifest_rows),
            "passed": sum(r["matches"] for r in manifest_rows),
            "missing_or_size_mismatch": [r for r in manifest_rows if not r["matches"]],
            "sha256_recomputed": False,
        },
    }, indent=2), encoding="utf-8")
    contract = {
        "contract_id": "DC-COLL-001-golden-c360",
        "owner": "Collections Data Office",
        "source_contract": "data/files/docs/data_contracts/DC-COLL-001_v1.0_EN.md",
        "refresh": {"cadence": "daily", "source_snapshot_date": SNAPSHOT_DATE},
        "retention": "Apply source-system retention schedules; retain crosswalk decisions and audit evidence per policy.",
        "allowed_uses": ["collections operations", "analytics", "model training with approved feature controls"],
        "prohibited_uses": ["marketing", "protected-attribute or proxy-based decisions", "unmasked PII to an LLM"],
        "schema": {
            "customers": ["golden_id", "language", "segment", "hardship_flag", "consent_sms", "consent_email", "consent_phone", "bureau_score"],
            "accounts": ["account_id", "golden_id", "product_type", "balance", "overdue_amount", "days_past_due", "snapshot_date"],
            "cases": ["case_id", "golden_id", "status", "stage", "opened_date", "product_type"],
            "promises": ["ptp_id", "case_id", "golden_id", "promised_date", "status"],
            "contacts": ["contact_id", "case_id", "golden_id", "channel", "contact_date", "outcome", "note_id"],
            "notes": ["note_id", "golden_id", "note_text_masked"],
            "transcripts": ["transcript_id", "contact_id", "case_id", "golden_id", "agent_id", "call_start_ts", "transcript_text_masked"],
            "bureau_scores": ["golden_id", "score_date", "bureau_score"],
            "policy_chunks": ["chunk_id", "doc_id", "title", "version", "effective_date", "section", "source_path", "content"],
            "crosswalk": ["source_system", "source_id", "golden_id", "match_confidence", "rule_used"],
        },
        "sensitivity_inventory": "sensitivity_inventory.csv; all source columns are deny-by-default",
        "data_semantics": {
            "overdue_amount": "Estimate: max(minimum_due - payment_made, 0) for DPD > 0, capped at max(balance, 0); not a source field.",
            "negative_balance": "Retained as a valid net-credit balance; estimated overdue is zero.",
            "contact_exclusions": "System events, invalid timestamps, and empty CRM references are excluded from curated agent contacts and counted in DQ exclusions.",
        },
        "known_limitations": [
            "Cross-system identity candidates use a custom DOB/name/phone/postal blocking and score; scores are not calibrated probabilities.",
            "Ambiguous cross-system candidates are isolated pending steward review; the observed review queue is large.",
            "The synthetic release has no labeled identity ground truth; match precision/recall cannot be measured.",
            "Text privacy masking is regex-based and should be reviewed before production use.",
        ],
        "quality_rules": [
            {"id": r["id"], "check": r["id"], "description": r["description"], "target": r["checked"], "minimum_pass_rate": 0.99}
            for r in rules
        ],
    }
    (submission / "data_contract_golden_c360.yaml").write_text(
        yaml.safe_dump(contract, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
