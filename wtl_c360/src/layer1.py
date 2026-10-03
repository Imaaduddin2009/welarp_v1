"""LAYER 1 - Data Product Factory: raw(bronze) -> curated(silver) -> match -> golden C360 -> DQ -> masked L2 db."""
import sys, json, time, duckdb, numpy as np, pandas as pd
from pathlib import Path
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
from schema_map import *

DB, L2DB = "out/c360.duckdb", "out/l2.duckdb"
def q(con, sql): return con.execute(sql).fetchdf()

# ---------------------------------------------------------------- 1. BRONZE (raw, never edited)
def bronze(con):
    for f in sorted(Path(RAW_DIR).glob("*.csv")):
        if f.name.startswith("_"): continue          # skip ground-truth files used only for evaluating the mock
        con.execute(f"""CREATE OR REPLACE TABLE raw_{f.stem} AS
            SELECT *, current_timestamp AS _loaded_at, '{f.name}' AS _source_file
            FROM read_csv('{f}', all_varchar=true, header=true)""")   # all_varchar: keeps leading zeros, no lossy typing

# ---------------------------------------------------------------- 2. SILVER (standardised; bad values set aside, raw kept)
def silver(con):
    parts = []
    for sysname, m in IDENTITY.items():
        c = lambda k: m[k] if m[k] else None
        if m["name"]:
            n = m["name"]
            first = f"CASE WHEN {n} LIKE '%,%' THEN trim(split_part({n}, ',', 2)) ELSE trim(split_part({n}, ' ', 1)) END"
            last  = f"CASE WHEN {n} LIKE '%,%' THEN trim(split_part({n}, ',', 1)) ELSE trim(string_split({n}, ' ')[-1]) END"
        else: first, last = m["first"], m["last"]
        parts.append(f"""SELECT '{sysname}' AS source_system, {m['id']} AS source_id, {first} AS first_raw, {last} AS last_raw,
            {m['dob']} AS dob_raw, {m['email']} AS email_raw, {m['phone']} AS phone_raw, {m['postal']} AS postal_raw, {m['nid']} AS nid_raw
            FROM raw_{m['table']}""")
    con.execute("CREATE OR REPLACE TABLE _base AS " + " UNION ALL ".join(parts))
    con.execute(f"""CREATE OR REPLACE TABLE silver_identity AS
    WITH p AS (SELECT *,
        COALESCE(try_strptime(dob_raw,'%Y-%m-%d'), try_strptime(dob_raw,'%d/%m/%Y'),
                 try_strptime(dob_raw,'%d %b %Y'), try_strptime(dob_raw,'%Y%m%d'))::DATE AS dob_p,
        regexp_replace(COALESCE(phone_raw,''),'[^0-9]','','g') AS ph_digits,
        upper(regexp_replace(COALESCE(postal_raw,''),'[^A-Za-z0-9]','','g')) AS zip_c FROM _base)
    SELECT source_system || ':' || source_id AS key, source_system, source_id,
        nullif(regexp_replace(lower(strip_accents(first_raw)),'[^a-z]','','g'),'') AS first_n,
        nullif(regexp_replace(lower(strip_accents(last_raw)),'[^a-z]','','g'),'')  AS last_n,
        CASE WHEN dob_p BETWEEN DATE '1900-01-01' AND DATE '{SNAPSHOT_DATE}' - INTERVAL 18 YEAR THEN dob_p END AS dob,
        CASE WHEN length(ph_digits) >= 10 THEN right(ph_digits, 10) END AS phone10,
        CASE WHEN regexp_matches(zip_c, '^[ABCEGHJKLMNPRSTVXY][0-9][A-Z][0-9][A-Z][0-9]$') THEN zip_c END AS postal,
        CASE WHEN email_raw LIKE '%@%' THEN lower(trim(email_raw)) END AS email,
        nullif(trim(nid_raw),'') AS nid,
        dob_raw, phone_raw, postal_raw
    FROM p""")
    # quarantine: raw value present but failed validation -> set aside with reason, never deleted
    con.execute("""CREATE OR REPLACE TABLE quarantine (source_table VARCHAR, record_id VARCHAR, field VARCHAR, raw_value VARCHAR, reason VARCHAR)""")
    for fld, raw, norm in [("dob","dob_raw","dob"),("phone","phone_raw","phone10"),("postal_code","postal_raw","postal")]:
        con.execute(f"""INSERT INTO quarantine SELECT source_system, source_id, '{fld}', {raw}, 'failed validation'
                        FROM silver_identity WHERE {raw} IS NOT NULL AND {norm} IS NULL""")

# ---------------------------------------------------------------- 3. ID MATCHING (tiered; precision over recall)
def match(con):
    # tier 1 - exact on a strong identifier => confidence 1.0
    con.execute("""CREATE OR REPLACE TABLE edges_t1 AS
      SELECT a.key AS a, b.key AS b, 'exact_national_id' AS rule, 1.0 AS score FROM silver_identity a JOIN silver_identity b ON a.nid=b.nid AND a.key<b.key
      UNION ALL
      SELECT a.key, b.key, 'exact_email_dob', 1.0 FROM silver_identity a JOIN silver_identity b
        ON a.email=b.email AND a.dob=b.dob AND a.key<b.key""")
    # tier 2 - blocking, then weighted similarity over the fields that are present (Splink-style, hand-rolled)
    con.execute(f"""CREATE OR REPLACE TABLE fuzzy_pairs AS
    WITH cand AS (
      SELECT a.key ka, b.key kb FROM silver_identity a JOIN silver_identity b ON a.dob=b.dob AND a.key<b.key
      UNION SELECT a.key, b.key FROM silver_identity a JOIN silver_identity b ON a.phone10=b.phone10 AND a.key<b.key
      UNION SELECT a.key, b.key FROM silver_identity a JOIN silver_identity b
            ON a.postal=b.postal AND left(a.last_n,3)=left(b.last_n,3) AND a.key<b.key),
    s AS (SELECT ka, kb,
      CASE WHEN a.first_n IS NOT NULL AND b.first_n IS NOT NULL AND a.last_n IS NOT NULL AND b.last_n IS NOT NULL
           THEN (jaro_winkler_similarity(a.first_n,b.first_n)+jaro_winkler_similarity(a.last_n,b.last_n))/2 END AS s_name,
      CASE WHEN a.dob IS NOT NULL AND b.dob IS NOT NULL THEN (a.dob=b.dob)::INT END AS s_dob,
      CASE WHEN a.phone10 IS NOT NULL AND b.phone10 IS NOT NULL THEN (a.phone10=b.phone10)::INT END AS s_phone,
      CASE WHEN a.postal IS NOT NULL AND b.postal IS NOT NULL THEN (a.postal=b.postal)::INT END AS s_postal
      FROM cand JOIN silver_identity a ON a.key=ka JOIN silver_identity b ON b.key=kb),
    t AS (SELECT *, (COALESCE(.35*s_name,0)+COALESCE(.30*s_dob,0)+COALESCE(.20*s_phone,0)+COALESCE(.15*s_postal,0)) /
         NULLIF((s_name IS NOT NULL)::INT*.35+(s_dob IS NOT NULL)::INT*.30+(s_phone IS NOT NULL)::INT*.20+(s_postal IS NOT NULL)::INT*.15,0) AS score,
         (s_name IS NOT NULL)::INT+(s_dob IS NOT NULL)::INT+(s_phone IS NOT NULL)::INT+(s_postal IS NOT NULL)::INT AS n_fields FROM s)
    SELECT *, CASE WHEN score>={AUTO_SCORE} AND n_fields>=3 AND s_name>=.85 AND (s_dob=1 OR s_phone=1) THEN 'auto'
                   WHEN score>={STEWARD_SCORE} THEN 'steward' ELSE 'reject' END AS decision FROM t""")
    keys = q(con, "SELECT key, nid, source_system FROM silver_identity")
    idx = {k: i for i, k in enumerate(keys.key)}
    t1 = q(con, "SELECT * FROM edges_t1")
    fz = q(con, "SELECT ka AS a, kb AS b, 'fuzzy_auto' AS rule, score FROM fuzzy_pairs WHERE decision='auto'")
    def comps(edges):
        r = edges.a.map(idx).values; c = edges.b.map(idx).values
        g = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(idx), len(idx)))
        return connected_components(g, directed=False)[1]
    allE = pd.concat([t1, fz], ignore_index=True)
    keys["comp"] = comps(allE)
    # safety net: a cluster with two different national-id hashes, or oversized, is a bad merge -> fall back to exact-only edges
    stats = keys.groupby("comp").agg(n=("key", "size"), n_nid=("nid", "nunique"))
    bad = set(stats[(stats.n_nid > 1) | (stats.n > 8)].index)
    n_conflict = len(bad)
    if bad:
        keep_fz = fz[~fz.a.map(idx).map(lambda i: keys.comp[i]).isin(bad)]
        allE = pd.concat([t1, keep_fz], ignore_index=True); keys["comp"] = comps(allE)
    # golden_id: anchor on lowest CRM id in the cluster, else a hash-based id for product-only clusters
    def gid(g):
        crm = sorted(k for k in g if k.startswith("crm:"))
        return "G" + crm[0][4:] if crm else "GX" + __import__("hashlib").md5(sorted(g)[0].encode()).hexdigest()[:10]
    keys["golden_id"] = keys.groupby("comp").key.transform(lambda s: gid(list(s)))
    # confidence per record = best retained edge touching it (anchor = 1.0)
    best = {}
    for e in allE.itertuples():
        for k in (e.a, e.b):
            if e.score > best.get(k, (0, ""))[0] or (e.score == best.get(k, (0, ""))[0] and e.rule.startswith("exact")): best[k] = (e.score, e.rule)
    anchors = set("crm:" + g[1:] for g in keys.golden_id if g.startswith("G") and not g.startswith("GX"))
    keys["match_confidence"] = [1.0 if k in anchors else best.get(k, (1.0, "singleton"))[0] for k in keys.key]
    keys["rule_used"] = ["anchor" if k in anchors else best.get(k, (1.0, "singleton"))[1] for k in keys.key]
    xw = keys.merge(q(con, "SELECT key, source_id FROM silver_identity"), on="key")[["source_system", "source_id", "golden_id", "match_confidence", "rule_used"]]
    con.register("xw_df", xw); con.execute("CREATE OR REPLACE TABLE id_crosswalk AS SELECT * FROM xw_df")
    # steward queue: uncertain pairs that did not end up merged anyway
    con.execute("""CREATE OR REPLACE TABLE steward_queue AS
      SELECT f.ka, f.kb, round(f.score,3) AS score, f.s_name, f.s_dob, f.s_phone, f.s_postal, 'pending' AS review_status
      FROM fuzzy_pairs f JOIN id_crosswalk x ON x.source_system||':'||x.source_id=f.ka JOIN id_crosswalk y ON y.source_system||':'||y.source_id=f.kb
      WHERE f.decision='steward' AND x.golden_id<>y.golden_id""")
    return dict(conflict_clusters_split=n_conflict)

# ---------------------------------------------------------------- 4. GOLDEN C360
def golden(con):
    bs = BUREAU_SCORES
    con.execute(f"""CREATE OR REPLACE TABLE c360_customer AS
    WITH crm AS (
      SELECT x.golden_id, r.*, row_number() OVER (PARTITION BY x.golden_id ORDER BY r.crm_customer_id) AS rn
      FROM raw_crm_customers r JOIN id_crosswalk x ON x.source_system='crm' AND x.source_id=r.crm_customer_id),
    bur AS (SELECT x.golden_id, max(try_cast(s.{bs[2]} AS INT)) AS bureau_score
            FROM raw_{bs[0]} s JOIN id_crosswalk x ON x.source_system='bureau' AND x.source_id=s.{bs[1]} GROUP BY 1),
    prod AS (SELECT golden_id, count(DISTINCT source_system) FILTER (WHERE source_system IN ('card','loan','deposit')) AS n_product_systems,
             min(match_confidence) AS min_match_confidence, count(*) AS n_source_records FROM id_crosswalk GROUP BY 1)
    SELECT c.golden_id, c.crm_customer_id AS survivor_crm_id, c.first_name, c.last_name, s.dob, s.postal AS postal_code, s.phone10 AS phone, s.email,
        c.language, c.segment, c.hardship_flag::BOOLEAN AS hardship_flag, c.consent_sms::BOOLEAN AS consent_sms,
        c.consent_email::BOOLEAN AS consent_email, c.consent_phone::BOOLEAN AS consent_phone,
        b.bureau_score, CASE WHEN b.bureau_score IS NULL THEN 'unknown' WHEN b.bureau_score<580 THEN 'poor' WHEN b.bureau_score<670 THEN 'fair'
          WHEN b.bureau_score<740 THEN 'good' ELSE 'excellent' END AS bureau_band,
        p.n_product_systems, p.n_source_records, p.min_match_confidence
    FROM crm c JOIN silver_identity s ON s.key='crm:'||c.crm_customer_id
    LEFT JOIN bur b USING (golden_id) LEFT JOIN prod p USING (golden_id) WHERE c.rn=1""")
    # protected attributes are parked in a restricted table: fairness audits only, never joined into L2/L3
    con.execute("""CREATE OR REPLACE TABLE c360_protected_restricted AS
      SELECT golden_id, gender, marital_status FROM (SELECT x.golden_id, r.gender, r.marital_status,
      row_number() OVER (PARTITION BY x.golden_id ORDER BY r.crm_customer_id) rn FROM raw_crm_customers r
      JOIN id_crosswalk x ON x.source_system='crm' AND x.source_id=r.crm_customer_id) WHERE rn=1""")
    # accounts: de-duplicate, map to golden_id, set aside impossible values
    unions = " UNION ALL ".join(f"SELECT '{s}' AS sys, * FROM (SELECT DISTINCT * EXCLUDE (_loaded_at,_source_file) FROM raw_{t})" for s, (t, _) in ACCOUNTS.items())
    con.execute(f"CREATE OR REPLACE TABLE _acc AS {unions}")
    con.execute("""CREATE OR REPLACE TABLE _acc_mapped AS
      SELECT a.account_id, x.golden_id, a.product_type, try_cast(a.balance AS DOUBLE) AS balance, try_cast(a.overdue_amount AS DOUBLE) AS overdue_amount,
             try_cast(a.days_past_due AS INT) AS days_past_due, try_cast(a.snapshot_date AS DATE) AS snapshot_date
      FROM _acc a LEFT JOIN id_crosswalk x ON x.source_system=a.sys AND x.source_id=a.src_customer_ref""")
    con.execute("""INSERT INTO quarantine SELECT 'accounts', account_id, 'days_past_due/overdue/golden_id', concat_ws('|',days_past_due,overdue_amount,balance),
        CASE WHEN days_past_due<0 THEN 'negative days_past_due' WHEN overdue_amount<0 THEN 'negative overdue'
             WHEN overdue_amount>balance THEN 'overdue exceeds balance' ELSE 'no golden_id (unlinked customer)' END
      FROM _acc_mapped WHERE days_past_due<0 OR overdue_amount<0 OR overdue_amount>balance OR golden_id IS NULL
        OR golden_id NOT IN (SELECT golden_id FROM c360_customer)""")
    con.execute("""CREATE OR REPLACE TABLE c360_account AS SELECT * FROM _acc_mapped WHERE golden_id IN (SELECT golden_id FROM c360_customer)
        AND days_past_due>=0 AND overdue_amount>=0 AND overdue_amount<=balance""")
    # collections activity is keyed by CRM id -> route through the crosswalk so duplicates collapse onto one golden_id
    xw = "JOIN id_crosswalk x ON x.source_system='crm' AND x.source_id=t.crm_customer_id"
    con.execute(f"CREATE OR REPLACE TABLE c360_case AS SELECT t.case_id, x.golden_id, t.status, t.stage, t.agent_id, try_cast(t.opened_date AS DATE) AS opened_date, try_cast(t.opened_date AS DATE) AS status_date, t.product_type, cu.hardship_flag, try_cast(t.opened_date AS DATE) AS hardship_start_date FROM raw_collections_cases t {xw} JOIN c360_customer cu ON cu.golden_id=x.golden_id")
    con.execute(f"CREATE OR REPLACE TABLE c360_promise AS SELECT t.ptp_id, t.case_id, x.golden_id, try_cast(t.promised_date AS DATE) AS promised_date, try_cast(t.promised_date AS DATE) AS status_date, try_cast(t.promised_amount AS DOUBLE) AS promised_amount, upper(t.status) AS status FROM raw_promises_to_pay t {xw}")
    con.execute(f"CREATE OR REPLACE TABLE c360_contact AS SELECT t.contact_id, t.case_id, x.golden_id, t.agent_id, t.channel, try_cast(t.contact_ts AS DATE) AS contact_date, t.outcome, t.note_id FROM raw_contact_history t {xw}")
    con.execute(f"CREATE OR REPLACE TABLE c360_note AS SELECT t.note_id, x.golden_id, t.note_text FROM raw_agent_notes t {xw}")
    con.execute(f"CREATE OR REPLACE TABLE pipeline_run AS SELECT current_timestamp AS run_ts, DATE '{SNAPSHOT_DATE}' AS data_snapshot_date")

# ---------------------------------------------------------------- 5. DATA QUALITY (rules mirror the data contract)
def dq(con):
    one = lambda s: int(con.execute(s).fetchone()[0])
    R = []
    def rule(i, desc, checked, failed, action): R.append(dict(rule=i, description=desc, checked=checked, failed=failed, status="PASS" if failed == 0 else ("WARN" if action.startswith("set") else "FAIL"), on_fail=action))
    n = one("SELECT count(*) FROM c360_customer")
    rule(1, "golden_id present and unique in c360_customer", n, n - one("SELECT count(DISTINCT golden_id) FROM c360_customer WHERE golden_id IS NOT NULL"), "block publish")
    tot = one("SELECT count(*) FROM id_crosswalk")
    rule(2, "every source record maps to exactly one golden_id", tot, tot - one("SELECT count(*) FROM (SELECT source_system, source_id FROM id_crosswalk GROUP BY 1,2 HAVING count(DISTINCT golden_id)=1)"), "block publish; steward")
    for t in ["c360_account", "c360_promise", "c360_contact", "c360_case", "c360_note"]:
        rule(3, f"{t}: every row has a valid golden_id", one(f"SELECT count(*) FROM {t}"), one(f"SELECT count(*) FROM {t} WHERE golden_id IS NULL OR golden_id NOT IN (SELECT golden_id FROM c360_customer)"), "set rows aside")
    rule(4, "days_past_due>=0, overdue>=0, overdue<=balance (published accounts)", one("SELECT count(*) FROM c360_account"),
         one("SELECT count(*) FROM c360_account WHERE days_past_due<0 OR overdue_amount<0 OR overdue_amount>balance"), "set rows aside")
    rule(5, "dob / phone / postal valid in customer record (raw kept)", n, one("SELECT count(*) FROM c360_customer WHERE dob IS NULL OR phone IS NULL OR postal_code IS NULL"), "set aside, keep raw")
    rule(6, "hardship and consent flags never empty", n, one("SELECT count(*) FROM c360_customer WHERE hardship_flag IS NULL OR consent_sms IS NULL OR consent_email IS NULL OR consent_phone IS NULL"), "block publish")
    rule(7, "freshness: snapshot date within 2 days of expected", 1, one(f"SELECT CASE WHEN (DATE '{SNAPSHOT_DATE}' - max(snapshot_date)) > 2 THEN 1 ELSE 0 END FROM c360_account"), "mark stale; alert owner")
    # source-data issues found (feeds dq_report.md)
    I = []
    crm_n = one("SELECT count(*) FROM raw_crm_customers"); I.append(("CRM duplicate customer records merged by matching", crm_n - n, crm_n))
    for fld, norm in [("dob", "dob"), ("phone", "phone10"), ("postal_code", "postal")]:
        I.append((f"{fld}: present but invalid -> quarantined (raw kept)", one(f"SELECT count(*) FROM quarantine WHERE field='{fld}'"), one("SELECT count(*) FROM silver_identity")))
        I.append((f"{fld}: missing", one(f"SELECT count(*) FROM silver_identity WHERE {'dob_raw' if fld=='dob' else 'phone_raw' if fld=='phone' else 'postal_raw'} IS NULL"), one("SELECT count(*) FROM silver_identity")))
    I.append(("dob stored in a non-ISO format (mixed formats standardised)", one("SELECT count(*) FROM silver_identity WHERE dob_raw IS NOT NULL AND dob_raw NOT LIKE '____-__-__'"), one("SELECT count(*) FROM silver_identity")))
    raw_acc = sum(one(f"SELECT count(*) FROM raw_{t}") for t, _ in ACCOUNTS.values())
    I.append(("exact duplicate account rows removed", raw_acc - one("SELECT count(*) FROM _acc"), raw_acc))
    for reason in ["negative days_past_due", "overdue exceeds balance", "no golden_id (unlinked customer)"]:
        I.append((f"accounts set aside: {reason}", one(f"SELECT count(*) FROM quarantine WHERE reason='{reason}'"), raw_acc))
    I.append(("notes containing phone/email, masked before Layer 2", one("SELECT count(*) FROM raw_agent_notes WHERE note_text LIKE '%@%'"), one("SELECT count(*) FROM raw_agent_notes")))
    I.append(("pairs routed to data-steward review", one("SELECT count(*) FROM steward_queue"), one("SELECT count(*) FROM fuzzy_pairs")))
    md = ["# Data quality report (auto-generated)\n", "## Issues found in source data\n", "| Issue | Records affected | Out of | Handling |", "|---|---|---|---|"]
    H = lambda d: ("merged under one golden_id" if "duplicate customer" in d else "raw kept; normalised value NULL; logged in quarantine" if "invalid" in d
        else "left NULL; matching uses remaining fields" if "missing" in d else "parsed with 4 formats into one DATE type" if "non-ISO" in d
        else "removed (exact duplicates)" if "duplicate account" in d else "kept out of c360_account; row retained in quarantine" if "set aside" in d
        else "regex-masked in l2.notes" if "masked" in d else "queued for human review")
    md += [f"| {d} | {a:,} | {b:,} | {H(d)} |" for d, a, b in I]
    md += ["\n## Contract rules (run on every refresh)\n", "| # | Rule | Checked | Failed | Status | If it fails |", "|---|---|---|---|---|---|"]
    md += [f"| {r['rule']} | {r['description']} | {r['checked']:,} | {r['failed']:,} | {r['status']} | {r['on_fail']} |" for r in R]
    Path("out/dq_report.md").write_text("\n".join(md)); Path("out/dq_results.json").write_text(json.dumps(R, indent=1))
    return R

# ---------------------------------------------------------------- 6. MASKED L2 DATABASE (physically has no PII / protected columns)
def build_l2(con):
    Path(L2DB).unlink(missing_ok=True)
    con.execute(f"ATTACH '{L2DB}' AS l2db")
    mask = lambda c: f"""regexp_replace(regexp_replace({c},'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{{2,}}','[EMAIL]','g'),'\\+?\\d[\\d\\s().-]{{8,}}\\d','[PHONE]','g')"""
    con.execute("CREATE TABLE l2db.customers AS SELECT golden_id, language, segment, hardship_flag, data_snapshot_date AS consent_as_of_date, consent_sms, consent_email, consent_phone, bureau_score, bureau_band, n_product_systems FROM c360_customer CROSS JOIN pipeline_run")
    con.execute("CREATE TABLE l2db.accounts AS SELECT * FROM c360_account")
    con.execute("CREATE TABLE l2db.cases AS SELECT * FROM c360_case")
    con.execute("CREATE TABLE l2db.promises AS SELECT * FROM c360_promise")
    con.execute("CREATE TABLE l2db.contacts AS SELECT * FROM c360_contact")
    con.execute(f"CREATE TABLE l2db.notes AS SELECT note_id, golden_id, {mask('note_text')} AS note_text_masked FROM c360_note")
    con.execute(f"CREATE TABLE l2db.bureau_scores AS SELECT golden_id, data_snapshot_date AS score_date, bureau_score FROM c360_customer CROSS JOIN pipeline_run WHERE bureau_score IS NOT NULL")
    con.execute("CREATE TABLE l2db.pipeline_run AS SELECT * FROM pipeline_run")
    con.execute("DETACH l2db")

def evaluate_matching(con):
    """Only possible on the mock (has ground truth). On real data use steward-labelled samples instead."""
    p = Path(RAW_DIR) / "_truth_all.csv"
    if not p.exists(): return None
    t = pd.read_csv(p, dtype=str); xw = q(con, "SELECT * FROM id_crosswalk")
    m = xw.merge(t, left_on=["source_system", "source_id"], right_on=["system", "source_id"])
    pairs = lambda s: int((s * (s - 1) // 2).sum())
    pred = pairs(m.groupby("golden_id").size()); good = pairs(m.groupby(["golden_id", "pid"]).size()); true = pairs(m.groupby("pid").size())
    return dict(pairwise_precision=round(good / pred, 4), pairwise_recall=round(good / true, 4), predicted_pairs=pred, correct_pairs=good, true_pairs=true)

def run():
    Path("out").mkdir(exist_ok=True); Path(DB).unlink(missing_ok=True)
    con = duckdb.connect(DB); t0 = time.time(); log = {}
    for fn in (bronze, silver, match, golden, dq, build_l2):
        r = fn(con); log[fn.__name__] = r if isinstance(r, (dict, list)) else None; print(f"[layer1] {fn.__name__:<8} ok  ({time.time()-t0:.1f}s)")
    ev = evaluate_matching(con)
    if ev: print("[layer1] matching vs ground truth (mock only):", ev)
    return con

if __name__ == "__main__": run()
