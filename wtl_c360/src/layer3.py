"""LAYER 3 - Feature Store.
A 'feature' = one number/flag about a customer that a model or agent screen can use (e.g. 'broken promises in last 90 days').
The rule that matters: each feature is defined ONCE (FEATURE_SQL below). Training, nightly scoring and 'score one customer now'
all call the same function, so the model never sees a number in training that is calculated differently in real life.
Reads ONLY the masked Layer-2 database, so protected attributes and personal data cannot get in by construction."""
import sys, re, yaml, duckdb, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "config"))
from schema_map import SNAPSHOT_DATE, PROTECTED

SRC, STORE, VERSION = "out/l2.duckdb", "out/feature_store.duckdb", "v1"

# Text 'readers' (rules today; an LLM can replace the rules later without changing anything downstream)
TEXT_RULES = {
  "txt_job_loss":    r"lost my job|laid off|hours cut|unemploy",
  "txt_medical":     r"medical|hospital|illness|sick",
  "txt_income_delay":r"salary delayed|paycheque|paycheck",
  "txt_dispute":     r"dispute|already paid|paid already|wants a review",
  "txt_cooperative": r"will pay|sounded cooperative|promised to pay|forgot",
  "txt_struggling":  r"struggl|cannot make|can't afford|bills",
}

# name -> (group, comes from, plain-English meaning). This is also the registry written to config/feature_registry.yaml
REGISTRY = {
  "max_days_past_due":      ("overdue",  "accounts",  "Worst days-past-due across the customer's accounts"),
  "total_overdue_amount":   ("overdue",  "accounts",  "Total dollars overdue across card, loan and deposit"),
  "n_products_overdue":     ("overdue",  "accounts",  "How many accounts are overdue"),
  "total_balance":          ("overdue",  "accounts",  "Total balance across accounts"),
  "promises_made_90d":      ("promise",  "promises",  "Promises to pay made in the last 90 days"),
  "promises_broken_90d":    ("promise",  "promises",  "Of those, how many were broken"),
  "promise_kept_rate_90d":  ("promise",  "promises",  "Share of last-90-day promises that were kept (blank if none)"),
  "days_since_last_promise":("promise",  "promises",  "Days since the most recent promise"),
  "contact_attempts_14d":   ("contact",  "contacts",  "Times we contacted them in the last 14 days"),
  "last_contact_outcome":   ("contact",  "contacts",  "Outcome of the most recent contact"),
  "best_channel":           ("contact",  "contacts",  "Channel they answer most often (needs 2+ attempts)"),
  "consent_sms":            ("contact",  "customers", "Agreed to SMS"),
  "consent_email":          ("contact",  "customers", "Agreed to email"),
  "consent_phone":          ("contact",  "customers", "Agreed to phone calls"),
  "hardship_flag":          ("context",  "customers", "Officially flagged as financial hardship"),
  "bureau_score":           ("context",  "customers", "Credit bureau score"),
  "case_stage":             ("context",  "cases",     "Delinquency stage of the main case"),
  "case_status":            ("context",  "cases",     "Status of the main case (open, on_hold, disputed...)"),
  "days_case_open":         ("context",  "cases",     "Days since the main case was opened"),
  **{k: ("text", "notes", f"Agent notes mention: {v.split('|')[0]} (and similar)") for k, v in TEXT_RULES.items()},
  "hardship_review_needed": ("text",     "notes",     "Notes hint at hardship but the customer is NOT flagged -> a person should check"),
}
# Account features are 'as of the latest snapshot on or before the date'. The mock has ONE snapshot (28 Sep);
# the Maple release has quarterly snapshots through June 2026.
NO_HISTORY_IN_MOCK = {"max_days_past_due", "total_overdue_amount", "n_products_overdue", "total_balance"}

def feature_sql(as_of, golden_ids=None):
    """THE one definition. as_of = the 'today' we pretend it is; nothing on/after that date is ever used."""
    gf = ""
    gf_c = ""
    source_con = duckdb.connect(SRC, read_only=True)
    has_transcripts = source_con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='transcripts'"
    ).fetchone()[0] > 0
    source_con.close()
    if golden_ids:
        ids = ",".join("'" + re.sub(r"[^A-Za-z0-9]", "", g) + "'" for g in golden_ids)
        gf = f"AND golden_id IN ({ids})"
        gf_c = f"AND c.golden_id IN ({ids})"
    txt = ",\n    ".join(f"coalesce(bool_or(regexp_matches(n.note_text_masked, '{rx.replace(chr(39), chr(39)*2)}', 'i')), false) AS {k}" for k, rx in TEXT_RULES.items())
    transcript_text = (
        f"\n  UNION ALL\n  SELECT t.golden_id, t.transcript_text_masked AS note_text_masked\n"
        f"  FROM transcripts t WHERE t.call_start_ts::DATE < DATE '{as_of}' {gf}"
        if has_transcripts else ""
    )
    return f"""
WITH base AS (  -- one row per customer who had a case by as_of; main case = most recently opened
  SELECT golden_id, case_stage,
         CASE WHEN status_date < DATE '{as_of}' THEN status ELSE 'open' END AS case_status,
         days_case_open FROM (
  SELECT golden_id, stage AS case_stage, status, status_date,
         DATE '{as_of}' - opened_date AS days_case_open,
         row_number() OVER (PARTITION BY golden_id ORDER BY opened_date DESC, case_id) AS rn
  FROM cases WHERE opened_date <= DATE '{as_of}' {gf}) WHERE rn = 1),
acc AS (
  SELECT golden_id, max(days_past_due) AS max_days_past_due, round(sum(overdue_amount), 2) AS total_overdue_amount,
         count(*) FILTER (WHERE days_past_due > 0) AS n_products_overdue, round(sum(balance), 2) AS total_balance
  FROM (SELECT *, row_number() OVER (PARTITION BY account_id, golden_id ORDER BY snapshot_date DESC) rn FROM accounts WHERE snapshot_date <= DATE '{as_of}' {gf})
  WHERE rn = 1 GROUP BY 1),
pr AS (
  SELECT golden_id, count(*) AS promises_made_90d,
         count(*) FILTER (WHERE status = 'BROKEN' AND status_date < DATE '{as_of}') AS promises_broken_90d,
         round(count(*) FILTER (WHERE status = 'KEPT' AND status_date < DATE '{as_of}') * 1.0 /
               nullif(count(*) FILTER (WHERE status IN ('KEPT', 'BROKEN') AND status_date < DATE '{as_of}'), 0), 3) AS promise_kept_rate_90d,
         DATE '{as_of}' - max(promised_date) AS days_since_last_promise
  FROM promises WHERE promised_date < DATE '{as_of}' AND promised_date >= DATE '{as_of}' - INTERVAL 90 DAY {gf} GROUP BY 1),
ct AS (
  SELECT golden_id, count(*) FILTER (WHERE contact_date >= DATE '{as_of}' - INTERVAL 14 DAY) AS contact_attempts_14d,
         arg_max(nullif(outcome, 'na'), contact_date) AS last_contact_outcome
  FROM contacts WHERE contact_date < DATE '{as_of}' {gf} GROUP BY 1),
ch AS (
  SELECT golden_id, channel AS best_channel FROM (
    SELECT golden_id, channel, count(*) n, avg((outcome IN ('answered','promise_made'))::INT) rate FROM contacts
    WHERE contact_date < DATE '{as_of}' {gf} GROUP BY 1, 2 HAVING count(*) >= 2)
  QUALIFY row_number() OVER (PARTITION BY golden_id ORDER BY rate DESC, n DESC, channel) = 1),
hf AS (
  SELECT golden_id,
         bool_or(hardship_flag AND coalesce(hardship_start_date, opened_date) < DATE '{as_of}') AS hardship_flag
  FROM cases WHERE opened_date <= DATE '{as_of}' {gf} GROUP BY 1),
br AS (
  SELECT s.golden_id, avg(s.bureau_score) AS bureau_score
  FROM bureau_scores s
  JOIN (
    SELECT golden_id, max(score_date) AS score_date
    FROM bureau_scores WHERE score_date <= DATE '{as_of}' {gf} GROUP BY 1
  ) latest USING (golden_id, score_date)
  GROUP BY s.golden_id),
tx AS (
  SELECT n.golden_id, {txt}
  FROM (
    SELECT c.golden_id, n.note_text_masked FROM contacts c
    JOIN notes n ON n.note_id = c.note_id
    WHERE c.contact_date < DATE '{as_of}' {gf_c}
    {transcript_text}
  ) n GROUP BY 1)
SELECT b.golden_id, DATE '{as_of}' AS as_of,
  a.max_days_past_due, a.total_overdue_amount, a.n_products_overdue, a.total_balance,
  coalesce(p.promises_made_90d, 0) AS promises_made_90d, coalesce(p.promises_broken_90d, 0) AS promises_broken_90d,
  p.promise_kept_rate_90d, p.days_since_last_promise,
  coalesce(k.contact_attempts_14d, 0) AS contact_attempts_14d, k.last_contact_outcome, h.best_channel,
  CASE WHEN cu.consent_as_of_date <= DATE '{as_of}' THEN cu.consent_sms END AS consent_sms,
  CASE WHEN cu.consent_as_of_date <= DATE '{as_of}' THEN cu.consent_email END AS consent_email,
  CASE WHEN cu.consent_as_of_date <= DATE '{as_of}' THEN cu.consent_phone END AS consent_phone,
  coalesce(hf.hardship_flag, cu.hardship_flag) AS hardship_flag, br.bureau_score,
  b.case_stage, b.case_status, b.days_case_open,
  {", ".join(f"coalesce(t.{k}, false) AS {k}" for k in TEXT_RULES)},
  (NOT coalesce(hf.hardship_flag, cu.hardship_flag) AND (coalesce(t.txt_job_loss,false) OR coalesce(t.txt_medical,false) OR coalesce(t.txt_struggling,false))) AS hardship_review_needed
FROM base b JOIN customers cu USING (golden_id)
LEFT JOIN acc a USING (golden_id) LEFT JOIN pr p USING (golden_id) LEFT JOIN ct k USING (golden_id) LEFT JOIN ch h USING (golden_id) LEFT JOIN hf USING (golden_id) LEFT JOIN br USING (golden_id) LEFT JOIN tx t USING (golden_id)
ORDER BY b.golden_id"""

def compute(as_of=SNAPSHOT_DATE, golden_ids=None):
    """Used by training, nightly batch AND live single-customer scoring."""
    con = duckdb.connect(SRC, read_only=True)
    try: return con.execute(feature_sql(as_of, golden_ids)).fetchdf()
    finally: con.close()

def label_sql(as_of, horizon=30):
    return f"""SELECT golden_id, (status = 'KEPT')::INT AS label_kept_30d FROM (
      SELECT golden_id, status, row_number() OVER (PARTITION BY golden_id ORDER BY promised_date, ptp_id) rn FROM promises
      WHERE promised_date >= DATE '{as_of}' AND promised_date < DATE '{as_of}' + INTERVAL {horizon} DAY
        AND status_date < DATE '{as_of}' + INTERVAL {horizon} DAY
        AND status IN ('KEPT', 'BROKEN')) WHERE rn = 1"""

def training_set(as_of, horizon=30):
    """Features as they looked on `as_of` + what really happened in the next `horizon` days (the answer the model learns from)."""
    X = compute(as_of); con = duckdb.connect(SRC, read_only=True)
    y = con.execute(label_sql(as_of, horizon)).fetchdf(); con.close()
    return X.merge(y, on="golden_id", how="inner")


def temporal_training_sets(as_of_dates, horizon=30):
    """Build an auditable panel of point-in-time feature/label snapshots."""
    dates = sorted(set(as_of_dates))
    if not dates:
        raise ValueError("At least one as_of date is required")
    return pd.concat(
        [training_set(as_of, horizon) for as_of in dates],
        ignore_index=True,
    )


def online(golden_id, as_of=SNAPSHOT_DATE):
    """'Score one customer right now' - literally the same query, filtered to one customer."""
    df = compute(as_of, [golden_id]); return df.iloc[0].to_dict() if len(df) else None

def publish(as_of=SNAPSHOT_DATE):
    """Nightly batch: materialise features for the whole queue into the store, stamped with the version."""
    df = compute(as_of); Path(STORE).unlink(missing_ok=True); con = duckdb.connect(STORE)
    df["feature_version"] = VERSION; con.register("f", df); con.execute("CREATE TABLE features_live AS SELECT *, current_timestamp::VARCHAR AS computed_at FROM f")
    Path("config/feature_registry.yaml").write_text(yaml.safe_dump({"version": VERSION, "features": {k: dict(group=g, source=s, meaning=m,
        point_in_time_safe=k not in NO_HISTORY_IN_MOCK or "no: mock has only 1 snapshot (real data has monthly)") for k, (g, s, m) in REGISTRY.items()}}, sort_keys=False))
    con.close(); return df

# ---------------------------------------------------------------- automatic safety tests
def test_no_protected(df):
    """Fails the build if a protected attribute (or an obvious stand-in) appears as a feature."""
    banned = set(PROTECTED) | {"gender", "marital_status", "postal_code", "postal", "first_name", "last_name", "name", "employer", "dob", "age", "phone", "email"}
    hit = [c for c in df.columns if c.lower() in banned or any(b in c.lower() for b in ("gender", "marital", "citizen", "newcomer", "postal", "employer"))]
    cols_in_source = {r[0] for r in duckdb.connect(SRC, read_only=True).execute("SELECT column_name FROM information_schema.columns").fetchall()}
    leaked = [c for c in cols_in_source if c.lower() in banned]
    assert not hit and not leaked, f"protected/identifying feature found: {hit or leaked}"
    return f"{len(df.columns)} columns checked; none protected; source DB holds no protected/identifying columns"

def test_parity(n=25):
    """Same customer computed in bulk vs one-at-a-time must give identical values (training/serving skew check)."""
    bulk = compute().set_index("golden_id"); ids = list(bulk.index[:n]); bad = 0
    for g in ids:
        one = pd.Series(online(g)).drop("golden_id")
        a = bulk.loc[g].astype(object).where(bulk.loc[g].notna(), None); b = one.astype(object).where(one.notna(), None)
        bad += int(not all((a[c] == b[c]) or (a[c] is None and b[c] is None) for c in a.index))
    assert bad == 0, f"{bad} customers differ between bulk and live"
    return f"{len(ids)} customers: bulk == live for every feature"

def test_point_in_time(as_of):
    """Independently recompute temporal features from records available before as_of."""
    con = duckdb.connect(SRC, read_only=True); pr = con.execute("SELECT golden_id, promised_date, status_date, status FROM promises").fetchdf()
    ct = con.execute("SELECT golden_id, contact_date FROM contacts").fetchdf()
    hf = con.execute("SELECT golden_id, opened_date, hardship_start_date, hardship_flag FROM cases").fetchdf()
    br = con.execute("SELECT golden_id, score_date, bureau_score FROM bureau_scores").fetchdf()
    cu = con.execute("SELECT golden_id, consent_as_of_date, consent_sms, consent_email, consent_phone FROM customers").fetchdf()
    con.close()
    t = pd.Timestamp(as_of); X = compute(as_of).set_index("golden_id")
    pr["promised_date"] = pd.to_datetime(pr.promised_date); pr["status_date"] = pd.to_datetime(pr.status_date)
    ct["contact_date"] = pd.to_datetime(ct.contact_date)
    hf["opened_date"] = pd.to_datetime(hf.opened_date)
    hf["hardship_start_date"] = pd.to_datetime(hf.hardship_start_date)
    br["score_date"] = pd.to_datetime(br.score_date)
    cu["consent_as_of_date"] = pd.to_datetime(cu.consent_as_of_date)
    m = pr[(pr.promised_date < t) & (pr.promised_date >= t - pd.Timedelta(days=90))].groupby("golden_id").size()
    resolved = pr[(pr.promised_date < t) & (pr.promised_date >= t - pd.Timedelta(days=90))
                  & (pr.status_date < t) & pr.status.isin(["KEPT", "BROKEN"])]
    broken = resolved[resolved.status == "BROKEN"].groupby("golden_id").size()
    kept_rate = resolved.assign(kept=resolved.status == "KEPT").groupby("golden_id").kept.mean().round(3)
    c = ct[(ct.contact_date < t) & (ct.contact_date >= t - pd.Timedelta(days=14))].groupby("golden_id").size()
    h = hf[hf.opened_date <= t].assign(
        active=lambda rows: rows.hardship_flag
        & (rows.hardship_start_date.fillna(rows.opened_date) < t)
    ).groupby("golden_id").active.any()
    recent_bureau = br[br.score_date <= t]
    latest_bureau_dates = recent_bureau.groupby("golden_id").score_date.max()
    b = recent_bureau[
        recent_bureau.score_date.eq(recent_bureau.golden_id.map(latest_bureau_dates))
    ].groupby("golden_id").bureau_score.mean()
    consent = cu.set_index("golden_id")
    consent_is_known = consent.consent_as_of_date <= t
    same = lambda actual, expected: (actual.eq(expected) | (actual.isna() & expected.isna())).all()
    ok1 = same(X.promises_made_90d, m.reindex(X.index).fillna(0))
    ok2 = same(X.contact_attempts_14d, c.reindex(X.index).fillna(0))
    ok3 = same(X.hardship_flag, h.reindex(X.index).fillna(False))
    ok4 = same(X.bureau_score, b.reindex(X.index))
    ok5 = same(X.promises_broken_90d, broken.reindex(X.index).fillna(0))
    ok6 = same(X.promise_kept_rate_90d, kept_rate.reindex(X.index))
    ok7 = same(X.consent_sms, consent.consent_sms.reindex(X.index).where(consent_is_known.reindex(X.index)))
    ok8 = same(X.consent_email, consent.consent_email.reindex(X.index).where(consent_is_known.reindex(X.index)))
    ok9 = same(X.consent_phone, consent.consent_phone.reindex(X.index).where(consent_is_known.reindex(X.index)))
    assert all((ok1, ok2, ok3, ok4, ok5, ok6, ok7, ok8, ok9)), "point-in-time mismatch"
    return f"as_of {as_of}: promises, contacts, hardship, bureau and consent features match prior-date recomputations"


def test_text_point_in_time(as_of):
    """Compare every text feature with masked text records available strictly before as_of."""
    con = duckdb.connect(SRC, read_only=True)
    has_transcripts = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name='transcripts'"
    ).fetchone()[0] > 0
    transcript_sql = (
        f"UNION ALL SELECT golden_id, transcript_text_masked AS text FROM transcripts "
        f"WHERE call_start_ts::DATE < DATE '{as_of}'"
        if has_transcripts else ""
    )
    rules_sql = ", ".join(
        f"coalesce(bool_or(regexp_matches(text, '{pattern.replace(chr(39), chr(39) * 2)}', 'i')), false) AS {name}"
        for name, pattern in TEXT_RULES.items()
    )
    expected = con.execute(f"""
        SELECT golden_id, {rules_sql}
        FROM (
          SELECT c.golden_id, n.note_text_masked AS text
          FROM contacts c JOIN notes n ON n.note_id=c.note_id
          WHERE c.contact_date < DATE '{as_of}'
          {transcript_sql}
        ) docs GROUP BY golden_id
    """).fetchdf().set_index("golden_id")
    con.close()
    actual = compute(as_of).set_index("golden_id")
    for name in TEXT_RULES:
        wanted = expected[name].reindex(actual.index).fillna(False).astype(bool)
        observed = actual[name].fillna(False).astype(bool)
        assert observed.equals(wanted), f"text feature {name} used a future or mismatched source at {as_of}"
    return f"as_of {as_of}: all text flags match masked notes/transcripts before that date"


if __name__ == "__main__":
    df = publish(); print("published", df.shape, "->", STORE)
