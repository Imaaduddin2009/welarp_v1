"""LAYER 2 - Insight & NLP: scope check -> semantic layer -> NL-to-SQL (LLM or offline templates) -> sqlglot guard
-> execute on the MASKED db -> answer + SQL shown + freshness.  Notes/calls go to a text-search route (RAG)."""
import os, re, json, time, yaml, duckdb, sqlglot, pandas as pd
from pathlib import Path
from sqlglot import exp

CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config" / "semantic.yaml"))
L2DB = os.getenv("C360_L2_DB", "out/l2.duckdb")
AUDIT = os.getenv("C360_AUDIT_LOG", "out/audit_log.jsonl")
SNAP = CFG["snapshot_date"]; ALLOWED = set(CFG["tables"]); FORBID = set(CFG["forbidden_columns"])
MAX_ROWS = 1000
LEAD_MIN_CELL_SIZE = 10

# ------------------------------------------------------------------ 1. scope check (rules; deterministic and auditable)
RULES = [
  ("protected", r"\b(gender|male|female|marital|citizen\w*|immigra\w*|newcomer|household|dependants?|accessib\w*|disab\w*|vulnerab\w*|accent\w*|race|racial|ethnic\w*|religio\w*|nationality|sexual\w*|postal areas?|postal codes?|postcodes?|zip codes?|FSAs?|area codes?)\b",
   "it uses or tries to infer a protected attribute, which must not be used in collections decisions",
   "How many hardship customers have an account more than 30 days past due?"),
  ("pii", r"\b(phone numbers?|email addresses?|home address(es)?|date of birth|dob|sin|national id|full names?|names? of)\b",
   "personal data is masked in this system, so names and contact details cannot be returned",
   "How many customers consented to phone contact?"),
  ("bulk", r"\b(export|dump|download|extract)\b|\b(every|all)\b.*\b(customers?'? (data|records|details)|records)\b",
   "bulk export of customer data is not allowed", "How many customers are in each segment?"),
  ("write", r"\b(delete|drop|update|insert|alter|truncate|modify|overwrite)\b",
   "this system is read-only", "How many open cases are there by stage?"),
]
def scope_check(question, user):
    ql = question.lower()
    for name, rx, why, suggest in RULES:
        if re.search(rx, ql): return dict(ok=False, reason=why, suggestion=suggest, rule=name)
    if not any(v in ql for v in CFG["vocabulary"]):
        return dict(ok=False, rule="out_of_scope", reason="it is not about Maple Bank collections data",
                    suggestion="Which product type has the highest total overdue amount?")
    return dict(ok=True)

# ------------------------------------------------------------------ 2. route
def route(question):
    ql = question.lower()
    if re.search(
        r"\b(policy|procedure|regulation|regulatory|permitted|allowed|must|should|"
        r"can we|can an agent|how many .*attempts?|consent required|grace period)\b",
        ql,
    ):
        return "text"
    aggregate = re.match(r"\s*(how many|count|total|average|avg|what (is|are) the (total|average|number))", ql)
    if re.search(r"\b(said|say|saying|told|mention\w*|reason|transcripts?|talk\w* about|complain\w*|explain\w*)\b", ql) and not aggregate: return "text"
    return "sql"

# ------------------------------------------------------------------ 3a. NL -> SQL via LLM (only if ANTHROPIC_API_KEY is set; one cheap call per question)
def schema_prompt():
    t = "\n".join(f"- {n}: {v['description']}\n" + "\n".join(f"    {c}: {d}" for c, d in v["columns"].items()) for n, v in CFG["tables"].items())
    return (f"You write ONE read-only DuckDB SELECT for a collections analytics tool. Use ONLY these tables/columns:\n{t}\n\nJoins: {CFG['joins']}\n"
            f"Business terms: {CFG['terms']}\nToday is {SNAP}. Never use the wall clock; use DATE '{SNAP}'.\n"
            "Output ONLY the SQL, no markdown. If the question cannot be answered from these tables, output exactly: REFUSE: <short reason>.")
def llm_sql(question, error=None):
    import anthropic
    msgs = [{"role": "user", "content": question + (f"\nYour previous SQL failed with: {error}. Fix it." if error else "")}]
    r = anthropic.Anthropic().messages.create(model=os.getenv("WTL_MODEL", "claude-haiku-4-5-20251001"), max_tokens=500, system=schema_prompt(), messages=msgs)
    return re.sub(r"^```\w*|```$", "", r.content[0].text.strip(), flags=re.M).strip()

# ------------------------------------------------------------------ 3b. offline templates (no API key needed; also the safe fallback)
def _n(ql, default):
    m = re.search(r"(\d+)\s*(day|days)", ql) or re.search(r"(?:last|past)\s+(\d+)", ql); return int(m.group(1)) if m else default
def _prod(ql): return next((p for p in ("card", "loan", "deposit") if p in ql), None)
def template_sql(question):
    ql = question.lower()
    if re.search(r"broken promise", ql) or ("promise" in ql and "broke" in ql):
        n = _n(ql, 30); base = f"FROM promises WHERE status = 'BROKEN' AND promised_date >= DATE '{SNAP}' - INTERVAL {n} DAY"
        if re.search(r"\b(which|list|who)\b", ql): return f"SELECT DISTINCT golden_id {base} ORDER BY golden_id"
        return f"SELECT count(DISTINCT golden_id) AS customers_with_broken_promise {base}"
    if re.search(r"(more than|over|at least|>)\s*(\d+).*broken|broken.*(more than|over)\s*(\d+)", ql):
        k = int(re.findall(r"\d+", ql)[0]) if re.findall(r"\d+", ql) else 1
        return f"SELECT golden_id, count(*) AS broken_promises FROM promises WHERE status='BROKEN' AND promised_date >= DATE '{SNAP}' - INTERVAL 90 DAY GROUP BY 1 HAVING count(*) > {k} ORDER BY 2 DESC"
    if "kept rate" in ql or "promise-kept" in ql or ("promise" in ql and "rate" in ql):
        return "SELECT round(100.0 * count(*) FILTER (WHERE status = 'KEPT') / count(*), 1) AS promise_kept_rate_pct FROM promises WHERE status IN ('KEPT','BROKEN')"
    if "hardship" in ql and re.search(r"overdue|past due|delinquen", ql):
        return "SELECT count(DISTINCT c.golden_id) AS hardship_customers_delinquent FROM customers c JOIN accounts a USING (golden_id) WHERE c.hardship_flag AND a.days_past_due > 0"
    if "hardship" in ql: return "SELECT count(*) AS hardship_customers FROM customers WHERE hardship_flag"
    if re.search(r"total overdue|overdue amount|overdue.*by product|most overdue|highest.*overdue", ql):
        return "SELECT product_type, round(sum(overdue_amount), 2) AS total_overdue FROM accounts GROUP BY 1 ORDER BY 2 DESC"
    if re.search(r"average days past due|avg days past due|average dpd", ql):
        p = _prod(ql); w = f"WHERE product_type = '{p}' AND days_past_due > 0" if p else "WHERE days_past_due > 0"
        return f"SELECT round(avg(days_past_due), 1) AS avg_days_past_due FROM accounts {w}"
    if "delinquent" in ql or "past due" in ql:
        p = _prod(ql)
        if re.match(r"\s*how many", ql) and p: return f"SELECT count(*) AS delinquent_{p}_accounts FROM accounts WHERE product_type='{p}' AND days_past_due > 0"
        return "SELECT product_type, count(*) AS delinquent_accounts FROM accounts WHERE days_past_due > 0 GROUP BY 1 ORDER BY 2 DESC"
    if re.search(r"agent", ql) and re.search(r"top|best|most|kept", ql):
        return "SELECT c.agent_id, count(*) FILTER (WHERE p.status='KEPT') AS promises_kept, count(*) AS promises_total FROM promises p JOIN cases c USING (case_id) GROUP BY 1 ORDER BY 2 DESC LIMIT 5"
    if "consent" in ql:
        ch = next((c for c in ("sms", "email", "phone") if c in ql), "phone")
        return f"SELECT count(*) AS customers_with_{ch}_consent FROM customers WHERE consent_{ch}"
    if "channel" in ql or "contact" in ql:
        return "SELECT channel, count(*) AS contacts, round(100.0*count(*) FILTER (WHERE outcome IN ('answered','promise_made'))/count(*),1) AS answer_rate_pct FROM contacts GROUP BY 1 ORDER BY 2 DESC"
    if "case" in ql and ("stage" in ql or "status" in ql):
        return f"SELECT {'stage' if 'stage' in ql else 'status'}, count(*) AS cases FROM cases GROUP BY 1 ORDER BY 2 DESC"
    if "segment" in ql: return "SELECT segment, count(*) AS customers FROM customers GROUP BY 1 ORDER BY 2 DESC"
    if re.search(r"job|laid off|unemploy", ql) and re.search(r"how many|count|number", ql):
        return "SELECT count(DISTINCT golden_id) AS customers_mentioning_job_loss FROM notes WHERE note_text_masked ILIKE '%lost my job%' OR note_text_masked ILIKE '%hours cut%' OR note_text_masked ILIKE '%laid off%'"
    return None

# ------------------------------------------------------------------ 4. guard (read-only, approved tables/columns, role scoping, row cap)
class Blocked(Exception): pass
def guard(sql, user):
    stmts = sqlglot.parse(sql, read="duckdb")
    if len(stmts) != 1: raise Blocked("only a single statement is allowed")
    tree = stmts[0]
    if not isinstance(tree, (exp.Select, exp.Union)): raise Blocked("only SELECT queries are allowed")
    bad = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Command, exp.Alter, exp.Copy, exp.Pragma, exp.Set, exp.Attach)
    if tree.find(*bad): raise Blocked("write / admin statements are blocked")
    ctes = {c.alias for c in tree.find_all(exp.CTE)}
    for t in tree.find_all(exp.Table):
        if not isinstance(t.this, exp.Identifier): raise Blocked("table functions (e.g. read_csv) are blocked")
        if t.name in ctes: continue
        if t.db or t.catalog or t.name not in ALLOWED: raise Blocked(f"table '{t.name}' is not an approved table")
    for c in tree.find_all(exp.Column):
        if c.name.lower() in FORBID: raise Blocked(f"column '{c.name}' is protected or personal data")
    if user.get("role") == "lead":
        group = tree.args.get("group")
        group_sql = set()
        if group:
            for expression in group.expressions:
                if isinstance(expression, exp.Literal) and expression.is_int:
                    index = int(expression.this) - 1
                    if 0 <= index < len(tree.expressions):
                        expression = tree.expressions[index]
                        expression = expression.this if isinstance(expression, exp.Alias) else expression
                group_sql.add(expression.sql(dialect="duckdb").lower())
                if any(
                    column.name.lower() in {
                        "golden_id", "customer_id", "account_id", "case_id",
                        "contact_id", "ptp_id", "note_id", "transcript_id",
                        "source_id", "crm_customer_id",
                    }
                    for column in expression.find_all(exp.Column)
                ):
                    raise Blocked("team leads cannot group results by customer or record identifiers")
        for projection in tree.expressions:
            expression = projection.this if isinstance(projection, exp.Alias) else projection
            if expression.find(exp.AggFunc):
                continue
            if not group_sql or expression.sql(dialect="duckdb").lower() not in group_sql:
                raise Blocked("team leads can only see grouped aggregate summaries")
        if not tree.find(exp.AggFunc):
            raise Blocked("team leads can only see aggregate summaries")
        if group_sql:
            source = next(
                (table for table in tree.find_all(exp.Table)
                 if table.name not in ctes and table.name != "pipeline_run"),
                None,
            )
            if source is None:
                raise Blocked("grouped summaries require a customer-linked source table")
            customer_id = exp.Column(
                this=exp.to_identifier("golden_id"),
                table=exp.to_identifier(source.alias_or_name),
            )
            cell_size = exp.Count(this=exp.Distinct(expressions=[customer_id]))
            minimum_cell = exp.GTE(this=cell_size, expression=exp.Literal.number(LEAD_MIN_CELL_SIZE))
            having = tree.args.get("having")
            if having:
                having.set("this", exp.and_(having.this, minimum_cell))
            else:
                tree.set("having", exp.Having(this=minimum_cell))
    lim = tree.args.get("limit")
    if lim is None: tree = tree.limit(100)
    else:
        try:
            if int(lim.expression.name) > MAX_ROWS: tree = tree.limit(MAX_ROWS)
        except Exception: tree = tree.limit(100)
    if user.get("role") == "agent":                      # agents only ever see their own portfolio
        agent_id = str(user.get("agent_id", ""))
        agent_literal = exp.Literal.string(agent_id).sql(dialect="duckdb")
        def scope(node):
            if isinstance(node, exp.Table) and node.name in ALLOWED and node.name not in ctes and node.name != "pipeline_run":
                cond = (
                    f"agent_id = {agent_literal}"
                    if node.name == "cases"
                    else f"golden_id IN (SELECT golden_id FROM cases WHERE agent_id = {agent_literal})"
                )
                sub = sqlglot.parse_one(f"SELECT * FROM {node.name} WHERE {cond}", read="duckdb")
                return exp.Subquery(this=sub, alias=exp.TableAlias(this=exp.to_identifier(node.alias or node.name)))
            return node
        tree = tree.transform(scope)
    return tree.sql(dialect="duckdb")

# ------------------------------------------------------------------ 5. execute on the masked database
def run_sql(sql):
    con = duckdb.connect(L2DB, read_only=True)
    try:
        con.execute("SET enable_external_access=false")
        return con.execute(sql).fetchdf()
    finally: con.close()

# ------------------------------------------------------------------ text search route (RAG; TF-IDF here, embeddings+FAISS at full scale)
CUES = r"lost my job|laid off|job loss|hours cut|medical|hospital|illness|struggl|cannot make|can't afford|bereave|death|divorce|disaster|emergency"
class NoteSearch:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        con = duckdb.connect(L2DB, read_only=True)
        self.df = con.execute("""
            SELECT DISTINCT ON (golden_id, text) source_type, source_id, golden_id,
                   source_citation, text AS t
            FROM (
              SELECT 'note' AS source_type, note_id AS source_id, golden_id,
                     note_id AS source_citation,
                     note_text_masked AS text FROM notes
              UNION ALL
              SELECT 'transcript' AS source_type, transcript_id AS source_id, golden_id,
                     transcript_id AS source_citation,
                     transcript_text_masked AS text FROM transcripts
              UNION ALL
              SELECT 'policy' AS source_type, chunk_id AS source_id, NULL AS golden_id,
                     doc_id || ' v' || version || ' ' || section AS source_citation,
                     content AS text FROM policy_chunks
            )
            WHERE nullif(trim(text), '') IS NOT NULL
            ORDER BY golden_id, text, source_type, source_id
        """).fetchdf()
        con.close()
        self.vec = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=50000, min_df=2)
        self.X = self.vec.fit_transform(self.df.t)
    def search(self, q, k=5, min_sim=0.08, user=None):
        sims = (self.X @ self.vec.transform([q]).T).toarray().ravel(); order = sims.argsort()[::-1]
        df = self.df
        if user and user.get("role") == "agent":
            con = duckdb.connect(L2DB, read_only=True)
            mine = set(con.execute(f"SELECT DISTINCT golden_id FROM cases WHERE agent_id = '{re.sub(r'[^A-Za-z0-9_]', '', user['agent_id'])}'").fetchdf().golden_id); con.close()
            order = [i for i in order if df.source_type.iloc[i] == "policy" or df.golden_id.iloc[i] in mine]
        return [
            dict(
                source_type=df.source_type.iloc[i],
                source_id=df.source_id.iloc[i],
                source_citation=df.source_citation.iloc[i],
                golden_id=df.golden_id.iloc[i],
                text=df.t.iloc[i],
                sim=round(float(sims[i]), 3),
            )
            for i in order[:k] if sims[i] >= min_sim
        ]

# ------------------------------------------------------------------ orchestrator
class QA:
    def __init__(self, user=None):
        self.user = user or dict(id="analyst1", role="analyst"); self._search = None
        self.use_llm = bool(os.getenv("ANTHROPIC_API_KEY"))
    def _fresh(self):
        c = duckdb.connect(L2DB, read_only=True); r = c.execute("SELECT run_ts::VARCHAR, data_snapshot_date::VARCHAR FROM pipeline_run").fetchone(); c.close()
        return f"data as of {r[1]}; pipeline refreshed {str(r[0])[:19]}"
    def _log(self, out):
        Path(AUDIT).parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT, "a") as f: f.write(json.dumps(dict(ts=time.time(), user=self.user, **{k: out[k] for k in ("question", "route", "sql_or_sources", "refused", "reason")}), default=str) + "\n")
    def _refuse(self, q, reason, suggestion=None, rt="none"):
        out = dict(question=q, answer=f"I can't answer that: {reason}." + (f" Try instead: \"{suggestion}\"" if suggestion else ""), sql_or_sources="", refused=True, reason=reason, route=rt, freshness=self._fresh(), table=None)
        self._log(out); return out
    def ask(self, q):
        sc = scope_check(q, self.user)
        if not sc["ok"]: return self._refuse(q, sc["reason"], sc["suggestion"])
        rt = route(q)
        if rt == "text":
            self._search = self._search or NoteSearch(); hits = self._search.search(q, user=self.user)
            if not hits: return self._refuse(q, "no relevant notes were found, so I can't answer this reliably", None, "text")
            alert = any(
                h["source_type"] != "policy" and re.search(CUES, h["text"], re.I)
                for h in hits
            )
            ans = f"Found {len(hits)} relevant masked record(s). " + " | ".join(
                f"[{h['source_type']}:{h['source_citation']}] \"{h['text']}\"" for h in hits[:3]
            ) + (" Possible hardship cue: refer for human review; no automatic hardship status was set." if alert else "")
            out = dict(question=q, answer=ans, sql_or_sources="; ".join(
                f"{h['source_type']}:{h['source_citation']}" for h in hits
            ), refused=False, reason="", route="text", freshness=self._fresh(), table=pd.DataFrame(hits))
            self._log(out); return out
        sql = None; src = "template"
        if self.use_llm:
            try:
                sql = llm_sql(q); src = "llm"
                if sql.upper().startswith("REFUSE"): return self._refuse(q, sql.split(":", 1)[-1].strip() or "cannot be answered from approved tables", None, "sql")
            except Exception as e: sql = None
        if sql is None: sql = template_sql(q); src = "template"
        if sql is None: return self._refuse(q, "I can't answer this reliably from the approved collections tables", "Which product type has the highest total overdue amount?", "sql")
        try:
            safe = guard(sql, self.user)
        except Blocked as b: return self._refuse(q, f"the generated query was blocked by the safety checker ({b})", None, "sql")
        except Exception as e: return self._refuse(q, f"the generated query could not be validated ({type(e).__name__})", None, "sql")
        try: df = run_sql(safe)
        except Exception as e:
            if src == "llm":                                  # one repair attempt for a binder/syntax error; guard still applies
                try: safe = guard(llm_sql(q, str(e)), self.user); df = run_sql(safe)
                except Exception as e2: return self._refuse(q, f"the query failed ({type(e2).__name__})", None, "sql")
            else: return self._refuse(q, f"the query failed ({type(e).__name__})", None, "sql")
        if df.shape == (1, 1): ans = f"{df.columns[0].replace('_', ' ')}: {df.iat[0, 0]}"
        elif df.empty: ans = "No matching records."
        else: ans = f"{len(df)} result row(s); see the structured results table."
        out = dict(question=q, answer=ans, sql_or_sources=safe, refused=False, reason="", route="sql", freshness=self._fresh(), table=df, via=src)
        self._log(out); return out

def run_benchmark(questions_csv, out_csv, id_col="question_id", q_col="question", user=None):
    """Writes the 4 columns the organisers asked for. Never hand-edit the result."""
    qa = QA(user); df = pd.read_csv(questions_csv); rows = []
    for r in df.itertuples():
        o = qa.ask(getattr(r, q_col))
        if o["refused"] or o["route"] != "sql" or o["table"] is None:
            a = o["answer"]
        elif o["table"].shape == (1, 1):
            a = str(o["table"].iat[0, 0])
        else:
            a = o["table"].to_csv(index=False).strip()
        rows.append(dict(question_id=getattr(r, id_col), answer=a, sql_or_sources=o["sql_or_sources"], refused=str(o["refused"]).lower()))
    pd.DataFrame(rows).to_csv(out_csv, index=False); return pd.DataFrame(rows)
