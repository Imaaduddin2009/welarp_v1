"""LAYER 2 - Insight & NLP: scope check -> offline intent templates -> sqlglot guard
-> execute on the MASKED db -> answer + SQL shown + freshness.  Notes/calls go to a text-search route (RAG)."""
import calendar, datetime as dt, os, re, json, time, yaml, duckdb, sqlglot, pandas as pd
from pathlib import Path
from sqlglot import exp

CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "config" / "semantic.yaml"))
L2DB = os.getenv("C360_L2_DB", "out/l2.duckdb")
AUDIT = os.getenv("C360_AUDIT_LOG", "out/audit_log.jsonl")
SNAP = CFG["snapshot_date"]; ALLOWED = set(CFG["tables"]); FORBID = set(CFG["forbidden_columns"])
MAX_ROWS = 1000
LEAD_MIN_CELL_SIZE = 10
DEFAULT_SUPPORTED_QUESTION = "Which product type has the highest total overdue amount?"
POLICY_CUES = re.compile(
    r"\b(policy|procedure|regulation|regulatory|permitted|allowed|must|should|"
    r"can we|can an agent|consent required|grace period|national do not call|"
    r"reduced[- ]payment|hardship programme|hardship program|programme lasts?|program lasts?|"
    r"before asking for payment|call attempts|how long|duration|portion|percentage|"
    r"how long.*programme|how long.*program|representative|cease[- ]communication|"
    r"adviser|advisor|authority|authori[sz]ation|insolvenc\w*|consumer proposal|trustee|"
    r"cease|no[- ]contact|stop[- ]contact|disput\w*|unfamiliar transaction|transaction.{0,30}unfamiliar|"
    r"what steps|before (?:asking|seeking|requesting).{0,20}(?:payment|money|pay))\b",
    re.I,
)


def _is_policy_question(question):
    return bool(POLICY_CUES.search(question))


def _normalize_question(question):
    normalized = re.sub(r"_+", " ", question.casefold())
    for canonical, aliases in CFG.get("question_synonyms", {}).items():
        for alias in sorted(aliases, key=len, reverse=True):
            normalized = re.sub(
                rf"(?<!\w){re.escape(alias.casefold())}(?!\w)",
                canonical,
                normalized,
            )
    return re.sub(r"\s+", " ", normalized).strip()


def _has_grouping_request(question):
    return bool(
        re.search(
            r"\b(by|per|across|among|between|group(?:ed)?(?:\s+by)?)\b",
            question,
        )
    )


# ------------------------------------------------------------------ 1. scope check (rules; deterministic and auditable)
RULES = [
  ("protected", r"\b(gender|male|female|marital|citizen\w*|immigra\w*|newcomer|household|dependants?|accessib\w*|disab\w*|vulnerab\w*|accent\w*|race|racial|ethnic\w*|religio\w*|nationality|sexual\w*|postal areas?|postal districts?|postal prefixes?|postal codes?|postcodes?|zip codes?|FSAs?|area codes?)\b",
   "it uses or tries to infer a protected attribute, which must not be used in collections decisions",
   "How many hardship customers have an account more than 30 days past due?"),
  ("pii", r"\b(phone numbers?|email addresses?|home address(es)?|date of birth|dob|sin|national id|full names?|names? of)\b",
   "personal data is masked in this system, so names and contact details cannot be returned",
   "How many customers consented to phone contact?"),
  ("bulk", r"\b(export|dump|download|extract)\b|\b(every|all)\b.*\b(customers?'? (data|records|details)|records)\b",
   "bulk export of customer data is not allowed", "How many customers are in each segment?"),
  ("write", r"\b(delete|drop|update|insert|alter|truncate|modify|overwrite|erase|purge|remove)\b",
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
    ql = _normalize_question(question)
    if re.search(r"\bhow long\b.*\b(programme|program)\b", ql):
        return "text"
    if re.search(
        r"\b(reduced[- ]payment|hardship) (?:programme|program)\b",
        ql,
    ) and re.search(r"\b(how long|duration|portion|percentage)\b", ql):
        return "text"
    if re.search(r"\bcase\s+cs-[A-Za-z0-9-]+\b", ql) and re.search(
        r"\b(call|transcript|summar(?:y|ize|ise))\b",
        ql,
    ):
        return "text"
    operational = re.search(
        r"\b(what share|what was|how many|count|total|volume|average|avg|mean|rate|compare|"
        r"percentage|percent|share|by .+|between .+ and .+|from .+ to .+)\b",
        ql,
    )
    policy_question = _is_policy_question(ql)
    if policy_question and not (operational and not re.search(r"\b(can we|can an agent|must|should)\b", ql)):
        return "text"
    if (
        re.search(
            r"\b(said|say|saying|told|mention\w*|reason|transcripts?|"
            r"talk\w* about|complain\w*|explain\w*|summar(?:y|ize|ise))\b",
            ql,
        )
        or (
            re.search(r"\bcase\s+CS-[A-Za-z0-9-]+\b", ql)
            and re.search(r"\b(call|transcript)\b", ql)
        )
    ) and not operational:
        return "text"
    return "sql"

# ------------------------------------------------------------------ 3. offline templates
def _n(ql, default):
    m = re.search(r"(\d+)\s*(day|days)", ql) or re.search(r"(?:last|past)\s+(\d+)", ql); return int(m.group(1)) if m else default
def _prod(ql): return next((p for p in ("card", "loan", "deposit") if p in ql), None)

def _month_bounds(question):
    ql = question.lower()
    months = {}
    for index in range(1, 13):
        months[calendar.month_name[index].lower()] = index
        months[calendar.month_abbr[index].lower()] = index
    month_pattern = "|".join(sorted(months, key=len, reverse=True))
    relative_match = re.search(
        r"\b(?:last|past|trailing|previous|preceding)\s+"
        r"(?:(\d+|a|an|one)\s*)?"
        r"(days?|weeks?|months?)\b",
        ql,
    )
    if relative_match:
        amount_text = relative_match.group(1)
        amount = 1 if amount_text in {None, "a", "an", "one"} else int(amount_text)
        unit = relative_match.group(2)
        end = dt.date.fromisoformat(SNAP) + dt.timedelta(days=1)
        if unit.startswith("month"):
            month_index = end.year * 12 + end.month - 1 - amount
            year, month_zero = divmod(month_index, 12)
            month = month_zero + 1
            day = min(end.day, calendar.monthrange(year, month)[1])
            start = dt.date(year, month, day)
        else:
            days = amount * (7 if unit.startswith("week") else 1)
            start = end - dt.timedelta(days=days)
        return start.isoformat(), end.isoformat()
    numeric_month = re.search(r"\b(20\d{2})[-/](0?[1-9]|1[0-2])\b", ql)
    if not numeric_month:
        numeric_month = re.search(r"\b(0?[1-9]|1[0-2])/(20\d{2})\b", ql)
        numeric_year, numeric_value = (
            (numeric_month.group(2), numeric_month.group(1))
            if numeric_month else (None, None)
        )
    else:
        numeric_year, numeric_value = numeric_month.group(1), numeric_month.group(2)
    if numeric_month:
        year, month = int(numeric_year), int(numeric_value)
        start = f"{year:04d}-{month:02d}-01"
        end_year, end_month = (year + 1, 1) if month == 12 else (year, month + 1)
        return start, f"{end_year:04d}-{end_month:02d}-01"
    year_match = re.search(r"\b(20\d{2})\b", ql)
    range_match = re.search(
        rf"\b({month_pattern})\s*(?:-|–|to|through)\s*({month_pattern})\s+"
        rf"{year_match.group(1)}\b",
        ql,
    ) if year_match else None
    year = int(year_match.group(1)) if year_match else None
    if range_match:
        first, last = months[range_match.group(1)], months[range_match.group(2)]
        start = f"{year:04d}-{first:02d}-01"
        end_year, end_month = (year + 1, 1) if last == 12 else (year, last + 1)
        return start, f"{end_year:04d}-{end_month:02d}-01"
    month_match = re.search(rf"\b({month_pattern})\s+(20\d{{2}})\b", ql)
    if not month_match:
        month_match = re.search(rf"\b(20\d{{2}})\s+({month_pattern})\b", ql)
        if month_match:
            year, month_name = int(month_match.group(1)), month_match.group(2)
        else:
            year = None
    else:
        month_name, year = month_match.group(1), int(month_match.group(2))
    if month_match:
        month = months[month_name]
        start = f"{year:04d}-{month:02d}-01"
        end_year, end_month = (year + 1, 1) if month == 12 else (year, month + 1)
        return start, f"{end_year:04d}-{end_month:02d}-01"
    year_match = re.search(r"\b(?:in|during|throughout|for)\s+(20\d{2})\b", ql)
    if not year_match:
        return None
    year = int(year_match.group(1))
    return f"{year:04d}-01-01", f"{year + 1:04d}-01-01"

def _months_in_question(question):
    months = {}
    for index in range(1, 13):
        months[calendar.month_name[index].lower()] = index
        months[calendar.month_abbr[index].lower()] = index
    pattern = "|".join(sorted(months, key=len, reverse=True))
    ql = question.lower()
    years = [int(year) for year in re.findall(r"\b(20\d{2})\b", ql)]
    points = []
    for match in re.finditer(rf"\b({pattern})\b", ql):
        nearby = ql[match.end():match.end() + 12]
        year_match = re.search(r"\b(20\d{2})\b", nearby)
        year = int(year_match.group(1)) if year_match else (
            years[0] if len(set(years)) == 1 else None
        )
        if year is not None:
            point = (year, months[match.group(1)])
            if point not in points:
                points.append(point)
    return points

def _explicit_day_comparison(question):
    months = {name.lower(): index for index, name in enumerate(calendar.month_name) if name}
    pattern = "|".join(months)
    ql = question.lower()
    match = re.search(
        rf"\bbetween\s+(\d{{1,2}})\s+and\s+(\d{{1,2}})\s+({pattern})\s+(20\d{{2}})\b",
        ql,
    )
    if not match:
        match = re.search(
            rf"\bfrom\s+(\d{{1,2}})(?:\s+(?:through|to)\s+|\s*-\s*)"
            rf"(\d{{1,2}})\s+({pattern})\s+(20\d{{2}})\b",
            ql,
        )
    if not match:
        return None
    first_day, last_day = int(match.group(1)), int(match.group(2))
    month, year = months[match.group(3)], int(match.group(4))
    start = dt.date(year, month, first_day)
    end_exclusive = dt.date(year, month, last_day) + dt.timedelta(days=1)
    duration = (end_exclusive - start).days
    prior_start = start - dt.timedelta(days=duration)
    month_label = calendar.month_abbr[month]
    current_label = f"{first_day}-{last_day} {month_label}"
    prior_label = f"{prior_start.day}-{(start - dt.timedelta(days=1)).day} {month_label}"
    return prior_start.isoformat(), start.isoformat(), end_exclusive.isoformat(), current_label, prior_label

def template_sql(question):
    ql = _normalize_question(question)
    if re.search(r"\bweather\b|\bforecast\b", ql):
        return None
    if re.search(
        r"\bprior(?:ly)? broken\b|\bpreviously (?:broken|failed)\b",
        ql,
    ) and re.search(r"rate|change|vary|differ", ql):
        return """
            SELECT least(prior_broken_count, 3) AS prior_broken,
                   round(100.0 * avg(CASE WHEN status = 'broken' THEN 1.0 ELSE 0.0 END), 1) AS broken_rate_pct,
                   count(*) AS promises
            FROM benchmark_promises
            WHERE status IN ('kept', 'partially_kept', 'broken') AND prior_broken_count IS NOT NULL
            GROUP BY 1 ORDER BY 1
        """
    if re.search(r"\b(average|avg|mean)\b", ql) and re.search(r"days past due|dpd", ql) and "queue" in ql:
        return """
            SELECT queue, round(avg(current_dpd), 1) AS avg_dpd, count(*) AS open_cases
            FROM benchmark_cases
            WHERE is_open AND queue IS NOT NULL AND current_dpd IS NOT NULL
            GROUP BY queue ORDER BY queue
        """
    if re.search(r"\bdisput\w*\b", ql) and re.search(
        r"between .+ and .+|compared with|before|preceding|previous", ql
    ):
        date_comparison = _explicit_day_comparison(question)
        if date_comparison:
            prior_start, current_start, current_end, current_label, prior_label = date_comparison
            return f"""
            SELECT CASE WHEN contact_ts >= TIMESTAMP '{current_start}' THEN '{current_label}'
                      ELSE '{prior_label}' END AS period,
                 count(*) AS disputes
            FROM benchmark_contacts
            WHERE dispute_raised_flag
            AND contact_ts >= TIMESTAMP '{prior_start}'
            AND contact_ts < TIMESTAMP '{current_end}'
            AND system_event_type = ''
            GROUP BY period ORDER BY min(contact_ts)
            """
    if re.search(r"\bdisput\w*\b", ql) and re.search(
        r"rate|share|proportion|percentage", ql
    ) and "channel" in ql:
        bounds = _month_bounds(question)
        date_filter = ""
        if bounds:
            start, end = bounds
            date_filter = (
                f"AND contact_ts >= TIMESTAMP '{start}' "
                f"AND contact_ts < TIMESTAMP '{end}'"
            )
        return f"""
            SELECT channel,
                   round(100.0 * avg(CASE WHEN dispute_raised_flag THEN 1.0 ELSE 0.0 END), 1) AS dispute_rate_pct,
                   count(*) AS contacts
            FROM benchmark_contacts
            WHERE channel IS NOT NULL AND system_event_type = ''
              {date_filter}
            GROUP BY channel ORDER BY channel
        """
    if "broken" in ql and "site" in ql:
        bounds = _month_bounds(question)
        if bounds:
            start, end = bounds
            end_exclusive = min(
              dt.date.fromisoformat(end),
              dt.date.fromisoformat(SNAP) + dt.timedelta(days=1),
            ).isoformat()
        else:
            start = f"{SNAP[:4]}-01-01"
            end_exclusive = (
                dt.date.fromisoformat(SNAP) + dt.timedelta(days=1)
            ).isoformat()
        return """
            SELECT agent_site AS site,
                 round(100.0 * avg(CASE WHEN status = 'broken' THEN 1.0 ELSE 0.0 END), 1) AS broken_rate_pct,
                 count(*) AS promises
            FROM benchmark_promises
            WHERE due_date >= DATE '{start}' AND due_date < DATE '{end_exclusive}'
            AND status IN ('kept', 'partially_kept', 'broken')
            GROUP BY agent_site ORDER BY agent_site
        """.format(start=start, end_exclusive=end_exclusive)
    if re.search(r"\b(outside|beyond)\b.*\b(permitted|allowed)\b.*\bhours?\b", ql) and "province" in ql:
        bounds = _month_bounds(question)
        if bounds:
            month_start = bounds[0]
        else:
            month_start = SNAP[:7] + "-01"
        return f"""
            SELECT province_code, out_of_hours_pct, calls
            FROM benchmark_out_of_hours_by_province
            WHERE month = DATE '{month_start}'
            ORDER BY province_code
        """
    if "right-party" in ql or "right party" in ql:
        bounds = _month_bounds(question)
        if bounds:
            start, end = bounds
        else:
            start = SNAP[:7] + "-01"
            end = (
                f"{int(SNAP[:4]) + 1}-01-01" if SNAP[5:7] == "12"
                else f"{SNAP[:5]}{int(SNAP[5:7]) + 1:02d}-01"
            )
        snapshot_end = min(dt.date.fromisoformat(end), dt.date.fromisoformat(SNAP)).isoformat()
        return f"""
            SELECT contact_hour_local,
                 round(100.0 * avg(CASE WHEN rpc_flag THEN 1.0 ELSE 0.0 END), 1) AS rpc_rate_pct,
                 count(*) AS attempts
            FROM benchmark_contacts
            WHERE channel = 'call' AND direction = 'outbound'
            AND contact_ts >= TIMESTAMP '{start}'
            AND contact_ts < TIMESTAMP '{snapshot_end}'
            AND system_event_type = ''
            GROUP BY contact_hour_local ORDER BY contact_hour_local
        """
    if "active card accounts" in ql and re.search(r"1 or more|1\+|at least one|one or more", ql) and "product" in ql:
        bounds = _month_bounds(question)
        snapshot = bounds[0] if bounds else SNAP
        return f"""
            SELECT product_code,
                 round(100.0 * avg(CASE WHEN dpd >= 1 THEN 1.0 ELSE 0.0 END), 2) AS delinquency_rate_pct,
                 count(*) AS accounts
            FROM benchmark_card_accounts
            WHERE snapshot_date = DATE '{snapshot}'
            AND account_status IN ('active', 'blocked') AND dpd IS NOT NULL
            GROUP BY product_code ORDER BY product_code
        """
    if re.search(r"rolled|roll forward|moved|passed|crossed|exceeded", ql) and re.search(
        r"worse bucket|higher delinquency|more than 30|above 30|past 30|pass(?:ed)? 30",
        ql,
    ):
        month_points = _months_in_question(question)
        if len(month_points) < 2:
            return None
        start_year, start_month = month_points[0]
        end_year, end_month = month_points[-1]
        start_date = dt.date(start_year, start_month, calendar.monthrange(start_year, start_month)[1])
        end_date = dt.date(end_year, end_month, calendar.monthrange(end_year, end_month)[1])
        return f"""
            WITH start_snapshot AS (
              SELECT account_key, product_type, dpd
              FROM benchmark_account_history
              WHERE snapshot_date = DATE '{start_date.isoformat()}' AND dpd BETWEEN 1 AND 30
            ),
            end_snapshot AS (
              SELECT account_key, dpd
              FROM benchmark_account_history
              WHERE snapshot_date = DATE '{end_date.isoformat()}' AND dpd IS NOT NULL
            )
            SELECT s.product_type,
                   round(100.0 * avg(CASE WHEN e.dpd > 30 THEN 1.0 ELSE 0.0 END), 1) AS roll_rate_pct,
                   count(*) AS accounts
            FROM start_snapshot s JOIN end_snapshot e USING (account_key)
            GROUP BY s.product_type ORDER BY s.product_type
        """
    if "cure" in ql and re.search(r"average|avg|mean", ql) and re.search(r"days|time", ql):
        dimension = "queue" if "queue" in ql else "primary_product" if "product" in ql else None
        if dimension:
            bounds = _month_bounds(question)
            date_filter = ""
            if bounds:
                date_filter = (
                    f"AND opened_date >= DATE '{bounds[0]}' "
                    f"AND opened_date < DATE '{bounds[1]}'"
                )
            return f"""
                SELECT {dimension}, round(avg(days_to_cure), 1) AS average_days_to_cure,
                       count(*) AS cured_cases
                FROM benchmark_cases
                WHERE cure_flag AND days_to_cure IS NOT NULL {date_filter}
                GROUP BY {dimension} ORDER BY {dimension}
            """
    if "cure" in ql and re.search(r"rate|share|fraction|proportion|percentage", ql):
        dimension = "queue" if "queue" in ql else "primary_product" if "product" in ql else None
        if dimension:
            days = _n(ql, 30)
            bounds = _month_bounds(question)
            date_filter = "WHERE true"
            if bounds:
                date_filter += (
                    f" AND opened_date >= DATE '{bounds[0]}' "
                    f"AND opened_date < DATE '{bounds[1]}'"
                )
            return f"""
                SELECT {dimension},
                       round(100.0 * avg(CASE WHEN cure_flag AND days_to_cure <= {days}
                           THEN 1.0 ELSE 0.0 END), 1) AS cured_within_{days}_days_pct,
                       count(*) AS cases
                FROM benchmark_cases
                {date_filter}
                GROUP BY {dimension} ORDER BY {dimension}
            """
    if re.search(r"\b(random|randomi[sz]ed|randomi[sz]ation)\b", ql) and re.search(
        r"\b(promise|strategy|test cell|assignment cell)\b",
        ql,
    ):
        requested_cells = re.findall(
            r"(?<!\w)([a-z][a-z0-9]*_[a-z0-9]+)(?!\w)",
            question,
            re.I,
        )
        requested_cells.extend(
            re.findall(
                r"\b([a-z][a-z0-9]*)\s+(?=(?:strategy|arm)\b)",
                question,
                re.I,
            )
        )
        unique_cells = {}
        for cell in requested_cells:
            unique_cells.setdefault(cell.casefold(), cell)
        requested_cells = list(unique_cells.values())
        cell_filter = (
            "AND p.test_cell IN ("
            + ", ".join(f"'{cell}'" for cell in requested_cells)
            + ")"
            if requested_cells
            else ""
        )
        bounds = _month_bounds(question)
        date_filter = ""
        if bounds:
            date_filter += (
                f"AND p.due_date <= DATE '{SNAP}' "
                f"AND p.due_date >= DATE '{bounds[0]}' "
                f"AND p.due_date < DATE '{bounds[1]}'"
            )
        outcome = "broken" if re.search(r"\b(broken|failed)\b", ql) else "kept"
        return """
            SELECT p.test_cell,
                   round(100.0 * avg(CASE WHEN p.status = '{outcome}' THEN 1.0 ELSE 0.0 END), 1) AS {outcome}_pct,
                   count(*) AS promises
            FROM benchmark_promises p
            WHERE p.assignment_method = 'random'
              {cell_filter}
              {date_filter}
              AND p.status IN ('kept', 'partially_kept', 'broken')
            GROUP BY p.test_cell ORDER BY p.test_cell
        """.format(
            cell_filter=cell_filter,
            date_filter=date_filter,
            outcome=outcome,
        )
    if "hardship" in ql and re.search(r"rate|share|proportion|frequency", ql):
        dimension = (
            "current_bucket" if "bucket" in ql
            else "queue" if "queue" in ql
            else "primary_product" if "product" in ql
            else None
        )
        if dimension:
            bounds = _month_bounds(question)
            date_filter = (
                f"AND opened_date >= DATE '{bounds[0]}' "
                f"AND opened_date < DATE '{bounds[1]}'"
                if bounds else ""
            )
            open_filter = "AND is_open" if re.search(r"\bopen\b", ql) else ""
            return f"""
                SELECT {dimension}, round(100.0 * avg(CASE WHEN hardship_flag THEN 1.0 ELSE 0.0 END), 1) AS hardship_rate_pct,
                       count(*) AS cases
                FROM benchmark_cases
                WHERE true {date_filter} {open_filter}
                GROUP BY {dimension} ORDER BY {dimension}
            """
    if "cease" in ql and re.search(r"rate|share|proportion|percentage", ql) and "hour" in ql:
        return """
            SELECT contact_hour_local,
                   round(100.0 * avg(CASE WHEN cease_request_flag THEN 1.0 ELSE 0.0 END), 1) AS cease_request_rate_pct,
                   count(*) AS outbound_calls
            FROM benchmark_contacts
            WHERE channel = 'call' AND direction = 'outbound' AND system_event_type = ''
            GROUP BY contact_hour_local ORDER BY contact_hour_local
        """
    if (
        "promise" in ql
        and re.search(r"\b(rate|share|percentage)\b", ql)
        and re.search(r"\bdue\b", ql)
        and re.search(r"\b(each|every|per|by|monthly|months?)\b", ql)
    ):
        year_match = re.search(r"\b(20\d{2})\b", ql)
        year = int(year_match.group(1)) if year_match else int(SNAP[:4])
        return f"""
            SELECT strftime(due_date, '%Y-%m') AS period,
                   round(100.0 * avg(CASE WHEN status = 'kept' THEN 1.0 ELSE 0.0 END), 1) AS promise_kept_rate_pct,
                   count(*) AS promises
            FROM benchmark_promises
            WHERE due_date >= DATE '{year}-01-01' AND due_date < DATE '{year + 1}-01-01'
              AND due_date <= DATE '{SNAP}'
              AND status IN ('kept', 'partially_kept', 'broken')
            GROUP BY period ORDER BY period
        """
    if "promise-kept" in ql and re.search(r"\b(rate|share|percentage)\b", ql) and (
        "due in" in ql or "due for" in ql or "promises" in ql or "commitments" in ql
    ):
        bounds = _month_bounds(question)
        if bounds:
            start, end = bounds
            return f"""
                SELECT round(100.0 * avg(CASE WHEN status = 'kept' THEN 1.0 ELSE 0.0 END), 1) AS promise_kept_rate_pct
                FROM benchmark_promises
                WHERE due_date >= DATE '{start}' AND due_date < DATE '{end}'
                  AND due_date <= DATE '{SNAP}'
                  AND status IN ('kept', 'partially_kept', 'broken')
            """
    if (
        "promise" in ql
        and re.search(r"\b(rate|percentage|share)\b", ql)
        and re.search(r"\bdue\b", ql)
    ):
        bounds = _month_bounds(question)
        if bounds:
            start, end = bounds
            return f"""
                SELECT round(100.0 * avg(CASE WHEN status = 'kept' THEN 1.0 ELSE 0.0 END), 1)
                       AS promise_kept_rate_pct
                FROM benchmark_promises
                WHERE due_date >= DATE '{start}' AND due_date < DATE '{end}'
                  AND due_date <= DATE '{SNAP}'
                  AND status IN ('kept', 'partially_kept', 'broken')
            """
    if re.search(r"\b(open|active)\b", ql) and "case" in ql and "bucket" in ql:
        return """
            SELECT current_bucket, count(*) AS open_cases
            FROM benchmark_cases
            WHERE is_open
            GROUP BY current_bucket ORDER BY
                CASE current_bucket WHEN '1-30' THEN 1 WHEN '31-60' THEN 2
                    WHEN '61-90' THEN 3 WHEN '91-120' THEN 4 WHEN '121-150' THEN 5
                    WHEN '151-180' THEN 6 WHEN '180+' THEN 7 ELSE 0 END
        """
    if "broken" in ql and "rate" in ql and "previously" in ql:
        return """
            SELECT least(prior_broken_count, 3) AS prior_broken,
                   round(100.0 * avg(CASE WHEN status = 'broken' THEN 1.0 ELSE 0.0 END), 1) AS broken_rate_pct,
                   count(*) AS promises
            FROM benchmark_promises
            WHERE status IN ('kept', 'partially_kept', 'broken') AND prior_broken_count IS NOT NULL
            GROUP BY 1 ORDER BY 1
        """
    if re.search(r"(broken|failed|missed|unkept) promise", ql) or (
        "promise" in ql and re.search(r"\b(broke|failed|missed)\b", ql)
    ):
        n = _n(ql, 30); base = f"FROM promises WHERE status = 'BROKEN' AND promised_date >= DATE '{SNAP}' - INTERVAL {n} DAY"
        if re.search(r"\b(which|list|who)\b", ql): return f"SELECT DISTINCT golden_id {base} ORDER BY golden_id"
        return f"SELECT count(DISTINCT golden_id) AS customers_with_broken_promise {base}"
    if re.search(r"(more than|over|at least|>)\s*(\d+).*broken|broken.*(more than|over)\s*(\d+)", ql):
        k = int(re.findall(r"\d+", ql)[0]) if re.findall(r"\d+", ql) else 1
        return f"SELECT golden_id, count(*) AS broken_promises FROM promises WHERE status='BROKEN' AND promised_date >= DATE '{SNAP}' - INTERVAL 90 DAY GROUP BY 1 HAVING count(*) > {k} ORDER BY 2 DESC"
    if (
        re.search(r"\b(kept|honoured|honored|fulfilled)\b", ql)
        and "promise" in ql
        and re.search(r"\b(rate|share|percentage)\b", ql)
        and not _has_grouping_request(ql)
    ):
        return "SELECT round(100.0 * count(*) FILTER (WHERE status = 'KEPT') / count(*), 1) AS promise_kept_rate_pct FROM promises WHERE status IN ('KEPT','BROKEN')"
    if (
        "hardship" in ql
        and re.search(r"overdue|past due|delinquen", ql)
        and re.search(r"\b(count|how many|number|customers?)\b", ql)
    ):
        return "SELECT count(DISTINCT c.golden_id) AS hardship_customers_delinquent FROM customers c JOIN accounts a USING (golden_id) WHERE c.hardship_flag AND a.days_past_due > 0"
    if "hardship" in ql and re.search(r"\b(count|how many|number)\b", ql):
        return "SELECT count(*) AS hardship_customers FROM customers WHERE hardship_flag"
    if re.search(r"total overdue|overdue amount|overdue.*by product|most overdue|highest.*overdue", ql):
        return "SELECT product_type, round(sum(overdue_amount), 2) AS total_overdue FROM accounts GROUP BY 1 ORDER BY 2 DESC"
    if re.search(r"average days past due|avg days past due|average dpd", ql):
        p = _prod(ql); w = f"WHERE product_type = '{p}' AND days_past_due > 0" if p else "WHERE days_past_due > 0"
        return f"SELECT round(avg(days_past_due), 1) AS avg_days_past_due FROM accounts {w}"
    if ("delinquent" in ql or "past due" in ql) and re.search(
        r"\b(count|how many|number|rate|share|percentage|average|avg)\b", ql
    ):
        p = _prod(ql)
        if re.match(r"\s*how many", ql) and p: return f"SELECT count(*) AS delinquent_{p}_accounts FROM accounts WHERE product_type='{p}' AND days_past_due > 0"
        return "SELECT product_type, count(*) AS delinquent_accounts FROM accounts WHERE days_past_due > 0 GROUP BY 1 ORDER BY 2 DESC"
    if re.search(r"agent", ql) and re.search(r"top|best|most|kept", ql):
        return "SELECT c.agent_id, count(*) FILTER (WHERE p.status='KEPT') AS promises_kept, count(*) AS promises_total FROM promises p JOIN cases c USING (case_id) GROUP BY 1 ORDER BY 2 DESC LIMIT 5"
    if "consent" in ql:
        ch = next((c for c in ("sms", "email", "phone") if c in ql), "phone")
        return f"SELECT count(*) AS customers_with_{ch}_consent FROM customers WHERE consent_{ch}"
    if (
        ("channel" in ql or "contact" in ql)
        and re.search(r"\b(rate|share|percentage|count|how many|number|volume)\b", ql)
    ):
        return "SELECT channel, count(*) AS contacts, round(100.0*count(*) FILTER (WHERE outcome IN ('answered','promise_made'))/count(*),1) AS answer_rate_pct FROM contacts GROUP BY 1 ORDER BY 2 DESC"
    if (
        "case" in ql
        and ("stage" in ql or "status" in ql)
        and re.search(r"\b(count|how many|number|total|volume)\b", ql)
    ):
        return f"SELECT {'stage' if 'stage' in ql else 'status'}, count(*) AS cases FROM cases GROUP BY 1 ORDER BY 2 DESC"
    if "segment" in ql and re.search(r"\b(count|how many|number|total)\b", ql):
        return "SELECT segment, count(*) AS customers FROM customers GROUP BY 1 ORDER BY 2 DESC"
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
        if c.name.lower() in FORBID:
            raise Blocked(f"column '{c.name}' is protected or personal data")
    if any(
        column.name.lower() == "account_key"
        for projection in tree.expressions
        for column in projection.find_all(exp.Column)
    ):
        raise Blocked("account_key is an internal temporal join token and cannot be returned")
    group_clause = tree.args.get("group")
    if group_clause and any(
        column.name.lower() == "account_key"
        for expression in group_clause.expressions
        for column in expression.find_all(exp.Column)
    ):
        raise Blocked("account_key is an internal temporal join token and cannot be grouped")
    if any(
        isinstance(projection, exp.Star)
        for projection in tree.expressions
    ) and any(table.name == "benchmark_account_history" for table in tree.find_all(exp.Table)):
        raise Blocked("internal account history join tokens cannot be returned")
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
            if source.name.startswith("benchmark_"):
                cell_size = exp.Count(this=exp.Star())
            else:
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
        ag = re.sub(r"[^A-Za-z0-9_]", "", user["agent_id"])
        def scope(node):
            if isinstance(node, exp.Table) and node.name in ALLOWED and node.name not in ctes and node.name != "pipeline_run":
                cond = f"agent_id = '{ag}'" if node.name == "cases" else f"golden_id IN (SELECT golden_id FROM cases WHERE agent_id = '{ag}')"
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
    def search(self, q, k=5, min_sim=0.08, user=None, policy_only=False):
        ql = re.sub(r"[-_]+", " ", q.casefold())
        matched_expansions = [
            expansion
            for phrase, expansion in CFG.get("retrieval_expansions", {}).items()
            if re.sub(r"[-_]+", " ", phrase.casefold()) in ql
        ]
        queries = (
            [(q, 0.25), *((expansion, 2.0) for expansion in matched_expansions)]
            if matched_expansions else [(q, 1.0)]
        )
        scores = [
            weight * (self.X @ self.vec.transform([query]).T).toarray().ravel()
            for query, weight in queries
        ]
        sims = sum(scores[1:], scores[0])
        ranked = sims.copy()
        precedence = CFG.get("policy_precedence", [])
        official_ids = tuple(doc_id + " " for doc_id in precedence)
        official = self.df.source_type.eq("policy").to_numpy() & self.df.source_citation.astype(
            str
        ).str.startswith(official_ids).to_numpy()
        ranked += official * (sims >= 0.25) * 0.6
        if not re.search(r"\b(fr|french|fran[cç]ais)\b", ql):
            ranked -= (
                self.df.source_citation.astype(str).str.contains("-FR", regex=False)
            ).to_numpy() * 0.1
        order = ranked.argsort()[::-1]
        df = self.df
        if user and user.get("role") == "agent":
            con = duckdb.connect(L2DB, read_only=True)
            mine = set(con.execute(f"SELECT DISTINCT golden_id FROM cases WHERE agent_id = '{re.sub(r'[^A-Za-z0-9_]', '', user['agent_id'])}'").fetchdf().golden_id); con.close()
            order = [i for i in order if df.source_type.iloc[i] == "policy" or df.golden_id.iloc[i] in mine]
        if policy_only:
            order = [i for i in order if df.source_type.iloc[i] == "policy"]
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


def _clean_policy_text(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return re.sub(r"[ \t]+\n", "\n", text).strip()


def _policy_hit_covers_requested_detail(question, text):
    asks_for_verification = re.search(
        r"\b(verify|verification|checks?|proof|evidence|confirm.{0,20}(authority|adviser|advisor|representative))\b",
        question,
        re.I,
    )
    if asks_for_verification and not re.search(
        r"\b(verify|verification|check|proof|evidence|authori[sz]ed|authority)\b",
        text,
        re.I,
    ):
        return False
    return True


def _summarize_call(text):
    sentences = [
        re.sub(r"\s+", " ", part).strip()
        for part in re.split(r"(?<=[.!?])\s+", text)
        if part.strip()
    ]
    issue = next(
        (part for part in sentences if re.search(
            r"\b(sick|ill|illness|medical|hours? a week|days a week|only getting|income|"
            r"laid off|job loss|lost (?:my )?(?:job|overtime)|making less|my EI|"
            r"employment insurance|can't afford|cannot afford|not able to pay|work since|"
            r"(?:hours?|shifts?).{0,30}(?:reduced|cut|fewer)|"
            r"(?:reduced|cut|fewer).{0,30}(?:hours?|shifts?))\b",
            part,
            re.I,
        )),
        None,
    )
    payment = next(
        (part for part in sentences if re.search(
            r"\b(i can pay|i'll pay|i will pay|payment of)\b", part, re.I
        )),
        None,
    )
    asked_to_stop = bool(re.search(r"\b(stop calling|stop contact|stop contacting|cease contact)\b", text, re.I))
    next_step = "stop contact" if asked_to_stop else (
        "hardship plan review" if re.search(r"\bhardship|reduced payment\b", text, re.I) else None
    )
    details = []
    if issue:
        details.append(f"Customer said: {issue}")
    if payment:
        details.append(f"Customer agreed: {payment}")
    if asked_to_stop:
        details.append("Customer asked to stop contact")
    if next_step:
        details.append(f"Next step: {next_step}")
    return "; ".join(details) if details else "No concise customer statement or commitment was identified in the recorded call."


def _period_order_key(label, question):
    label_text = str(label).casefold()
    question_text = question.casefold()
    tokens = re.findall(r"\d{1,4}|[a-z]+", label_text)
    month_names = {
        abbreviation.casefold(): name.casefold()
        for abbreviation, name in zip(calendar.month_abbr[1:], calendar.month_name[1:])
    }
    positions = []
    for token in tokens:
        variants = {token}
        if token in month_names:
            variants.add(month_names[token])
        for variant in variants:
            match = re.search(rf"(?<!\w){re.escape(variant)}(?!\w)", question_text)
            if match:
                positions.append(match.start())
    return min(positions, default=len(question_text))


def _format_answer(table, question):
    if table.empty:
        return "No matching records."
    if table.shape == (1, 1):
        return str(table.iat[0, 0])

    period_column = next(
        (
            column for column in table.columns
            if str(column).casefold() in {"period", "time_period", "date_range"}
        ),
        None,
    )
    if period_column and re.search(
        r"\b(compare|compared|versus|vs\.?|against|prior|previous|preceding|before|after)\b",
        question,
        re.I,
    ):
        table = table.assign(
            _question_period_order=table[period_column].map(
                lambda label: _period_order_key(label, question)
            )
        ).sort_values("_question_period_order", kind="stable").drop(
            columns="_question_period_order"
        )
    return table.to_csv(index=False).strip()


# ------------------------------------------------------------------ orchestrator
class QA:
    def __init__(self, user=None):
        self.user = user or dict(id="analyst1", role="analyst"); self._search = None
    def _fresh(self):
        c = duckdb.connect(L2DB, read_only=True); r = c.execute("SELECT run_ts::VARCHAR, data_snapshot_date::VARCHAR FROM pipeline_run").fetchone(); c.close()
        return f"data as of {r[1]}; pipeline refreshed {str(r[0])[:19]}"
    def _log(self, out):
        Path(AUDIT).parent.mkdir(parents=True, exist_ok=True)
        with open(AUDIT, "a") as f: f.write(json.dumps(dict(ts=time.time(), user=self.user, **{k: out[k] for k in ("question", "route", "sql_or_sources", "refused", "reason")}), default=str) + "\n")
    def _refuse(self, q, reason, suggestion=None, rt="none"):
        suggestion = suggestion or DEFAULT_SUPPORTED_QUESTION
        out = dict(question=q, answer=f"I can't answer that: {reason}." + (f" Try instead: \"{suggestion}\"" if suggestion else ""), sql_or_sources="", refused=True, reason=reason, route=rt, freshness=self._fresh(), table=None)
        self._log(out); return out
    def ask(self, q):
        sc = scope_check(q, self.user)
        if not sc["ok"]: return self._refuse(q, sc["reason"], sc["suggestion"])
        rt = route(q)
        if rt == "text":
            case_match = re.search(r"\bcase\s+(CS-[A-Za-z0-9-]+)", q, re.I)
            if case_match and re.search(
                r"\b(call|transcript|said|agree|summar(?:y|ize|ise))\b",
                q,
                re.I,
            ):
                case_id = case_match.group(1)
                con = duckdb.connect(L2DB, read_only=True)
                try:
                    row = con.execute("""
                        SELECT transcript_id, call_start_ts, transcript_text_masked
                        FROM transcripts
                        WHERE case_id = ?
                        ORDER BY call_start_ts DESC, transcript_id DESC
                        LIMIT 1
                    """, [case_id]).fetchone()
                finally:
                    con.close()
                if row is None:
                    return self._refuse(q, "no recorded call was found for that case", None, "text")
                transcript_id, call_start_ts, transcript_text = row
                citation = f"transcript:{transcript_id}"
                out = dict(
                    question=q,
                    answer=_summarize_call(transcript_text),
                    sql_or_sources=citation,
                    refused=False,
                    reason="",
                    route="text",
                    freshness=self._fresh(),
                    table=pd.DataFrame([{
                        "transcript_id": transcript_id,
                        "call_start_ts": call_start_ts,
                        "summary": _summarize_call(transcript_text),
                    }]),
                )
                self._log(out)
                return out
            policy_intent = _is_policy_question(q)
            self._search = self._search or NoteSearch()
            hits = self._search.search(
                q,
                k=50 if policy_intent else 5,
                min_sim=0.12 if policy_intent else 0.08,
                user=self.user,
                policy_only=policy_intent,
            )
            if not hits: return self._refuse(q, "no relevant notes were found, so I can't answer this reliably", None, "text")
            if policy_intent:
                hit = hits[0]
                if not _policy_hit_covers_requested_detail(q, hit["text"]):
                    return self._refuse(
                        q,
                        "the current policy source does not specify the requested verification steps",
                        rt="text",
                    )
                answer = _clean_policy_text(hit["text"])
                citation = hit["source_citation"]
                out = dict(
                    question=q,
                    answer=answer,
                    sql_or_sources=f"policy:{citation}",
                    refused=False,
                    reason="",
                    route="text",
                    freshness=self._fresh(),
                    table=pd.DataFrame([hit]),
                )
                self._log(out)
                return out
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
        sql = template_sql(q); src = "template"
        if sql is None: return self._refuse(q, "I can't answer this reliably from the approved collections tables", "Which product type has the highest total overdue amount?", "sql")
        try:
            safe = guard(sql, self.user)
        except Blocked as b: return self._refuse(q, f"the generated query was blocked by the safety checker ({b})", None, "sql")
        except Exception as e: return self._refuse(q, f"the generated query could not be validated ({type(e).__name__})", None, "sql")
        try: df = run_sql(safe)
        except Exception as e:
            return self._refuse(q, f"the query failed ({type(e).__name__})", None, "sql")
        ans = _format_answer(df, q)
        out = dict(question=q, answer=ans, sql_or_sources=safe, refused=False, reason="", route="sql", freshness=self._fresh(), table=df, via=src)
        self._log(out); return out

def run_benchmark(questions_csv, out_csv, id_col="question_id", q_col="question", user=None):
    """Writes the 4 columns the organisers asked for. Never hand-edit the result."""
    qa = QA(user); df = pd.read_csv(questions_csv); rows = []
    for r in df.itertuples():
        o = qa.ask(getattr(r, q_col))
        if o["refused"] or o["route"] != "sql" or o["table"] is None:
            a = o["answer"]
        else:
            a = _format_answer(o["table"], o["question"])
        rows.append(dict(question_id=getattr(r, id_col), answer=a, sql_or_sources=o["sql_or_sources"], refused=str(o["refused"]).lower()))
    pd.DataFrame(rows).to_csv(out_csv, index=False); return pd.DataFrame(rows)
