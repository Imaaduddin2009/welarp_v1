"""Diagnose Layer 4 label completeness and cohort drift without changing labels."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import layer4

DEFAULT_CUTOFF = "2026-09-28"
RESOLVED_STATUSES = {"KEPT", "BROKEN"}


def _window_promises(promises: pd.DataFrame, as_of: pd.Timestamp, horizon: int) -> pd.DataFrame:
    end = as_of + pd.Timedelta(days=horizon)
    rows = promises.loc[
        promises["promised_date"].ge(as_of) & promises["promised_date"].lt(end)
    ].copy()
    status_date = pd.to_datetime(rows["status_date"], errors="coerce")
    status = rows["status"].astype("string").str.upper()
    rows["could_be_censored"] = (
        ~status.isin(RESOLVED_STATUSES) | status_date.isna() | status_date.ge(end)
    )
    return rows


def build_report(
    panel: pd.DataFrame,
    promises: pd.DataFrame,
    dates: list[str],
    cutoff: str = DEFAULT_CUTOFF,
    horizon: int = 30,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Summarize the exact model panel and future promise population by snapshot."""
    parsed_cutoff = pd.Timestamp(cutoff)
    panel = panel.copy()
    panel["as_of"] = pd.to_datetime(panel["as_of"])
    promises = promises.copy()
    promises["promised_date"] = pd.to_datetime(promises["promised_date"])
    promises["status_date"] = pd.to_datetime(promises["status_date"], errors="coerce")

    reports = []
    prior_customers: set[str] = set()
    previous_date = None
    for date in sorted(set(dates)):
        as_of = pd.Timestamp(date)
        rows = panel.loc[panel["as_of"].eq(as_of)]
        customers = set(rows["golden_id"].dropna().astype(str))
        kept = int(rows["label_kept_30d"].eq(1).sum())
        broken = int(rows["label_kept_30d"].eq(0).sum())
        future_promises = _window_promises(promises, as_of, horizon)
        unresolved = int(future_promises["could_be_censored"].sum())
        promise_count = len(future_promises)
        new_customers = len(customers - prior_customers) if previous_date else len(customers)
        reports.append({
            "as_of": date,
            "rows": len(rows),
            "kept": kept,
            "broken": broken,
            "kept_rate": kept / len(rows) if len(rows) else None,
            "complete": as_of + pd.Timedelta(days=horizon) <= parsed_cutoff,
            "unresolved": unresolved,
            "unresolved_share": unresolved / promise_count if promise_count else None,
            "promise_makers": int(future_promises["golden_id"].nunique()),
            "new_labeled_customers": new_customers,
            "previous_snapshot": previous_date or "—",
        })
        prior_customers = customers
        previous_date = date

    observed_rates = [row["kept_rate"] for row in reports if row["kept_rate"] is not None]
    details: dict[str, object] = {
        "cutoff": cutoff,
        "horizon": horizon,
        "snapshot_count": len(reports),
        "kept_rate_range": (
            (min(observed_rates), max(observed_rates)) if observed_rates else None
        ),
        "first_to_last_kept_rate_change": (
            observed_rates[-1] - observed_rates[0] if len(observed_rates) > 1 else None
        ),
    }
    return reports, details


def _format_percent(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.1%}"


def print_table(reports: list[dict[str, object]]) -> None:
    headers = [
        "as_of", "rows", "kept", "broken", "kept rate", "30d complete",
        "unresolved promises (n/share)", "labeled customers new vs prior",
    ]
    output = ["| " + " | ".join(headers) + " |",
              "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in reports:
        unresolved = (
            f"{row['unresolved']:,} ({_format_percent(row['unresolved_share'])})"
            if row["unresolved_share"] is not None else "n/a"
        )
        prior = row["previous_snapshot"]
        new_customers = row["new_labeled_customers"]
        cohort_change = (
            f"{new_customers:,} / {prior}" if isinstance(new_customers, int)
            else "n/a"
        )
        values = [
            str(row["as_of"]),
            f"{row['rows']:,}" if isinstance(row["rows"], int) else "n/a",
            f"{row['kept']:,}" if isinstance(row["kept"], int) else "n/a",
            f"{row['broken']:,}" if isinstance(row["broken"], int) else "n/a",
            _format_percent(row["kept_rate"]),
            "yes" if row["complete"] else "no",
            unresolved,
            cohort_change,
        ]
        output.append("| " + " | ".join(values) + " |")
    print("\n".join(output))


def _from_persisted_scores(path: Path, dates: list[str], cutoff: str) -> list[dict[str, object]]:
    """Use only persisted holdout labels when source promises are unavailable."""
    scores = pd.read_csv(path, parse_dates=["as_of"])
    prior_customers: set[str] = set()
    previous_date = None
    previous_snapshot_has_rows = False
    reports = []
    for date in sorted(set(dates)):
        as_of = pd.Timestamp(date)
        rows = scores.loc[scores["as_of"].eq(as_of)]
        customers = set(rows["golden_id"].dropna().astype(str))
        n = len(rows)
        kept = int(rows["label_kept_30d"].eq(1).sum())
        broken = int(rows["label_kept_30d"].eq(0).sum())
        reports.append({
            "as_of": date,
            "rows": n if n else None,
            "kept": kept if n else None,
            "broken": broken if n else None,
            "kept_rate": kept / n if n else None,
            "complete": as_of + pd.Timedelta(days=30) <= pd.Timestamp(cutoff),
            "unresolved": None,
            "unresolved_share": None,
            "promise_makers": None,
            "new_labeled_customers": (
                len(customers - prior_customers)
                if n and previous_snapshot_has_rows else None
            ),
            "previous_snapshot": previous_date or "—",
        })
        if n:
            prior_customers = customers
        previous_snapshot_has_rows = bool(n)
        previous_date = date
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cutoff", default=DEFAULT_CUTOFF)
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--l2-db", type=Path, default=layer4.L2DB)
    parser.add_argument(
        "--scores", type=Path, default=ROOT / "out" / "layer4_scored_actions.csv"
    )
    args = parser.parse_args()
    if args.l2_db.is_file():
        dates = layer4.discover_as_of_dates(args.horizon)
        panel = layer4.action_panel(dates, args.horizon)
        with duckdb.connect(str(args.l2_db), read_only=True) as con:
            promises = con.execute("""
                SELECT golden_id, promised_date, status_date, status
                FROM promises
            """).fetchdf()
        reports, details = build_report(panel, promises, dates, args.cutoff, args.horizon)
        print_table(reports)
        print(
            f"\nObserved kept-rate range: {_format_percent(details['kept_rate_range'][0])}"
            f"–{_format_percent(details['kept_rate_range'][1])}"
            if details["kept_rate_range"] else "\nObserved kept-rate range: n/a"
        )
        change = details["first_to_last_kept_rate_change"]
        print(
            "First-to-last kept-rate change: "
            + (f"{change:+.1%}" if change is not None else "n/a")
        )
        print(
            "Unresolved-share denominator: all promise records with promised_date in "
            f"[as_of, as_of + {args.horizon} days)."
        )
        return 0

    if args.scores.is_file():
        dates = sorted(
            pd.read_csv(args.scores, usecols=["as_of"])["as_of"].dropna().unique().tolist()
        )
        reports = _from_persisted_scores(args.scores, dates, args.cutoff)
        print_table(reports)
        print(
            "\nWARNING: source Layer 2 database is missing; unresolved-promise "
            "counts/shares and early-snapshot rows/labels cannot be computed. "
            "The persisted score file contains only holdout snapshots."
        )
        print(
            "The metrics artifact reports 8,807 training rows across the four "
            "snapshots through 2026-06-30, but does not preserve their per-snapshot "
            "counts or labels."
        )
        return 2

    print(f"ERROR: Layer 2 database not found: {args.l2_db}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
