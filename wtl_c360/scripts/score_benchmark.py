"""Compare benchmark answers without hiding refusals or missing predictions."""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path

import pandas as pd


def normalize_answer(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def _csv_rows(value: object) -> list[list[str]]:
    if pd.isna(value):
        return []
    return list(csv.reader(io.StringIO(str(value).strip())))


def _tolerance(value: object) -> float | None:
    match = re.search(r"[+-]?\s*(\d+(?:\.\d+)?)", str(value))
    if not match:
        return 0.0 if str(value).strip().casefold() == "exact" else None
    return float(match.group(1))


def _numeric(value: str) -> float | None:
    try:
        return float(value.replace(",", "").replace("$", "").strip())
    except (TypeError, ValueError):
        return None


def _table_match(expected: object, actual: object, tolerance: object) -> bool | None:
    if "\n" not in str(expected) and "\r" not in str(expected):
        return None
    expected_rows = _csv_rows(expected)
    actual_rows = _csv_rows(actual)
    if not expected_rows:
        return None
    if not actual_rows:
        number = _numeric(str(actual))
        gold_numbers = [_numeric(cell) for row in expected_rows for cell in row]
        gold_numbers = [value for value in gold_numbers if value is not None]
        if number is None or len(gold_numbers) != 1:
            return None
        allowed = _tolerance(tolerance)
        return allowed is not None and abs(gold_numbers[0] - number) <= allowed
    if len(expected_rows[0]) == 1 and len(expected_rows) == 2 and len(actual_rows) == 1 and len(actual_rows[0]) == 1:
        gold_value, actual_value = _numeric(expected_rows[1][0]), _numeric(actual_rows[0][0])
        allowed = _tolerance(tolerance)
        return (
            gold_value is not None and actual_value is not None and allowed is not None
            and abs(gold_value - actual_value) <= allowed
        )
    if len(expected_rows[0]) != len(actual_rows[0]):
        return False
    if [cell.strip().casefold() for cell in expected_rows[0]] != [
        cell.strip().casefold() for cell in actual_rows[0]
    ]:
        return False
    expected_body, actual_body = expected_rows[1:], actual_rows[1:]
    if len(expected_body) != len(actual_body):
        return False
    first_column = 0
    expected_body = sorted(expected_body, key=lambda row: row[first_column].casefold())
    actual_body = sorted(actual_body, key=lambda row: row[first_column].casefold())
    allowed = _tolerance(tolerance)
    for expected_row, actual_row in zip(expected_body, actual_body):
        if len(expected_row) != len(actual_row):
            return False
        for index, (expected_cell, actual_cell) in enumerate(zip(expected_row, actual_row)):
            if index == first_column:
                if normalize_answer(expected_cell) != normalize_answer(actual_cell):
                    return False
                continue
            expected_number, actual_number = _numeric(expected_cell), _numeric(actual_cell)
            if expected_number is not None and actual_number is not None:
                metric_column = bool(re.search(r"(rate|pct|dpd|average|avg)", expected_rows[0][index], re.I))
                cell_tolerance = allowed if metric_column and allowed is not None else 0.0
                if abs(expected_number - actual_number) > cell_tolerance:
                    return False
            elif normalize_answer(expected_cell) != normalize_answer(actual_cell):
                return False
    return True


_FACT_STOP_WORDS = {
    "a", "an", "and", "as", "at", "be", "before", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "the", "to", "was", "were",
    "what", "when", "which", "with",
}
_FACT_SYNONYMS = {
    "ill": "illness", "sick": "illness", "medical": "illness",
    "said": "report", "reports": "report", "reported": "report",
    "agreed": "promise", "promised": "promise", "promising": "promise",
    "pay": "payment", "paid": "payment", "payments": "payment",
    "cut": "reduced", "reduction": "reduced", "hours": "work_hours",
    "week": "work_hours", "weeks": "work_hours", "day": "date", "days": "date",
}


def _fact_tokens(value: object) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+(?:\.\d+)?", normalize_answer(value))
    return {
        _FACT_SYNONYMS.get(token, token)
        for token in tokens
        if token not in _FACT_STOP_WORDS
    }


def _document_fact_match(expected: object, actual: object) -> bool:
    expected_tokens = _fact_tokens(expected)
    if not expected_tokens:
        return False
    overlap = len(expected_tokens & _fact_tokens(actual))
    return overlap / len(expected_tokens) >= 0.72


def _gold_sources_cited(gold_sources: object, actual_sources: object) -> bool:
    try:
        sources = json.loads(str(gold_sources))
    except (TypeError, json.JSONDecodeError):
        return False
    cited = normalize_answer(actual_sources)
    for source in sources if isinstance(sources, list) else []:
        identifiers = re.findall(r"[a-z][a-z0-9-]{2,}|\d{4,}", str(source).casefold())
        if identifiers and any(identifier in cited for identifier in identifiers):
            return True
    return False


def score(
    answers_path: Path,
    gold_path: Path,
    questions_path: Path | None = None,
) -> dict[str, object]:
    answers = pd.read_csv(answers_path, keep_default_na=False)
    gold = pd.read_csv(gold_path, keep_default_na=False)
    required_answers = {"question_id", "answer", "refused"}
    required_gold = {"question_id", "gold_answer"}
    if not required_answers.issubset(answers.columns):
        raise ValueError(f"Answers must contain {sorted(required_answers)}")
    if not required_gold.issubset(gold.columns):
        raise ValueError(f"Gold data must contain {sorted(required_gold)}")
    if "sql_or_sources" not in answers.columns:
        answers["sql_or_sources"] = ""
    gold_columns = ["question_id", "gold_answer"]
    if "gold_sources" in gold.columns:
        gold_columns.append("gold_sources")
    else:
        gold["gold_sources"] = ""
        gold_columns.append("gold_sources")
    joined = gold[gold_columns].merge(
        answers[["question_id", "answer", "refused", "sql_or_sources"]],
        on="question_id",
        how="left",
        validate="one_to_one",
    )
    if questions_path and questions_path.is_file():
        questions = pd.read_csv(questions_path, keep_default_na=False)
        tolerance_by_id = questions.set_index("question_id").tolerance.to_dict()
        joined["tolerance"] = joined.question_id.map(tolerance_by_id).fillna("")
    else:
        joined["tolerance"] = ""
    joined["expected_refused"] = joined.gold_answer.astype(str).str.match(r"(?i)^\s*refuse:")
    joined["refused"] = joined.refused.astype(str).str.lower().eq("true")
    joined["refusal_correct"] = joined.expected_refused.eq(joined.refused)
    joined["gold_sources_cited"] = joined.apply(
        lambda row: _gold_sources_cited(row.gold_sources, row.sql_or_sources),
        axis=1,
    )
    answer_matches = []
    for row in joined.itertuples():
        if row.expected_refused:
            answer_matches.append(None)
            continue
        expected = str(row.gold_answer)
        actual = str(row.answer)
        tolerance = row.tolerance
        tabular = _table_match(expected, actual, tolerance)
        if tabular is not None:
            answer_matches.append(tabular)
        elif normalize_answer(expected) == normalize_answer(actual):
            answer_matches.append(True)
        elif (
            row.gold_sources
            and _gold_sources_cited(row.gold_sources, row.sql_or_sources)
            and _document_fact_match(expected, actual)
        ):
            answer_matches.append(True)
        elif row.gold_sources and _gold_sources_cited(row.gold_sources, row.sql_or_sources):
            answer_matches.append(None)
        else:
            answer_matches.append(False)
    joined["answer_match"] = answer_matches
    joined["manual_document_review"] = joined.answer_match.isna() & ~joined.expected_refused
    joined["correct"] = joined.refusal_correct & (
        joined.expected_refused | joined.answer_match.fillna(False)
    )
    joined["missing_prediction"] = joined.answer.eq("")
    return {
        "questions": len(joined),
        "correct": int(joined.correct.sum()),
        "scored_answers": int((~joined.manual_document_review).sum()),
        "accuracy_on_scored": (
            float(joined.correct.sum() / (~joined.manual_document_review).sum())
            if (~joined.manual_document_review).any() else None
        ),
        "manual_document_review": int(joined.manual_document_review.sum()),
        "supported_answer_matches": int(joined.answer_match.eq(True).sum()),
        "refusal_correct": int(joined.refusal_correct.sum()),
        "incorrect_refusals": int((joined.expected_refused & ~joined.refused).sum()),
        "wrongly_refused": int((~joined.expected_refused & joined.refused).sum()),
        "refusals": int(joined.refused.sum()),
        "missing_predictions": int(joined.missing_prediction.sum()),
        "results": joined.where(pd.notna(joined), None).to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, default=Path("out/benchmark_answers.csv"))
    parser.add_argument("--gold", type=Path, default=Path("data/labels/benchmark_dev_answers.csv"))
    parser.add_argument("--questions", type=Path, default=Path("data/data/benchmark_questions.csv"))
    parser.add_argument("--output", type=Path, default=Path("submission/benchmark_scores.json"))
    args = parser.parse_args()
    if not args.answers.is_file() or not args.gold.is_file():
        parser.error("Both --answers and --gold must exist")
    result = score(args.answers, args.gold, args.questions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{result['correct']}/{result['questions']} correct among "
        f"{result['scored_answers']} automatically scored; "
        f"{result['manual_document_review']} document answers need review; "
        f"{result['refusals']} refused ({result['wrongly_refused']} incorrect); "
        f"{result['missing_predictions']} missing"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
