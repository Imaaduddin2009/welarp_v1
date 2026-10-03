"""Compare benchmark answers without hiding refusals or missing predictions."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def normalize_answer(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().casefold()


def score(answers_path: Path, gold_path: Path) -> dict[str, object]:
    answers = pd.read_csv(answers_path)
    gold = pd.read_csv(gold_path)
    required_answers = {"question_id", "answer", "refused"}
    required_gold = {"question_id", "gold_answer"}
    if not required_answers.issubset(answers.columns):
        raise ValueError(f"Answers must contain {sorted(required_answers)}")
    if not required_gold.issubset(gold.columns):
        raise ValueError(f"Gold data must contain {sorted(required_gold)}")
    joined = gold[["question_id", "gold_answer"]].merge(
        answers[["question_id", "answer", "refused"]],
        on="question_id",
        how="left",
        validate="one_to_one",
    )
    joined["refused"] = joined["refused"].fillna("true").astype(str).str.lower().eq("true")
    joined["exact_match"] = joined.apply(
        lambda row: normalize_answer(row["gold_answer"])
        == normalize_answer(row["answer"])
        and not row["refused"],
        axis=1,
    )
    joined["missing_prediction"] = joined["answer"].isna()
    return {
        "questions": len(joined),
        "exact_matches": int(joined.exact_match.sum()),
        "exact_match_rate": float(joined.exact_match.mean()) if len(joined) else 0.0,
        "refusals": int(joined.refused.sum()),
        "missing_predictions": int(joined.missing_prediction.sum()),
        "results": joined.where(pd.notna(joined), None).to_dict(orient="records"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", type=Path, default=Path("out/benchmark_answers.csv"))
    parser.add_argument("--gold", type=Path, default=Path("data/labels/benchmark_dev_answers.csv"))
    parser.add_argument("--output", type=Path, default=Path("submission/benchmark_scores.json"))
    args = parser.parse_args()
    if not args.answers.is_file() or not args.gold.is_file():
        parser.error("Both --answers and --gold must exist")
    result = score(args.answers, args.gold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"{result['exact_matches']}/{result['questions']} exact; "
        f"{result['refusals']} refused; {result['missing_predictions']} missing"
    )
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
