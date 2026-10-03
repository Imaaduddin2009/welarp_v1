"""Run the organiser-style question set and write the required four-column CSV."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--questions",
        type=Path,
        default=ROOT / "data" / "data" / "benchmark_questions.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "out" / "benchmark_answers.csv",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=ROOT / "out" / "l2_maple.duckdb",
    )
    args = parser.parse_args()
    if not args.questions.is_file():
        parser.error(f"questions file does not exist: {args.questions}")
    if not args.database.is_file():
        parser.error(f"Layer 2 database does not exist: {args.database}")

    os.environ["C360_L2_DB"] = str(args.database)
    import layer2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = layer2.run_benchmark(
        args.questions,
        args.output,
        id_col="question_id",
        q_col="question_text",
    )
    print(f"Wrote {len(result)} answers to {args.output}")
    print(f"Refused: {(result.refused == 'true').sum()}")


if __name__ == "__main__":
    main()
