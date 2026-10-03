"""Train and evaluate the temporal Layer 4 model and generate fairness artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_DATES = [
    "2026-03-31", "2026-04-30", "2026-05-31",
    "2026-06-30", "2026-07-31", "2026-08-14",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-through", default="2026-06-30")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out")
    args = parser.parse_args()
    import layer4

    metrics = layer4.run_model(DEFAULT_DATES, args.train_through, args.output_dir)
    print(f"LightGBM metrics: {metrics['model']}")
    print(f"Logistic baseline metrics: {metrics['logistic_baseline']}")
    print(f"Fairness audit: {metrics['fairness_audit']['available']}")
    print(f"Report: {ROOT / 'submission' / 'layer4_model_report.md'}")


if __name__ == "__main__":
    main()
