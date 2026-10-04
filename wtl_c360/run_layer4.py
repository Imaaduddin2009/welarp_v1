"""Train and evaluate the temporal Layer 4 model and generate fairness artifacts."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-through")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out")
    args = parser.parse_args()
    import layer4

    dates = layer4.discover_as_of_dates()
    train_through = args.train_through or dates[min(int(len(dates) * 0.7), len(dates) - 2)]
    metrics = layer4.run_model(dates, train_through, args.output_dir)
    print(f"LightGBM metrics: {metrics['model']}")
    print(f"Logistic baseline metrics: {metrics['logistic_baseline']}")
    print(f"Fairness audit: {metrics['fairness_audit']['available']}")
    print(f"Report: {ROOT / 'submission' / 'layer4_model_report.md'}")


if __name__ == "__main__":
    main()
