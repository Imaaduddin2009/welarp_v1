"""Run the verification and submission commands in a stable order."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMANDS = [
    [sys.executable, "-m", "pytest", "-q"],
    [sys.executable, "run_layer3_demo.py"],
    [sys.executable, "run_layer4.py"],
    [sys.executable, "scripts/run_benchmark.py"],
    [sys.executable, "scripts/score_benchmark.py"],
]


def main() -> None:
    for command in COMMANDS:
        print(f"\n>>> {' '.join(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    print("\nAll configured verification and submission commands completed.", flush=True)


if __name__ == "__main__":
    main()
