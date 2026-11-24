"""Train and evaluate the four agents, or replay the included best checkpoint."""

import argparse
import os
from pathlib import Path

from training.config import PROBE_STEPS, TRAIN_STEPS
from training.experiments import finalize, train_all


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("demo", "train", "smoke"), default="demo")
    parser.add_argument("--jobs", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    if args.command == "demo":
        from play import replay
        replay()
        return
    smoke = args.command == "smoke"
    root = Path("logs/smoke" if smoke else "logs")
    winners = train_all(root, args.jobs, PROBE_STEPS if smoke else TRAIN_STEPS, smoke=smoke)
    print(f"Best learned agent: {finalize(root, winners).upper()}")


if __name__ == "__main__":
    main()
