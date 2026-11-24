"""Run, resume, select, and audit the controlled experiments."""

import csv
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from training.common import SimulatorBatch, build_actor, evaluate
from training.config import (ALGORITHMS, PROBE_STEPS, REPEAT_SEEDS, STRESS_SEEDS, TEST_SEEDS,
                             TRAIN_SEED, TRAIN_STEPS, VALIDATION_SEEDS, WORKERS, configuration)
from training.dqn_training import train_dqn
from training.pg_training import train_pg


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def run_experiment(task):
    algorithm, run, seed, budget, root = task
    torch.set_num_threads(1)
    config = configuration(algorithm, run)
    location = Path(root) / f"{algorithm}-{run:02d}-seed{seed}"
    location.mkdir(exist_ok=True, parents=True)
    summary = location / "summary.json"
    if summary.exists() and (location / "policy.pt").is_file() and (location / "curve.csv").is_file():
        previous = json.loads(summary.read_text())
        if previous["budget"] == budget and previous["config"] == config.record():
            return previous
    torch.manual_seed(seed)
    actor = build_actor(config)
    simulation = SimulatorBatch(WORKERS, seed) if algorithm == "dqn" else None
    curve = []
    completed = 0

    def log(steps, losses, epsilon=None, entropy=None, simulation=simulation):
        nonlocal completed
        if steps and steps % PROBE_STEPS and steps != budget:
            return False
        episodes = [] if simulation is None else simulation.completed
        completed += len(episodes)
        evaluation = evaluate(actor, algorithm, VALIDATION_SEEDS)["mean_return"]
        curve.append({"steps": steps, "episodes": completed,
                      "training_return": float(np.mean(episodes)) if episodes else None,
                      "validation_return": evaluation,
                      "loss": float(np.mean(losses)) if losses else None,
                      "epsilon": epsilon, "entropy": entropy})
        episodes.clear()
        return True

    log(0, [])
    if algorithm == "dqn":
        train_dqn(actor, simulation, config, budget, log, seed)
    else:
        train_pg(actor, config, budget, log, seed)
    final = {"config": config.record(), "seed": seed, "budget": budget,
             "validation_return": curve[-1]["validation_return"], "episodes": completed}
    with (location / "curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=curve[0].keys())
        writer.writeheader()
        writer.writerows(curve)
    torch.save({"algorithm": algorithm, "run": run, "seed": seed, "state_dict": actor.state_dict()},
               location / "policy.pt")
    write_json(summary, final)
    print(f"{algorithm.upper():9} run {run:02d} seed {seed:3d}  val {final['validation_return']:7.2f}",
          flush=True)
    return final


def train_all(root, jobs, budget, *, smoke=False):
    tasks = [(algo, run, TRAIN_SEED, budget, str(root / "runs"))
             for run in range(1, 2 if smoke else 11) for algo in ALGORITHMS]
    if jobs == 1:
        summaries = list(map(run_experiment, tasks))
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            summaries = list(pool.map(run_experiment, tasks))
    winners = {algo: max((row for row in summaries if row["config"]["algorithm"] == algo),
                         key=lambda row: row["validation_return"])["config"]["run"]
               for algo in ALGORITHMS}
    if not smoke:
        repeats = [(algo, run, seed, budget, str(root / "runs"))
                   for algo, run in winners.items() for seed in REPEAT_SEEDS]
        if jobs == 1:
            summaries.extend(map(run_experiment, repeats))
        else:
            with ProcessPoolExecutor(max_workers=jobs) as pool:
                summaries.extend(pool.map(run_experiment, repeats))
    write_json(root / "sweep.json", summaries)
    return winners


def load_policy(checkpoint):
    record = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = configuration(record["algorithm"], record["run"])
    actor = build_actor(config)
    actor.load_state_dict(record["state_dict"])
    return actor.eval(), config, record["seed"]


def finalize(root, winners):
    torch.set_num_threads(1)
    scores = {}
    for algorithm in (*ALGORITHMS, "random", "rule"):
        seeds = (TRAIN_SEED, *REPEAT_SEEDS) if algorithm in winners else (None,)
        for seed in seeds:
            actor = None
            if seed is not None:
                run = winners[algorithm]
                artifact = root / "runs" / f"{algorithm}-{run:02d}-seed{seed}" / "policy.pt"
                if not artifact.is_file():
                    continue
                actor, _, _ = load_policy(artifact)
                if seed == TRAIN_SEED and root == Path("logs"):
                    model = Path("models") / ("dqn" if algorithm == "dqn" else "pg") / f"{algorithm}.pt"
                    model.parent.mkdir(exist_ok=True, parents=True)
                    model.write_bytes(artifact.read_bytes())
            entry = {"test": evaluate(actor, algorithm, TEST_SEEDS, details=True),
                     "stress": evaluate(actor, algorithm, STRESS_SEEDS, stress=True, details=True)}
            scores.setdefault(algorithm, {})[str(seed)] = entry
    best_algorithm = max(winners, key=lambda algo: scores[algo][str(TRAIN_SEED)]["test"]["mean_return"])
    write_json(root / "evaluation.json", {"winners": winners, "best_algorithm": best_algorithm,
                                         "scores": scores})
    return best_algorithm
