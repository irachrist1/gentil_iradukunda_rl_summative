"""Reload an evaluated policy and show its decisions in the simulator."""

import argparse
import json
import os
from pathlib import Path

import torch

from environment.custom_env import DISTRICTS, HORIZON, FoodDispatchEnv
from training.common import policy_logits
from training.config import ALGORITHMS, TEST_SEEDS
from training.experiments import load_policy


def replay(algorithm=None, seed=TEST_SEEDS[0], *, stress=False, delay=0.75, headless=False,
           capture=None, capture_day=7, trace=None):
    torch.set_num_threads(1)
    if algorithm is None:
        algorithm = json.loads(Path("logs/evaluation.json").read_text())["best_algorithm"]
    path = Path("models") / ("dqn" if algorithm == "dqn" else "pg") / f"{algorithm}.pt"
    actor, config, train_seed = load_policy(path)
    env = FoodDispatchEnv(stress=stress)
    observation, _ = env.reset(seed=seed)
    sums = {"return": 0.0, "unmet": 0, "demand": 0}
    history = []
    display = None
    if not headless or capture:
        if headless:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        from environment.rendering import DispatchDisplay
        display = DispatchDisplay(headless=headless)
        display.draw(env, algorithm, seed, sums, history)
    print(f"{algorithm.upper()} experiment {config.run:02d} | training seed {train_seed} | "
          f"scenario seed {seed} | {'stress' if stress else 'nominal'} | {path}", flush=True)
    try:
        for _ in range(HORIZON):
            if display and not display.wait(0 if headless else delay):
                break
            action = policy_logits(actor, observation[None, :],
                                   dqn=algorithm == "dqn").argmax(dim=-1).item()
            observation, reward, done, _, info = env.step(action)
            for key in ("demand", "unmet", "delivered"):
                info[key] = info[key].tolist()
            info["reward"] = reward
            info["unmet_total"] = sum(info["unmet"])
            history.append(info)
            sums["return"] += reward
            sums["unmet"] += info["unmet_total"]
            sums["demand"] += sum(info["demand"])
            shortages = ", ".join(f"{name} {qty}" for name, qty in zip(DISTRICTS, info["unmet"]) if qty)
            print(f"day {info['day']:02d} | {info['action']:<23} | hub {env.hub:2d} | "
                  f"ETA {info['eta'] or '-'} | arrivals {sum(info['delivered']):2d} | "
                  f"unmet {info['unmet_total']:2d} | reward {reward:6.2f}"
                  f"{' | ' + shortages if shortages else ''}"
                  f"{' | rejected' if info['rejected'] else ''}", flush=True)
            if display:
                display.draw(env, algorithm, seed, sums, history)
                if capture and env.day == capture_day:
                    display.save(capture)
        print(f"{len(history):2d}/{HORIZON} days | return {sums['return']:.2f} | "
              f"unmet {sums['unmet']}/{sums['demand']} crates", flush=True)
        if trace:
            Path(trace).write_text(json.dumps({"algorithm": algorithm, "run": config.run,
                                               "seed": seed, "stress": stress, "totals": sums,
                                               "days": history}, indent=2) + "\n")
        if display and not headless and len(history) == HORIZON:
            display.wait(5)
    finally:
        if display:
            display.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=ALGORITHMS)
    parser.add_argument("--seed", type=int, default=TEST_SEEDS[0])
    parser.add_argument("--stress", action="store_true")
    parser.add_argument("--delay", type=float, default=0.75)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--capture")
    parser.add_argument("--capture-day", type=int, choices=range(1, HORIZON + 1), default=7)
    parser.add_argument("--trace")
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay must be non-negative")
    replay(**vars(args))


if __name__ == "__main__":
    main()
