"""Rebuild the evidence figures from saved curve and evaluation logs."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from training.config import ALGORITHMS, TRAIN_SEED

LABELS = {"dqn": "DQN", "reinforce": "REINFORCE", "ppo": "PPO", "a2c": "A2C",
          "random": "Random", "rule": "Dispatch rule"}
COLORS = {"dqn": "#1473a3", "reinforce": "#b56028", "ppo": "#218265", "a2c": "#795795"}
THRESHOLD = -16


def read_curve(algorithm, run, seed=TRAIN_SEED):
    path = Path(f"logs/runs/{algorithm}-{run:02d}-seed{seed}/curve.csv")
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) if row[key] else np.nan for row in rows])
            for key in rows[0]}


def comparison(values, a, b, split):
    difference = np.asarray(values[a][str(TRAIN_SEED)][split]["episode_returns"]) - np.asarray(
        values[b]["None" if b in ("random", "rule") else str(TRAIN_SEED)][split]["episode_returns"])
    margin = 1.985 * difference.std(ddof=1) / np.sqrt(difference.size)
    center = difference.mean()
    return center, center - margin, center + margin


def format_axis(ax):
    ax.grid(axis="y", color="#e4ebe7")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#a4b4ad")


def save(fig, name):
    fig.savefig(Path("assets") / f"{name}.png", dpi=220, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": "#172d2a",
                         "axes.labelcolor": "#172d2a", "xtick.color": "#354a43", "ytick.color": "#354a43"})
    result = json.loads(Path("logs/evaluation.json").read_text())
    curves = {a: read_curve(a, run) for a, run in result["winners"].items()}
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 4.0), sharex=True, sharey=True, layout="constrained")
    for a, ax in zip(ALGORITHMS, axes.ravel()):
        series = curves[a]
        ax.plot(series["episodes"], series["training_return"], color=COLORS[a], marker="o",
                markersize=3, linewidth=1.7)
        ax.set_title(LABELS[a], loc="left", fontweight="bold")
        ax.set_ylim(-62, -8)
        format_axis(ax)
    fig.supxlabel("Training episodes completed")
    fig.supylabel("Mean episode return")
    save(fig, "training_returns")

    fig, (ax, pg) = plt.subplots(1, 2, figsize=(8.6, 2.55), layout="constrained")
    dqn = curves["dqn"]
    ax.plot(dqn["episodes"], dqn["loss"], color=COLORS["dqn"], linewidth=1.8)
    ax.set(title="DQN temporal-difference loss", xlabel="Training episodes", ylabel="Mean Huber loss")
    for a in ("reinforce", "ppo", "a2c"):
        series = curves[a]
        pg.plot(series["episodes"], series["entropy"], label=LABELS[a], color=COLORS[a], linewidth=1.8)
    pg.set(title="Policy-gradient exploration", xlabel="Training episodes", ylabel="Mean entropy (nats)",
           ylim=(0.60, 1.85))
    pg.legend(loc="lower left", frameon=False, fontsize=8)
    for axis in (ax, pg):
        format_axis(axis)
    save(fig, "stability")

    fig, ax = plt.subplots(figsize=(8.6, 3.2), layout="constrained")
    for a in ALGORITHMS:
        series = curves[a]
        ax.plot(series["episodes"], series["validation_return"], label=LABELS[a], color=COLORS[a],
                linewidth=1.8, marker="o", markersize=3)
    ax.axhline(THRESHOLD, color="#53665d", linestyle="--", linewidth=1, label="−16 benchmark")
    ax.set(xlabel="Training episodes completed", ylabel="Mean validation return")
    ax.legend(ncol=3, loc="lower right", frameon=False, fontsize=8)
    format_axis(ax)
    save(fig, "convergence")

    labels = (*ALGORITHMS, "random", "rule")
    fig, ax = plt.subplots(figsize=(8.6, 3.35), layout="constrained")
    index = np.arange(len(labels))
    for split, offset, color, label in (
        ("test", -0.18, "#23785e", "Nominal"), ("stress", 0.18, "#b86d3c", "Stress")
    ):
        rows = [result["scores"][a]["None" if a in ("random", "rule") else str(TRAIN_SEED)][split]
                for a in labels]
        ax.bar(index + offset, [row["service_rate"] * 100 for row in rows], width=0.33,
               color=color, label=label)
    ax.set(xticks=index, xticklabels=[LABELS[a] for a in labels], ylabel="Demand served (%)", ylim=(0, 105))
    ax.legend(frameon=False, ncol=2, loc="upper right")
    format_axis(ax)
    save(fig, "generalization")


if __name__ == "__main__":
    main()
