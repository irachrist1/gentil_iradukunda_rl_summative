"""Ten controlled configurations per algorithm."""

from dataclasses import asdict, dataclass, replace

ALGORITHMS = ("dqn", "reinforce", "ppo", "a2c")
PROBE_STEPS = 2048
TRAIN_STEPS = 24576
WORKERS = 8
TRAIN_SEED = 41
REPEAT_SEEDS = (139, 271)
VALIDATION_SEEDS = tuple(range(10000, 10032))
TEST_SEEDS = tuple(range(20000, 20096))
STRESS_SEEDS = tuple(range(30000, 30096))


@dataclass(frozen=True)
class Config:
    algorithm: str
    run: int = 1
    lr: float = 0.001
    gamma: float = 0.97
    width: int = 64
    batch: int = 64
    replay: int = 8192
    target: int = 64
    epsilon_end: float = 0.08
    epsilon_steps: int = 12288
    entropy: float = 0.02
    rollout: int = 16
    episodes: int = 8
    normalize: bool = True
    clip: float = 0.20
    epochs: int = 4
    gae: float = 0.95
    value_coef: float = 0.50

    def record(self):
        return asdict(self)


VARIANTS = {
    "dqn": ({}, {"lr": 0.0003}, {"lr": 0.003}, {"gamma": 0.90},
            {"replay": 2048}, {"batch": 32}, {"target": 256},
            {"epsilon_steps": 4096}, {"epsilon_end": 0.20}, {"width": 32}),
    "reinforce": ({}, {"lr": 0.0003}, {"lr": 0.003}, {"gamma": 0.90},
                  {"episodes": 4}, {"episodes": 16}, {"normalize": False},
                  {"entropy": 0.0}, {"entropy": 0.08}, {"width": 32}),
    "ppo": ({}, {"lr": 0.0003}, {"lr": 0.003}, {"gamma": 0.90}, {"rollout": 8},
            {"clip": 0.10}, {"clip": 0.30}, {"epochs": 2}, {"gae": 0.80},
            {"entropy": 0.08}),
    "a2c": ({}, {"lr": 0.0003}, {"lr": 0.003}, {"gamma": 0.90}, {"rollout": 4},
            {"rollout": 8}, {"entropy": 0.0}, {"entropy": 0.08},
            {"value_coef": 0.25}, {"value_coef": 1.00}),
}


def configuration(algorithm, run):
    return replace(Config(algorithm, run), **VARIANTS[algorithm][run - 1])
