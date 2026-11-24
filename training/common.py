"""Shared networks, environment collection, and matched-seed policy evaluation."""

import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torch.distributions import Categorical
from torchrl.modules import ProbabilisticActor, QValueActor, ValueOperator

from environment.custom_env import ACTIONS, DISTRICTS, HORIZON, OBS_SIZE, FoodDispatchEnv


def network(width, outputs):
    return nn.Sequential(nn.Linear(OBS_SIZE, width), nn.ReLU(),
                         nn.Linear(width, width), nn.ReLU(), nn.Linear(width, outputs))


def build_actor(config):
    if config.algorithm == "dqn":
        return QValueActor(network(config.width, len(ACTIONS)),
                           in_keys=["observation"], action_space="categorical")
    logits = TensorDictModule(network(config.width, len(ACTIONS)),
                              in_keys=["observation"], out_keys=["logits"])
    return ProbabilisticActor(logits, in_keys=["logits"], distribution_class=Categorical,
                              return_log_prob=True)


def build_critic(config):
    return ValueOperator(network(config.width, 1), in_keys=["observation"])


@torch.no_grad()
def policy_logits(actor, obs, *, dqn):
    """Use the same input path for training, headless evaluation, and GUI replay."""
    data = TensorDict({"observation": torch.from_numpy(obs)}, batch_size=[len(obs)])
    return actor(data)["action_value"] if dqn else actor.module[0](data)["logits"]


class SimulatorBatch:
    def __init__(self, count, seed):
        self.envs = [FoodDispatchEnv() for _ in range(count)]
        self.rngs = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(count)]
        self.obs = np.stack([self._reset(i) for i in range(count)])
        self.returns = np.zeros(count, dtype=float)
        self.completed = []

    def _reset(self, i):
        return self.envs[i].reset(seed=int(self.rngs[i].integers(2**32)))[0]

    def step(self, actions):
        next_obs, rewards, terminal = [], [], []
        for i, (env, action) in enumerate(zip(self.envs, actions, strict=True)):
            after, reward, done, _, _ = env.step(action)
            next_obs.append(after)
            rewards.append(reward)
            terminal.append(done)
            self.returns[i] += reward
            if done:
                self.completed.append(float(self.returns[i]))
                self.returns[i] = 0
            self.obs[i] = self._reset(i) if done else after
        return np.stack(next_obs), np.asarray(rewards, dtype=np.float32), np.asarray(terminal)


def heuristic(env):
    """Reorder early and cover the district most likely to run short at arrival."""
    days = HORIZON - env.day
    supplier_due = env.supplier
    if days > 3 and env.hub <= 11 and not supplier_due:
        return len(ACTIONS) - 1
    if days <= 1 or env.hub < 6:
        return 0
    lead = 1 + env.roads[env.day].astype(int)
    eligible = lead < days
    urgency = 1.32 * lead + env.alerts[env.day] - env.stock - env.inbound
    urgency = np.where(eligible, urgency, -np.inf)
    return int(np.argmax(urgency)) + 1 if urgency.max() >= -3 else 0


def evaluate(actor, algorithm, seeds, *, stress=False, details=False):
    envs = [FoodDispatchEnv(stress=stress) for _ in seeds]
    obs = np.stack([env.reset(seed=seed)[0] for env, seed in zip(envs, seeds)])
    rng = np.random.default_rng(8731)
    returns = np.zeros(len(envs))
    unmet = np.zeros((len(envs), len(DISTRICTS)), dtype=int)
    demand = 0
    rejected = 0
    for _ in range(HORIZON):
        if algorithm == "random":
            actions = rng.integers(len(ACTIONS), size=len(envs))
        elif algorithm == "rule":
            actions = [heuristic(env) for env in envs]
        else:
            actions = policy_logits(actor, obs, dqn=algorithm == "dqn").argmax(dim=-1).tolist()
        for i, (env, action) in enumerate(zip(envs, actions)):
            obs[i], reward, _, _, info = env.step(action)
            returns[i] += reward
            unmet[i] += info["unmet"]
            demand += int(info["demand"].sum())
            rejected += int(info["rejected"])
    result = {"mean_return": float(returns.mean())}
    if details:
        result.update(return_sd=float(returns.std(ddof=1)),
                      return_se=float(returns.std(ddof=1) / np.sqrt(len(seeds))),
                      episodes=len(seeds), episode_returns=returns.tolist(),
                      unmet_per_episode=float(unmet.sum(axis=1).mean()),
                      service_rate=float(1 - unmet.sum() / demand),
                      unmet_by_district=unmet.mean(axis=0).tolist(),
                      rejected_per_episode=rejected / len(seeds))
    return result
