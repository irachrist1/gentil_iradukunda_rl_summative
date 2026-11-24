"""One Gymnasium decision per day for a simulated emergency food operation."""

import gymnasium as gym
import numpy as np

DISTRICTS = ("Musanze", "Nyabihu", "Ngororero", "Rubavu")
ACTIONS = ("Hold", *(f"Dispatch to {d}" for d in DISTRICTS), "Replenish hub")
HORIZON = 16
HUB_CAPACITY = 36
DISTRICT_CAPACITY = 18
SHIPMENT = 6
REPLENISHMENT = 24
OBS_SIZE = 3 + 4 * len(DISTRICTS)


class FoodDispatchEnv(gym.Env):
    """Synthetic scenarios with action-independent, seeded demand and road conditions."""

    metadata = {"render_modes": []}

    def __init__(self, *, stress=False):
        self.stress = stress
        self.action_space = gym.spaces.Discrete(len(ACTIONS))
        self.observation_space = gym.spaces.Box(0, 1, shape=(OBS_SIZE,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        rng = self.np_random
        n = len(DISTRICTS)
        # Each tape represents alerts visible in the morning and demand revealed that evening.
        risk = rng.random((HORIZON, n))
        self.alerts = risk < (0.35 if self.stress else 0.20)
        self.roads = rng.random((HORIZON, n)) < (0.23 if self.stress else 0.10)
        surprise = rng.random((HORIZON, n)) < 0.12
        self.demand = 1 + self.alerts.astype(int) + surprise.astype(int)
        self.stock = rng.integers(1, 6, size=n) if self.stress else rng.integers(2, 8, size=n)
        self.hub = int(rng.integers(12, 22) if self.stress else rng.integers(16, 28))
        self.inbound = np.zeros(n, dtype=int)
        self.supplier = 0
        self.day = 0
        return self._observation(), {}

    def _observation(self):
        finished = self.day == HORIZON
        alerts = np.zeros(len(DISTRICTS)) if finished else self.alerts[self.day]
        roads = np.zeros(len(DISTRICTS)) if finished else self.roads[self.day]
        district = np.column_stack((self.stock / DISTRICT_CAPACITY,
                                    self.inbound / SHIPMENT, alerts, roads))
        global_features = [1 - self.day / HORIZON, self.hub / HUB_CAPACITY,
                           self.supplier / REPLENISHMENT]
        return np.concatenate((global_features, district.ravel())).astype(np.float32)

    def step(self, action):
        if self.day >= HORIZON:
            raise RuntimeError("Reset before starting another episode.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}")
        action = int(action)
        day = self.day
        delays = self.roads[day]
        delivered = self.inbound.copy()
        supplier_received = self.supplier
        self.inbound.fill(0)
        self.supplier = 0
        cost = 0.0
        rejected = False
        eta = None

        if 1 <= action <= len(DISTRICTS):
            district = action - 1
            if self.hub >= SHIPMENT:
                eta = 2 if delays[district] else 1
                if eta == 1:
                    delivered[district] += SHIPMENT
                else:
                    self.inbound[district] += SHIPMENT
                self.hub -= SHIPMENT
                cost = 0.18
            else:
                rejected = True
        elif action == len(ACTIONS) - 1:
            if supplier_received:
                rejected = True
            else:
                self.supplier = REPLENISHMENT
                eta = 2
                cost = 0.30

        demand = self.demand[day]
        unmet = np.maximum(demand - self.stock, 0)
        self.stock = np.maximum(self.stock - demand, 0) + delivered
        self.hub += supplier_received
        waste = int(np.maximum(self.stock - DISTRICT_CAPACITY, 0).sum()
                    + max(self.hub - HUB_CAPACITY, 0))
        self.stock = np.minimum(self.stock, DISTRICT_CAPACITY)
        self.hub = min(self.hub, HUB_CAPACITY)
        self.day += 1

        # At the next morning boundary, ETA-1 entries have arrived into stock.
        reward = -1.25 * int(unmet.sum()) - cost - 0.35 * rejected - 0.12 * waste
        info = {"action": ACTIONS[action], "day": self.day, "eta": eta,
                "rejected": rejected, "demand": demand.copy(), "unmet": unmet,
                "delivered": delivered, "supplier_received": supplier_received, "waste": waste}
        return self._observation(), reward, self.day == HORIZON, False, info
