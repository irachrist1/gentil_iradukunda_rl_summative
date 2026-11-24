"""Contract and delayed-transition regression checks."""

import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from environment.custom_env import ACTIONS, HORIZON, SHIPMENT, FoodDispatchEnv
from training.config import ALGORITHMS, configuration


class EnvironmentTests(unittest.TestCase):
    def test_gymnasium_contract(self):
        check_env(FoodDispatchEnv(), skip_render_check=True)

    def test_same_seed_daily_demand_despite_different_actions(self):
        envs = [FoodDispatchEnv(), FoodDispatchEnv()]
        first, second = [env.reset(seed=19)[0] for env in envs]
        np.testing.assert_array_equal(first, second)
        for _ in range(HORIZON):
            held = envs[0].step(0)
            shipped = envs[1].step(1)
            np.testing.assert_array_equal(held[-1]["demand"], shipped[-1]["demand"])
        self.assertTrue(held[2] and shipped[2])
        self.assertFalse(held[3] or shipped[3])

    def test_slow_road_commits_stock_before_delivery(self):
        env = FoodDispatchEnv()
        env.reset(seed=9)
        env.roads[0, 0] = True
        hub = env.hub
        _, _, _, _, first = env.step(1)
        self.assertEqual(env.hub, hub - SHIPMENT)
        self.assertEqual(first["delivered"][0], 0)
        self.assertEqual(first["eta"], 2)
        self.assertEqual(env.inbound[0], SHIPMENT)
        _, _, _, _, second = env.step(0)
        self.assertEqual(second["delivered"][0], SHIPMENT)
        self.assertEqual(env.inbound[0], 0)

    def test_supplier_order_and_rejection(self):
        env = FoodDispatchEnv()
        env.reset(seed=9)
        action = len(ACTIONS) - 1
        _, _, _, _, first = env.step(action)
        self.assertEqual(first["eta"], 2)
        self.assertEqual(first["supplier_received"], 0)
        _, _, _, _, second = env.step(action)
        self.assertTrue(second["rejected"])
        self.assertEqual(second["supplier_received"], 24)

    def test_all_sweep_configs_are_unique_without_run_id(self):
        for algo in ALGORITHMS:
            configs = [configuration(algo, i).record() for i in range(1, 11)]
            for config in configs:
                del config["run"]
            self.assertEqual(len({tuple(row.items()) for row in configs}), 10)


if __name__ == "__main__":
    unittest.main()
