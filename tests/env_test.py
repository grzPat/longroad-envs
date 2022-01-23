import unittest
from longroad.envs import IntegerRoadRaw
import pickle
import numpy as np


class TestEvironments(unittest.TestCase):
    def test_IntegerRoadRaw(self):
        env = IntegerRoadRaw(agentsize=10)
        env.seed(seed=1234)
        obs = env.reset()
        obs = env.step(np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1]))
        # with open("tests/test_obs", "wb") as f:
        #     pickle.dump(obs,f)
        with open("tests/test_obs", "rb") as f:
            test_obs = pickle.load(f)
        self.assertEqual(
            obs[0].tolist(),
            test_obs[0].tolist(),
            "Observation not the same")
        self.assertEqual(
            obs[1].tolist(),
            test_obs[1].tolist(),
            "Rewards not the same")


if __name__ == '__main__':
    unittest.main()
