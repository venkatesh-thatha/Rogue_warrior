"""Validate the Gymnasium contract, reward edge cases, and seeded baselines."""

import unittest

import numpy as np
from gymnasium.utils.env_checker import check_env

from rogue_warrior.baselines import evaluate_random
from rogue_warrior.env import DrivingEnv, RewardConfig
from rogue_warrior.game import GameConfig


class DrivingEnvTests(unittest.TestCase):
    def test_gymnasium_checker(self) -> None:
        env = DrivingEnv()
        try:
            check_env(env, skip_render_check=True)
        finally:
            env.close()

    def test_observations_and_info_match_simulation(self) -> None:
        env = DrivingEnv()
        observation, info = env.reset(seed=12)
        np.testing.assert_array_equal(observation, np.array([0.5, 0, 0, 0], dtype=np.float32))
        self.assertEqual(info["episode_return"], 0)
        for action in [np.int64(0), 1, 2, 3, 4] * 40:
            observation, reward, terminated, truncated, info = env.step(action)
            self.assertTrue(env.observation_space.contains(observation))
            self.assertTrue(np.all(np.isfinite(observation)))
            self.assertIs(type(reward), float)
            self.assertFalse(terminated or truncated)
            self.assertAlmostEqual(reward, sum(info["reward_components"].values()))
            self.assertEqual(info["steps"], env.world.state.steps)

    def test_finish_bonus_once_and_reset_clears_return(self) -> None:
        env = DrivingEnv(GameConfig(race_distance=1))
        env.reset()
        total = 0
        while True:
            _, reward, terminated, truncated, info = env.step(0)
            total += reward
            if terminated:
                break
        self.assertFalse(truncated)
        self.assertEqual(info["reward_components"]["finish"], 5)
        self.assertAlmostEqual(total, 6)
        self.assertAlmostEqual(info["episode_return"], total)
        with self.assertRaises(RuntimeError):
            env.step(0)
        _, info = env.reset()
        self.assertEqual(info["episode_return"], 0)
        self.assertEqual(info["reward_components"], {"progress": 0, "finish": 0, "timeout": 0})

    def test_timeout_is_truncation_and_idle_earns_no_progress(self) -> None:
        env = DrivingEnv(GameConfig(time_step=0.1, max_duration=0.3))
        env.reset()
        rewards = []
        for _ in range(3):
            observation, reward, terminated, truncated, info = env.step(4)
            rewards.append(reward)
        self.assertEqual(rewards, [0, 0, -1])
        self.assertFalse(terminated)
        self.assertTrue(truncated)
        self.assertEqual(info["termination_reason"], "timeout")
        self.assertTrue(env.observation_space.contains(observation))

    def test_steering_reward_depends_only_on_distance_not_lane_changes(self) -> None:
        straight, steering = DrivingEnv(), DrivingEnv()
        straight.reset()
        steering.reset()
        for _ in range(120):
            self.assertEqual(straight.step(0)[1], steering.step(1)[1])
        self.assertNotEqual(straight.world.state.x, steering.world.state.x)

    def test_invalid_action_does_not_advance_physics_or_return(self) -> None:
        env = DrivingEnv()
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset()
        previous = env.world.state
        for invalid in (-1, 5, 1.0, "0", True, np.bool_(False)):
            with self.assertRaises(ValueError):
                env.step(invalid)
            self.assertEqual(env.world.state, previous)
            self.assertEqual(env._episode_return, 0)

    def test_rendered_frames_are_independent_and_do_not_advance_physics(self) -> None:
        env = DrivingEnv(render_mode="rgb_array")
        env.reset()
        env.step(0)
        previous = env.world.state
        frame = env.render()
        self.assertEqual(frame.shape, (240, 320, 3))
        self.assertEqual(frame.dtype, np.uint8)
        np.testing.assert_array_equal(frame, env.render())
        frame[:] = 0
        self.assertTrue(np.any(env.render() != 0))
        self.assertEqual(env.world.state, previous)
        env.close()
        env.close()

    def test_reward_configuration_and_validation(self) -> None:
        env = DrivingEnv(GameConfig(race_distance=0.01), RewardConfig(progress=2, finish_bonus=0))
        env.reset()
        while not env.world.state.terminated:
            env.step(0)
        self.assertAlmostEqual(env._episode_return, 2)
        for invalid in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                RewardConfig(finish_bonus=invalid)
        with self.assertRaises(ValueError):
            DrivingEnv(render_mode="invalid")

    def test_random_rollouts_reproduce_with_explicit_seeds(self) -> None:
        first = evaluate_random(episodes=2, seed=42)
        second = evaluate_random(episodes=2, seed=42)
        self.assertEqual(first, second)
        self.assertEqual([episode["seed"] for episode in first["episodes"]], [42, 43])
        for result in first["episodes"]:
            self.assertIn(result["outcome"], ("finish", "timeout"))
            self.assertGreater(result["steps"], 0)


if __name__ == "__main__":
    unittest.main()
