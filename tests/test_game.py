"""Verify physical behavior and race lifecycle on an empty road."""

import unittest

from rogue_warrior.actions import Action
from rogue_warrior.game import GameConfig, GameWorld


class GameWorldTests(unittest.TestCase):
    def test_acceleration_builds_speed_and_respects_limit(self) -> None:
        world = GameWorld()
        self.assertEqual(world.state.speed, 0)
        first = world.step(Action.ACCELERATE)
        self.assertGreater(first.speed, 0)
        self.assertGreater(first.distance, 0)
        for _ in range(600):
            world.step(Action.ACCELERATE)
        self.assertEqual(world.state.speed, world.config.max_speed)

    def test_braking_slows_faster_than_coasting_and_never_reverses(self) -> None:
        braking, coasting = GameWorld(), GameWorld()
        for _ in range(60):
            braking.step(Action.ACCELERATE)
            coasting.step(Action.ACCELERATE)
        original_speed = braking.state.speed
        braking.step(Action.BRAKE)
        coasting.step(Action.COAST)
        self.assertLess(braking.state.speed, coasting.state.speed)
        self.assertLess(coasting.state.speed, original_speed)
        for _ in range(100):
            braking.step(Action.BRAKE)
        self.assertEqual(braking.state.speed, 0)
        distance = braking.state.distance
        braking.step(Action.BRAKE)
        self.assertEqual(braking.state.distance, distance)

    def test_steering_reaches_boundaries_without_leaving_road(self) -> None:
        for action, expected in (
            (Action.ACCELERATE_LEFT, 0.4),
            (Action.ACCELERATE_RIGHT, 11.6),
        ):
            with self.subTest(action=action):
                world = GameWorld()
                initial_x = world.state.x
                for _ in range(600):
                    state = world.step(action)
                    self.assertGreaterEqual(state.x, 0.4)
                    self.assertLessEqual(state.x, 11.6)
                self.assertNotEqual(state.x, initial_x)
                self.assertAlmostEqual(state.x, expected)

    def test_idle_bike_stays_still_and_times_out(self) -> None:
        world = GameWorld(GameConfig(time_step=0.1, max_duration=0.3))
        for _ in range(3):
            state = world.step(Action.COAST)
        self.assertEqual(state.distance, 0)
        self.assertEqual(state.x, 6)
        self.assertTrue(state.truncated)
        self.assertFalse(state.terminated)
        self.assertEqual(state.termination_reason, "timeout")

    def test_finish_clamps_distance_and_requires_reset(self) -> None:
        world = GameWorld(GameConfig(race_distance=1))
        while world.state.termination_reason is None:
            world.step(Action.ACCELERATE)
        self.assertTrue(world.state.terminated)
        self.assertFalse(world.state.truncated)
        self.assertEqual(world.state.distance, 1)
        with self.assertRaises(RuntimeError):
            world.step(0)
        state = world.reset()
        self.assertEqual((state.x, state.speed, state.distance, state.steps), (6, 0, 0, 0))
        self.assertIsNone(state.termination_reason)

    def test_reads_do_not_advance_world_and_action_sequences_repeat(self) -> None:
        first, second = GameWorld(), GameWorld()
        actions = [0] * 60 + [1] * 60 + [2] * 60 + [3] * 60 + [4] * 60
        for action in actions:
            previous = first.state
            self.assertIs(first.state, previous)
            self.assertEqual(first.step(action), second.step(action))
        self.assertEqual(first.state.elapsed, len(actions) * first.config.time_step)

    def test_invalid_actions_preserve_state(self) -> None:
        world = GameWorld()
        for invalid in (-1, 5, 10, 11, 1.5, "0", True):
            previous = world.state
            with self.assertRaises(ValueError):
                world.step(invalid)
            self.assertEqual(world.state, previous)

    def test_invalid_config_is_rejected(self) -> None:
        for name, value in (
            ("time_step", 0), ("max_speed", -1), ("race_distance", float("nan")),
            ("drag", float("inf")), ("max_duration", True), ("bike_width", 12),
        ):
            with self.subTest(name=name, value=value):
                with self.assertRaises(ValueError):
                    GameConfig(**{name: value})


if __name__ == "__main__":
    unittest.main()
