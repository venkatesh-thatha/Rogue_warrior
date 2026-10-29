"""Check input, fixed-time stepping, and rendering using SDL's headless driver."""

import os
from collections import defaultdict
import unittest
from unittest.mock import patch

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame

from rogue_warrior.actions import Action
from rogue_warrior.game import GameConfig, GameWorld
from rogue_warrior.env import DrivingEnv
from rogue_warrior.play import FixedStepDriver, driving_action, play
from rogue_warrior.visuals import Renderer
from rogue_warrior.watch import watch_random


class VisualTests(unittest.TestCase):
    def test_watching_random_policy_matches_headless_at_different_frame_rates(self) -> None:
        reference = DrivingEnv()
        reference.reset(seed=42)
        reference.action_space.seed(42)
        while reference.world.state.termination_reason is None:
            reference.step(reference.action_space.sample())
        for fps in (30, 120):
            watched = DrivingEnv()
            watched.reset(seed=42)
            watched.action_space.seed(42)
            driver = FixedStepDriver(watched.world, step=watched.step)
            while watched.world.state.termination_reason is None:
                driver.advance(1 / fps, watched.action_space.sample)
            self.assertEqual(watched.world.state, reference.world.state)
            self.assertEqual(watched._episode_return, reference._episode_return)

    def test_agent_viewer_handles_replay_and_exit(self) -> None:
        clock = unittest.mock.Mock()
        clock.tick.return_value = 17
        env = DrivingEnv()
        events = [[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)], [],
                  [pygame.event.Event(pygame.QUIT)]]
        with (patch("rogue_warrior.watch.DrivingEnv", return_value=env),
              patch("builtins.print"),
              patch("pygame.time.Clock", return_value=clock),
              patch("pygame.event.get", side_effect=events)):
            watch_random(42)
        self.assertEqual(env.world.state.steps, 1)
        self.assertFalse(pygame.display.get_init())

    def test_projection_has_depth_and_follows_bike_laterally(self) -> None:
        renderer, world = Renderer(), GameWorld()
        config = world.config
        near_left = renderer.project(10, 0, world.state, config)
        near_right = renderer.project(10, config.road_width, world.state, config)
        far_left = renderer.project(200, 0, world.state, config)
        far_right = renderer.project(200, config.road_width, world.state, config)
        self.assertGreater(near_right[0] - near_left[0], far_right[0] - far_left[0])
        self.assertGreater(near_left[1], far_left[1])
        center_before = renderer.project(20, config.road_width / 2, world.state, config)
        for _ in range(120):
            world.step(Action.ACCELERATE_LEFT)
        center_after = renderer.project(20, config.road_width / 2, world.state, config)
        self.assertGreater(center_after[0], center_before[0])

    def test_keys_match_shared_actions(self) -> None:
        cases = [
            ((False, False, False, False), Action.COAST),
            ((True, False, False, False), Action.ACCELERATE),
            ((False, False, True, False), Action.ACCELERATE_LEFT),
            ((True, False, False, True), Action.ACCELERATE_RIGHT),
            ((True, True, True, False), Action.BRAKE),
            ((False, False, True, True), Action.ACCELERATE),
        ]
        for keys, expected in cases:
            with self.subTest(keys=keys):
                self.assertEqual(driving_action(**dict(zip(("up", "down", "left", "right"), keys))), expected)

    def test_display_frame_rates_produce_same_physics(self) -> None:
        worlds = [GameWorld() for _ in range(3)]
        for world, fps in zip(worlds, (30, 60, 120)):
            driver = FixedStepDriver(world)
            for _ in range(fps * 2):
                driver.advance(1 / fps, Action.ACCELERATE)
        self.assertEqual(worlds[0].state.steps, 120)
        self.assertEqual(worlds[0].state, worlds[1].state)
        self.assertEqual(worlds[0].state, worlds[2].state)

    def test_finish_and_restart_clear_pending_time(self) -> None:
        world = GameWorld(GameConfig(race_distance=0.01))
        driver = FixedStepDriver(world)
        driver.advance(0.25, Action.ACCELERATE)
        self.assertTrue(world.state.terminated)
        state = world.state
        driver.advance(0.25, Action.ACCELERATE)
        self.assertEqual(world.state, state)
        driver.reset()
        self.assertEqual(world.state.steps, 0)
        self.assertEqual(driver.accumulator, 0)

    def test_render_is_repeatable_and_does_not_change_physics(self) -> None:
        renderer, world = Renderer(), GameWorld()
        for _ in range(120):
            world.step(Action.ACCELERATE_LEFT)
        state = world.state
        first = pygame.image.tobytes(renderer.draw(state, world.config), "RGB")
        second = pygame.image.tobytes(renderer.draw(state, world.config), "RGB")
        self.assertEqual(first, second)
        self.assertEqual(world.state, state)
        self.assertEqual(renderer.surface.get_size(), (320, 240))
        self.assertGreater(len(set(first)), 10)

    def test_window_close_and_escape_exit_cleanly(self) -> None:
        events = [
            pygame.event.Event(pygame.QUIT),
            pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE),
        ]
        for event in events:
            with self.subTest(event=event), patch("pygame.event.get", return_value=[event]):
                play()
                self.assertFalse(pygame.display.get_init())

    def test_window_loop_finishes_race_and_restarts(self) -> None:
        world = GameWorld()
        outcomes = []
        original_reset = world.reset

        def reset():
            outcomes.append(world.state.termination_reason)
            return original_reset()

        clock = unittest.mock.Mock()
        clock.tick.return_value = 17
        events = ([[]] * 1100 + [[pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r)]]
                  + [[]] * 3 + [[pygame.event.Event(pygame.QUIT)]])
        with (patch("rogue_warrior.play.GameWorld", return_value=world),
              patch.object(world, "reset", side_effect=reset),
              patch("pygame.time.Clock", return_value=clock),
              patch("pygame.event.get", side_effect=events),
              patch("pygame.key.get_pressed", return_value=defaultdict(bool, {pygame.K_UP: True}))):
            play()
        self.assertEqual(outcomes, ["finish"])
        self.assertEqual(world.state.steps, 3)
        self.assertIsNone(world.state.termination_reason)


if __name__ == "__main__":
    unittest.main()
