"""Gymnasium adapter and inspectable rewards for the empty-road game."""

from dataclasses import dataclass, fields
from math import isfinite
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .actions import DRIVING_ACTIONS
from .game import GameConfig, GameWorld


@dataclass(frozen=True)
class RewardConfig:
    """Initial reward weights, independent of the game's physical rules."""

    progress: float = 1.0
    finish_bonus: float = 5.0
    timeout_penalty: float = 1.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{field.name} must be finite and nonnegative.")


class DrivingEnv(gym.Env[np.ndarray, int]):
    """Five driving actions and four normalized, Markov state features.

    Each step advances one physical tick. The renderer is loaded only when
    rgb_array output is requested; normal rollouts do not create a window.
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: GameConfig | None = None,
        rewards: RewardConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'.")
        self.render_mode = render_mode
        self.world = GameWorld(config)
        self.rewards = rewards if rewards is not None else RewardConfig()
        self.action_space = spaces.Discrete(len(DRIVING_ACTIONS))
        self.observation_space = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)
        self.metadata = {**self.metadata, "render_fps": round(1 / self.world.config.time_step)}
        self._renderer = None
        self._has_reset = False
        self._episode_return = 0.0
        self._components = {"progress": 0.0, "finish": 0.0, "timeout": 0.0}

    def _observation(self) -> np.ndarray:
        state, config = self.world.state, self.world.config
        return np.array([
            state.x / config.road_width,
            state.speed / config.max_speed,
            state.distance / config.race_distance,
            min(state.elapsed / config.max_duration, 1.0),
        ], dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        state, config = self.world.state, self.world.config
        return {
            "x": state.x,
            "speed": state.speed,
            "distance": state.distance,
            "race_progress": state.distance / config.race_distance,
            "elapsed": state.elapsed,
            "steps": state.steps,
            "finish": state.terminated,
            "termination_reason": state.termination_reason,
            "episode_return": self._episode_return,
            "reward_components": self._components.copy(),
        }

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if options:
            raise ValueError("Reset options are not supported by this prototype.")
        super().reset(seed=seed)
        self.world.reset()
        self._has_reset = True
        self._episode_return = 0.0
        self._components = {"progress": 0.0, "finish": 0.0, "timeout": 0.0}
        return self._observation(), self._info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self._has_reset:
            raise RuntimeError("Call reset() before the first step.")
        if (isinstance(action, (bool, np.bool_))
                or not isinstance(action, (int, np.integer))
                or not self.action_space.contains(action)):
            raise ValueError("Action must be an integer from 0 to 4.")
        previous_distance = self.world.state.distance
        state = self.world.step(int(action))
        self._components = {
            "progress": self.rewards.progress * (
                state.distance - previous_distance
            ) / self.world.config.race_distance,
            "finish": self.rewards.finish_bonus if state.terminated else 0.0,
            "timeout": -self.rewards.timeout_penalty if state.truncated else 0.0,
        }
        reward = float(sum(self._components.values()))
        self._episode_return += reward
        return self._observation(), reward, state.terminated, state.truncated, self._info()

    def render(self) -> np.ndarray:
        if not self._has_reset:
            raise RuntimeError("Call reset() before rendering.")
        if self.render_mode != "rgb_array":
            raise ValueError("Construct DrivingEnv(render_mode='rgb_array') to render.")
        import pygame
        from .visuals import Renderer

        if self._renderer is None:
            self._renderer = Renderer()
        surface = self._renderer.draw(self.world.state, self.world.config)
        return pygame.surfarray.array3d(surface).transpose(1, 0, 2).copy()

    def close(self) -> None:
        """Release this adapter's renderer without closing unrelated game windows."""
        self._renderer = None
