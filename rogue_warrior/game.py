"""Fixed-step empty-road physics, independent of graphics and RL rewards."""

from dataclasses import dataclass, fields
from math import isfinite

from .actions import Action, DRIVING_ACTIONS


@dataclass(frozen=True)
class GameConfig:
    """Prototype settings: meters, seconds, and speeds in meters per second."""

    time_step: float = 1.0 / 60.0
    road_width: float = 12.0
    bike_width: float = 0.8
    max_speed: float = 35.0
    acceleration: float = 8.0
    braking: float = 14.0
    drag: float = 1.2
    steering_speed: float = 4.5
    race_distance: float = 500.0
    max_duration: float = 60.0

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isfinite(value) or value <= 0:
                raise ValueError(f"{field.name} must be a positive, finite number.")
        if self.bike_width >= self.road_width:
            raise ValueError("Bike width must be smaller than road width.")


@dataclass(frozen=True)
class BikeState:
    """Read-only snapshot; the world owns simulation updates."""

    x: float
    speed: float = 0.0
    distance: float = 0.0
    elapsed: float = 0.0
    steps: int = 0
    termination_reason: str | None = None

    @property
    def terminated(self) -> bool:
        return self.termination_reason == "finish"

    @property
    def truncated(self) -> bool:
        return self.termination_reason == "timeout"


class GameWorld:
    """Advance an empty-road race by exactly one fixed tick per action."""

    def __init__(self, config: GameConfig | None = None) -> None:
        self.config = config if config is not None else GameConfig()
        self.reset()

    @property
    def state(self) -> BikeState:
        return self._state

    def reset(self) -> BikeState:
        """Place a stationary bike at the road center for a fresh race."""
        self._state = BikeState(x=self.config.road_width / 2)
        return self._state

    def step(self, action: int) -> BikeState:
        """Update speed, position, distance, and outcome without rendering."""
        if self.state.termination_reason is not None:
            raise RuntimeError("Race ended. Call reset() before stepping again.")
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError("Driving action must be an integer from 0 to 4.")
        try:
            chosen = Action(action)
        except ValueError as error:
            raise ValueError("Driving action must be an integer from 0 to 4.") from error
        if chosen not in DRIVING_ACTIONS:
            raise ValueError("Combat actions are not implemented yet; choose 0 to 4.")

        config, state = self.config, self.state
        accelerating = chosen in DRIVING_ACTIONS[:3]
        acceleration = config.acceleration if accelerating else 0.0
        if chosen == Action.BRAKE:
            acceleration -= config.braking
        acceleration -= config.drag
        speed = max(0.0, min(config.max_speed, state.speed + acceleration * config.time_step))
        average_speed = (state.speed + speed) / 2
        steering = -1 if chosen == Action.ACCELERATE_LEFT else (
            1 if chosen == Action.ACCELERATE_RIGHT else 0
        )
        lateral_change = (
            steering * config.steering_speed * average_speed
            / config.max_speed * config.time_step
        )
        half_width = config.bike_width / 2
        x = max(half_width, min(config.road_width - half_width, state.x + lateral_change))
        distance = min(config.race_distance, state.distance + average_speed * config.time_step)
        steps = state.steps + 1
        elapsed = steps * config.time_step
        reason = None
        if distance >= config.race_distance:
            reason = "finish"
        elif elapsed >= config.max_duration:
            reason = "timeout"
        self._state = BikeState(x, speed, distance, elapsed, steps, reason)
        return self._state
