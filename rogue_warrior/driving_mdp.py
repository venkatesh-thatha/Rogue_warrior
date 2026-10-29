"""A deterministic, finite driving MDP with an explicit transition model."""

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite

from .actions import Action, TEACHING_ACTIONS


LANES = 3
PROGRESS_POSITIONS = 6
FINISH = LANES * PROGRESS_POSITIONS
STATE_COUNT = FINISH + 1
START = 1
GAMMA = 0.9
PROGRESS_REWARD = 1.0
FINISH_BONUS = 5.0


@dataclass(frozen=True)
class TeachingConfig:
    """Optional fixed hazardous cell; the original empty road is the default."""

    hazard_cell: tuple[int, int] | None = None
    hazard_penalty: float = 4.0

    def __post_init__(self) -> None:
        if not isfinite(self.hazard_penalty) or self.hazard_penalty < 0:
            raise ValueError("Hazard penalty must be finite and nonnegative.")
        if self.hazard_cell is not None:
            progress, lane = self.hazard_cell
            if (type(progress) is not int or type(lane) is not int
                    or not 1 <= progress < PROGRESS_POSITIONS or not 0 <= lane < LANES):
                raise ValueError("Hazard must be a road cell after the start row.")


@dataclass(frozen=True)
class Transition:
    """One possible outcome in the known transition/reward model."""

    probability: float
    next_state: int
    reward: float
    terminated: bool


def transitions(
    state: int, action: int, config: TeachingConfig | None = None,
) -> tuple[Transition, ...]:
    """Return all outcomes for a state/action, including absorbing FINISH."""
    if type(state) is not int or not 0 <= state < STATE_COUNT:
        raise ValueError(f"State must be an integer from 0 to {FINISH}.")
    if isinstance(action, bool) or not isinstance(action, (int, Action)):
        raise ValueError("Action must be 0, 1, or 2.")
    try:
        chosen = Action(action)
    except ValueError as error:
        raise ValueError("Action must be 0, 1, or 2.") from error
    if chosen not in TEACHING_ACTIONS:
        raise ValueError("Action must be 0, 1, or 2.")

    if state == FINISH:
        return (Transition(1.0, FINISH, 0.0, True),)

    progress, lane = divmod(state, LANES)
    lane_change = {
        Action.ACCELERATE: 0,
        Action.ACCELERATE_LEFT: -1,
        Action.ACCELERATE_RIGHT: 1,
    }[chosen]
    next_lane = max(0, min(LANES - 1, lane + lane_change))
    next_progress = progress + 1
    finished = next_progress == PROGRESS_POSITIONS
    next_state = FINISH if finished else LANES * next_progress + next_lane
    reward = PROGRESS_REWARD + (FINISH_BONUS if finished else 0.0)
    if config is not None and not finished and (next_progress, next_lane) == config.hazard_cell:
        reward -= config.hazard_penalty
    return (Transition(1.0, next_state, reward, finished),)


class DrivingMDP:
    """Episode runner using the same model that later DP methods can inspect."""

    def __init__(self, config: TeachingConfig | None = None) -> None:
        self.config = config if config is not None else TeachingConfig()
        self.state = START

    def reset(self) -> int:
        """Start a fresh episode in the center lane at progress zero."""
        self.state = START
        return self.state

    def step(self, action: int) -> Transition:
        """Apply one action; require a reset after the episode finishes."""
        if self.state == FINISH:
            raise RuntimeError("Episode finished. Call reset() before stepping again.")
        outcome = transitions(self.state, action, self.config)[0]
        self.state = outcome.next_state
        return outcome


def discounted_return(rewards: Iterable[float], gamma: float = GAMMA) -> float:
    """Calculate G_0 = r_1 + gamma*r_2 + gamma**2*r_3 + ... ."""
    if not isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("Gamma must be finite and between 0 and 1.")
    total = 0.0
    weight = 1.0
    for reward in rewards:
        if not isfinite(reward):
            raise ValueError("Rewards must be finite.")
        total += weight * reward
        weight *= gamma
    return total
