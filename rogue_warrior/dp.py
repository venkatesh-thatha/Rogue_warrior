"""Week 3 dynamic programming for known finite transition/reward models."""

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from .actions import TEACHING_ACTIONS
from .driving_mdp import STATE_COUNT, TeachingConfig, Transition, transitions


@dataclass(frozen=True)
class TabularModel:
    """Outcomes indexed by state and action column; preserve external action IDs."""

    outcomes: tuple[tuple[tuple[Transition, ...], ...], ...]
    action_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.outcomes or not self.action_ids or len(set(self.action_ids)) != len(self.action_ids):
            raise ValueError("Model needs states and unique action IDs.")
        for row in self.outcomes:
            if len(row) != len(self.action_ids):
                raise ValueError("Every state must have an outcome list for every action.")
            for outcomes in row:
                if not outcomes:
                    raise ValueError("State/action outcomes cannot be empty.")
                for outcome in outcomes:
                    if (not isfinite(outcome.probability) or outcome.probability < 0
                            or not isfinite(outcome.reward)
                            or not 0 <= outcome.next_state < self.state_count):
                        raise ValueError("Invalid transition outcome.")
                if not np.isclose(sum(item.probability for item in outcomes), 1, rtol=0, atol=1e-12):
                    raise ValueError("Transition probabilities must sum to one.")

    @property
    def state_count(self) -> int:
        return len(self.outcomes)

    @property
    def action_count(self) -> int:
        return len(self.action_ids)


def driving_model(config: TeachingConfig | None = None) -> TabularModel:
    """Expose the teaching MDP's exact model without sampling episodes."""
    return TabularModel(
        tuple(tuple(transitions(state, action, config) for action in TEACHING_ACTIONS)
              for state in range(STATE_COUNT)),
        tuple(int(action) for action in TEACHING_ACTIONS),
    )


def gymnasium_model(env: Any) -> TabularModel:
    """Adapt explicit P tables such as FrozenLake's, not arbitrary simulators."""
    base = env.unwrapped
    if not hasattr(base, "P"):
        raise ValueError("Environment must expose a known tabular transition model P.")
    return TabularModel(
        tuple(tuple(tuple(Transition(float(p), int(next_state), float(reward), bool(done))
                          for p, next_state, reward, done in base.P[state][action])
                    for action in range(base.action_space.n))
              for state in range(base.observation_space.n)),
        tuple(range(base.action_space.n)),
    )


def uniform_policy(model: TabularModel) -> np.ndarray:
    """A state-by-action probability table for uniform random choices."""
    return np.full((model.state_count, model.action_count), 1 / model.action_count)


def _validated_policy(model: TabularModel, policy: np.ndarray) -> np.ndarray:
    result = np.asarray(policy, dtype=np.float64)
    if (result.shape != (model.state_count, model.action_count)
            or not np.all(np.isfinite(result)) or np.any(result < 0)
            or not np.allclose(result.sum(axis=1), 1, rtol=0, atol=1e-12)):
        raise ValueError("Policy must have a normalized probability row for each state.")
    return result


def action_values(model: TabularModel, values: np.ndarray, gamma: float) -> np.ndarray:
    """One-step lookahead Q(s,a); never bootstrap through terminal outcomes."""
    if not isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("Gamma must be finite and between zero and one.")
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (model.state_count,) or not np.all(np.isfinite(values)):
        raise ValueError("Values must be a finite vector with one entry per state.")
    result = np.zeros((model.state_count, model.action_count))
    for state, row in enumerate(model.outcomes):
        for action, outcomes in enumerate(row):
            result[state, action] = sum(
                item.probability * (item.reward + (
                    0 if item.terminated else gamma * values[item.next_state]
                )) for item in outcomes
            )
    return result


@dataclass(frozen=True)
class Evaluation:
    values: np.ndarray
    sweeps: int
    residual: float


def policy_evaluation(
    model: TabularModel, policy: np.ndarray, gamma: float = 0.9,
    tolerance: float = 1e-10, max_sweeps: int = 10000,
) -> Evaluation:
    """Iterate the Bellman expectation equation using synchronous sweeps."""
    policy = _validated_policy(model, policy)
    if not isfinite(tolerance) or tolerance <= 0 or type(max_sweeps) is not int or max_sweeps < 1:
        raise ValueError("Tolerance and sweep cap must be positive.")
    values = np.zeros(model.state_count)
    for sweep in range(1, max_sweeps + 1):
        values = np.sum(policy * action_values(model, values, gamma), axis=1)
        target = np.sum(policy * action_values(model, values, gamma), axis=1)
        residual = float(np.max(np.abs(target - values)))
        if residual <= tolerance:
            return Evaluation(values, sweep, residual)
    raise RuntimeError(f"Policy evaluation did not converge in {max_sweeps} sweeps.")


@dataclass(frozen=True)
class Iteration:
    policy: np.ndarray
    values: np.ndarray
    iterations: int
    evaluation_sweeps: int
    residual: float


def policy_iteration(
    model: TabularModel, gamma: float = 0.9, tolerance: float = 1e-10,
    max_iterations: int = 1000, max_evaluation_sweeps: int = 10000,
) -> Iteration:
    """Alternate evaluation and greedy improvement until the policy is stable.

    Keep an existing deterministic action on numerical ties within 1e-12;
    otherwise choose the first maximizing action column. This avoids cycling.
    """
    if type(max_iterations) is not int or max_iterations < 1:
        raise ValueError("Policy iteration cap must be a positive integer.")
    policy = uniform_policy(model)
    total_sweeps = 0
    for iteration in range(1, max_iterations + 1):
        evaluation = policy_evaluation(model, policy, gamma, tolerance, max_evaluation_sweeps)
        total_sweeps += evaluation.sweeps
        q_values = action_values(model, evaluation.values, gamma)
        improved = np.zeros_like(policy)
        for state in range(model.state_count):
            best = int(np.argmax(q_values[state]))
            previous = int(np.argmax(policy[state]))
            if policy[state, previous] == 1 and np.isclose(
                q_values[state, previous], q_values[state, best], rtol=0, atol=1e-12,
            ):
                best = previous
            improved[state, best] = 1
        if np.array_equal(improved, policy):
            residual = float(np.max(np.abs(q_values.max(axis=1) - evaluation.values)))
            return Iteration(policy, evaluation.values, iteration, total_sweeps, residual)
        policy = improved
    raise RuntimeError(f"Policy iteration did not stabilize in {max_iterations} iterations.")
