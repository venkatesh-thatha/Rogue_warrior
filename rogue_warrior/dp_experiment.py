"""Inspect DP values, follow an improved route, and validate on FrozenLake."""

from dataclasses import asdict
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from .dp import gymnasium_model, driving_model, policy_evaluation, policy_iteration, uniform_policy
from .driving_mdp import DrivingMDP, FINISH, GAMMA, LANES, START, TeachingConfig, discounted_return


def _evaluate_lake(env, policy: np.ndarray, episodes: int, seed: int) -> dict:
    """Evaluate without policy updates, with a 100-step episode cutoff."""
    rng = np.random.default_rng(seed)
    returns, discounted, lengths = [], [], []
    successes = 0
    timeouts = 0
    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        rewards = []
        while True:
            action = int(rng.choice(env.action_space.n, p=policy[state]))
            state, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                timeouts += int(truncated)
                break
        returns.append(sum(rewards))
        discounted.append(discounted_return(rewards, GAMMA))
        lengths.append(len(rewards))
        successes += int(sum(rewards) > 0)
    return {
        "episodes": episodes, "seed_start": seed, "max_episode_steps": 100,
        "success_rate": successes / episodes, "timeouts": timeouts,
        "mean_return": float(np.mean(returns)),
        "mean_discounted_return": float(np.mean(discounted)),
        "mean_steps": float(np.mean(lengths)),
    }


def run_dp_demo(episodes: int = 200, seed: int = 42, output: Path | None = None) -> None:
    """Run known-model planning; this is not neural-network training."""
    if episodes < 1 or seed < 0:
        raise ValueError("Episodes must be positive and seed nonnegative.")
    if output is not None and output.exists():
        raise FileExistsError(f"Report already exists; choose a new path: {output}")
    config = TeachingConfig(hazard_cell=(3, 1), hazard_penalty=4)
    model = driving_model(config)
    random_values = policy_evaluation(model, uniform_policy(model), GAMMA)
    improved = policy_iteration(model, GAMMA)
    print("Week 3: known-model policy evaluation and policy iteration")
    print("Teaching hazard: entering (progress=3, lane=1) costs 4; finish remains reachable.")
    print(f"Random policy V(start):   {random_values.values[START]:.5f}")
    print(f"Improved policy V(start): {improved.values[START]:.5f}")
    print(f"Policy iterations: {improved.iterations}; evaluation sweeps: {improved.evaluation_sweeps}")
    print(f"Optimality residual: {improved.residual:.2e}")
    print("\nImproved action and V(s): F=forward, L=left, R=right")
    print("             Left           Center          Right")
    labels = ("F", "L", "R")
    for progress in range(5, -1, -1):
        cells = []
        for lane in range(LANES):
            state = progress * LANES + lane
            action = int(np.argmax(improved.policy[state]))
            mark = "*" if (progress, lane) == config.hazard_cell else " "
            cells.append(f"{labels[action]} {improved.values[state]:7.4f}{mark}")
        print(f"row {progress}:   " + "     ".join(cells))
    print("* marks the hazardous arrival cell; V(s) excludes past rewards.")

    runner = DrivingMDP(config)
    route = [runner.state]
    rewards = []
    actions = []
    while runner.state != FINISH:
        column = int(np.argmax(improved.policy[runner.state]))
        action = model.action_ids[column]
        outcome = runner.step(action)
        route.append(runner.state)
        actions.append(action)
        rewards.append(outcome.reward)
    print(f"\nImproved route (state IDs): {route}")
    print(f"Rewards: {rewards}; discounted return: {discounted_return(rewards):.5f}")

    report = {
        "method": "tabular_policy_iteration", "gamma": GAMMA, "tolerance": 1e-10,
        "software": {"gymnasium": gym.__version__, "numpy": np.__version__},
        "teaching_mdp": {
            "config": asdict(config), "random_start_value": float(random_values.values[START]),
            "random_evaluation_sweeps": random_values.sweeps,
            "random_residual": random_values.residual,
            "improved_start_value": float(improved.values[START]),
            "policy_iterations": improved.iterations,
            "evaluation_sweeps": improved.evaluation_sweeps, "residual": improved.residual,
            "values": improved.values.tolist(), "policy": improved.policy.tolist(),
            "route": route, "actions": actions, "rewards": rewards,
        }, "frozen_lake": [],
    }
    for slippery in (False, True):
        with gym.make("FrozenLake-v1", map_name="4x4", is_slippery=slippery,
                      max_episode_steps=100) as env:
            lake_model = gymnasium_model(env)
            random = uniform_policy(lake_model)
            evaluated = policy_evaluation(lake_model, random, GAMMA)
            planned = policy_iteration(lake_model, GAMMA)
            random_results = _evaluate_lake(env, random, episodes, seed)
            improved_results = _evaluate_lake(env, planned.policy, episodes, seed)
            name = "slippery" if slippery else "deterministic"
            print(f"\nFrozenLake ({name}), {episodes} evaluation episodes per policy:")
            print(f"  Random:   V(start)={evaluated.values[0]:.6f}, success={random_results['success_rate']:.1%}")
            print(f"  Improved: V(start)={planned.values[0]:.6f}, success={improved_results['success_rate']:.1%}")
            report["frozen_lake"].append({
                "slippery": slippery, "random_start_value": float(evaluated.values[0]),
                "random_sweeps": evaluated.sweeps, "random_residual": evaluated.residual,
                "improved_start_value": float(planned.values[0]),
                "iterations": planned.iterations, "evaluation_sweeps": planned.evaluation_sweeps,
                "residual": planned.residual, "values": planned.values.tolist(),
                "policy": planned.policy.tolist(),
                "random_evaluation": random_results, "improved_evaluation": improved_results,
            })
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
            file.write("\n")
        print(f"\nSaved: {output}")
