"""Seeded random-action evaluation; this policy does not learn."""

from dataclasses import asdict
import json
from pathlib import Path
import platform
from typing import Any

import gymnasium
import numpy

from .env import DrivingEnv


def evaluate_random(episodes: int = 3, seed: int = 42) -> dict[str, Any]:
    """Run uniform random actions on deterministic empty-road scenarios."""
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 1:
        raise ValueError("Episodes must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Seed must be a nonnegative integer.")
    env = DrivingEnv()
    results = []
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            env.reset(seed=episode_seed)
            # Gymnasium's action-space RNG is seeded separately from the env.
            env.action_space.seed(episode_seed)
            finished = False
            while not finished:
                _, _, terminated, truncated, info = env.step(env.action_space.sample())
                finished = terminated or truncated
            results.append({
                "seed": episode_seed,
                "steps": info["steps"],
                "elapsed_seconds": info["elapsed"],
                "progress": info["race_progress"],
                "return": info["episode_return"],
                "outcome": info["termination_reason"],
            })
        return {
            "policy": "uniform_random",
            "game_config": asdict(env.world.config),
            "reward_config": asdict(env.rewards),
            "software": {
                "python": platform.python_version(),
                "gymnasium": gymnasium.__version__,
                "numpy": numpy.__version__,
            },
            "episodes": results,
            "completion_rate": sum(item["outcome"] == "finish" for item in results) / episodes,
            "mean_progress": sum(item["progress"] for item in results) / episodes,
            "mean_return": sum(item["return"] for item in results) / episodes,
        }
    finally:
        env.close()


def run_random_baseline(episodes: int, seed: int, output: Path | None = None) -> None:
    """Print episode summaries and optionally save a new JSON report."""
    if output is not None and output.exists():
        raise FileExistsError(f"Report already exists; choose a new path: {output}")
    report = evaluate_random(episodes, seed)
    print("Random-action baseline (no learning)")
    for result in report["episodes"]:
        print(
            f"seed={result['seed']} | {result['outcome']} | steps={result['steps']}"
            f" | progress={result['progress']:.1%} | return={result['return']:.4f}"
        )
    print(f"Completion: {report['completion_rate']:.1%}; mean progress: {report['mean_progress']:.1%}")
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as file:
            json.dump(report, file, indent=2)
            file.write("\n")
        print(f"Saved: {output}")
