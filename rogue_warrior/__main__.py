"""Run the teaching model or a headless empty-road physics demo."""

import argparse
from pathlib import Path

from .actions import TEACHING_ACTIONS
from .driving_mdp import Action, DrivingMDP, FINISH, GAMMA, LANES, discounted_return
from .game import GameWorld


def describe(state: int) -> str:
    """Format a tabular state for a human reader."""
    if state == FINISH:
        return "FINISH"
    progress, lane = divmod(state, LANES)
    cells = ["[BIKE]" if index == lane else "[    ]" for index in range(LANES)]
    return f"progress={progress}/6, lane={lane}: {' '.join(cells)}"


def run(interactive: bool) -> None:
    """Run one six-action episode without graphics or external dependencies."""
    model = DrivingMDP()
    rewards: list[float] = []
    print("Rogue Warrior: finite driving MDP")
    print(describe(model.state))
    while model.state != FINISH:
        if interactive:
            raw_action = input("0=forward, 1=forward-left, 2=forward-right, q=quit: ").strip()
            if raw_action.lower() == "q":
                print("Stopped before the finish.")
                return
            try:
                action = Action(int(raw_action))
                if action not in TEACHING_ACTIONS:
                    raise ValueError("Choose a teaching-model action.")
            except ValueError:
                print("Choose 0, 1, 2, or q.")
                continue
        else:
            action = Action.ACCELERATE
        outcome = model.step(action)
        rewards.append(outcome.reward)
        print(f"{action.name} -> {describe(model.state)}, reward={outcome.reward:g}")
    print(f"Steps: {len(rewards)}; total reward: {sum(rewards):g}")
    print(f"Discounted return (gamma={GAMMA}): {discounted_return(rewards):.5f}")


def run_driving_demo() -> None:
    """Accelerate through a short race, printing simulated seconds."""
    world = GameWorld()
    print("Rogue Warrior: empty-road physics (headless)")
    print("Action 0: accelerate straight; simulation runs without a window.")
    while world.state.termination_reason is None:
        state = world.step(Action.ACCELERATE)
        if state.steps % 60 == 0 or state.termination_reason is not None:
            print(
                f"t={state.elapsed:.2f}s | speed={state.speed * 3.6:.1f} km/h"
                f" | distance={state.distance:.1f}/{world.config.race_distance:g}m"
                f" | x={state.x:.2f}m"
            )
    print(f"Outcome: {world.state.termination_reason}; ticks: {world.state.steps}")


def main() -> None:
    """Parse the terminal mode and start an episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("demo", "play", "drive-demo", "game", "random-agent", "dp-demo"))
    parser.add_argument("--episodes", type=int, default=3, help="Random baseline episode count")
    parser.add_argument("--seed", type=int, default=42, help="Random baseline starting seed")
    parser.add_argument("--output", type=Path, help="Save a new baseline or DP JSON report")
    parser.add_argument("--eval-episodes", type=int, default=200, help="FrozenLake episodes per DP policy")
    parser.add_argument("--watch", action="store_true", help="Watch one random-agent race in a window")
    args = parser.parse_args()
    if args.watch and args.mode != "random-agent":
        parser.error("--watch is available with random-agent")
    if args.watch and args.output is not None:
        parser.error("Run without --watch to save a baseline report")
    try:
        if args.mode == "dp-demo":
            from .dp_experiment import run_dp_demo

            if args.eval_episodes < 1 or args.seed < 0:
                parser.error("--eval-episodes must be positive and --seed nonnegative")
            run_dp_demo(args.eval_episodes, args.seed, args.output)
        elif args.mode == "random-agent":
            from .baselines import run_random_baseline

            if args.episodes < 1 or args.seed < 0:
                parser.error("--episodes must be positive and --seed nonnegative")
            if args.watch:
                from .watch import watch_random

                watch_random(args.seed)
            else:
                run_random_baseline(args.episodes, args.seed, args.output)
        elif args.mode == "game":
            from .play import play

            play()
        elif args.mode == "drive-demo":
            run_driving_demo()
        else:
            run(interactive=args.mode == "play")
    except (EOFError, KeyboardInterrupt):
        print("\nStopped.")


if __name__ == "__main__":
    main()
