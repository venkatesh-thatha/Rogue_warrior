"""Check the teaching model against its documented finite-MDP contract."""

from itertools import product
import unittest

from rogue_warrior.actions import TEACHING_ACTIONS
from rogue_warrior.driving_mdp import (
    Action, DrivingMDP, FINISH, START, STATE_COUNT, discounted_return, transitions,
)


class DrivingMDPTests(unittest.TestCase):
    def test_all_transition_distributions_are_valid(self) -> None:
        for state in range(STATE_COUNT):
            for action in TEACHING_ACTIONS:
                with self.subTest(state=state, action=action):
                    outcomes = transitions(state, action)
                    self.assertAlmostEqual(sum(item.probability for item in outcomes), 1.0)
                    for outcome in outcomes:
                        self.assertGreater(outcome.probability, 0.0)
                        self.assertIn(outcome.next_state, range(STATE_COUNT))
                        self.assertEqual(outcome.terminated, outcome.next_state == FINISH)

    def test_documented_steering_and_boundary_examples(self) -> None:
        cases = [
            (1, 0, 4),  # center -> center at next progress
            (1, 1, 3),  # center -> left
            (1, 2, 5),  # center -> right
            (0, 1, 3),  # left boundary stays left
            (2, 2, 5),  # right boundary stays right
        ]
        for state, action, expected in cases:
            with self.subTest(state=state, action=action):
                outcome = transitions(state, action)[0]
                self.assertEqual(outcome.next_state, expected)
                self.assertEqual(outcome.reward, 1.0)

    def test_every_six_action_route_finishes_with_the_worked_return(self) -> None:
        # All 729 possible routes must match the empty-road contract.
        for actions in product(TEACHING_ACTIONS, repeat=6):
            model = DrivingMDP()
            rewards = []
            for index, action in enumerate(actions):
                outcome = model.step(action)
                rewards.append(outcome.reward)
                self.assertEqual(outcome.terminated, index == 5)
            self.assertEqual(model.state, FINISH)
            self.assertEqual(rewards, [1, 1, 1, 1, 1, 6])
            self.assertAlmostEqual(discounted_return(rewards), 7.63804)

    def test_finish_is_absorbing_without_repeating_the_bonus(self) -> None:
        for action in TEACHING_ACTIONS:
            outcome = transitions(FINISH, action)[0]
            self.assertEqual(outcome.next_state, FINISH)
            self.assertEqual(outcome.reward, 0.0)

    def test_episode_requires_reset_after_finish(self) -> None:
        model = DrivingMDP()
        for _ in range(6):
            model.step(0)
        with self.assertRaises(RuntimeError):
            model.step(0)
        self.assertEqual(model.reset(), START)
        self.assertEqual(model.step(0).next_state, 4)

    def test_invalid_inputs_do_not_change_episode_state(self) -> None:
        model = DrivingMDP()
        for invalid in (-1, 3, 1.5, "0", True):
            with self.subTest(action=invalid):
                with self.assertRaises(ValueError):
                    model.step(invalid)
                self.assertEqual(model.state, START)
        for invalid in (-1, STATE_COUNT, 1.5, True):
            with self.assertRaises(ValueError):
                transitions(invalid, 0)

    def test_discount_edges_and_invalid_numbers(self) -> None:
        self.assertEqual(discounted_return([1, 1, 1, 1, 1, 6], gamma=1), 11)
        self.assertEqual(discounted_return([1, 6], gamma=0), 1)
        self.assertEqual(discounted_return([]), 0)
        for invalid in (-0.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                discounted_return([1], gamma=invalid)
        with self.assertRaises(ValueError):
            discounted_return([float("nan")])


if __name__ == "__main__":
    unittest.main()
