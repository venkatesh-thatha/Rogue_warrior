"""Independent checks for Bellman updates, planning, and terminal handling."""

from itertools import product
import unittest

import gymnasium as gym
import numpy as np

from rogue_warrior.actions import TEACHING_ACTIONS
from rogue_warrior.dp import (
    TabularModel, action_values, driving_model, gymnasium_model,
    policy_evaluation, policy_iteration, uniform_policy,
)
from rogue_warrior.driving_mdp import (
    DrivingMDP, FINISH, START, TeachingConfig, Transition, discounted_return, transitions,
)


class DPTests(unittest.TestCase):
    def test_hazard_applies_on_arrival_and_empty_model_stays_unchanged(self) -> None:
        config = TeachingConfig(hazard_cell=(3, 1))
        self.assertEqual(transitions(7, 0)[0].reward, 1)
        self.assertEqual(transitions(7, 0, config)[0].reward, -3)
        self.assertEqual(transitions(7, 1, config)[0].reward, 1)
        self.assertEqual(transitions(10, 0, config)[0].reward, 1)
        self.assertEqual(transitions(FINISH, 0, config)[0].reward, 0)

    def test_driving_values_match_exhaustive_routes(self) -> None:
        model = driving_model(TeachingConfig(hazard_cell=(3, 1)))
        route_returns = []
        for actions in product(TEACHING_ACTIONS, repeat=6):
            runner = DrivingMDP(TeachingConfig(hazard_cell=(3, 1)))
            rewards = [runner.step(action).reward for action in actions]
            route_returns.append(discounted_return(rewards))
        random = policy_evaluation(model, uniform_policy(model))
        improved = policy_iteration(model)
        self.assertAlmostEqual(random.values[START], np.mean(route_returns))
        self.assertAlmostEqual(random.values[START], 6.55804)
        self.assertAlmostEqual(improved.values[START], max(route_returns))
        self.assertAlmostEqual(improved.values[START], 7.63804)
        self.assertEqual(improved.values[FINISH], 0)
        self.assertTrue(np.all(improved.values >= random.values - 1e-10))
        self.assertLessEqual(improved.residual, 1e-10)

    def test_stochastic_self_loop_has_known_analytic_value(self) -> None:
        model = TabularModel((
            ((Transition(0.5, 0, 0, False), Transition(0.5, 1, 2, True)),),
            ((Transition(1, 1, 0, True),),),
        ), (0,))
        evaluation = policy_evaluation(model, uniform_policy(model), gamma=0.9)
        self.assertAlmostEqual(evaluation.values[0], 1 / (1 - 0.5 * 0.9))
        self.assertLessEqual(evaluation.residual, 1e-10)

    def test_terminal_transition_never_bootstraps(self) -> None:
        model = TabularModel((((Transition(1, 0, 2, True),),),), (0,))
        q_values = action_values(model, np.array([999.0]), 0.9)
        self.assertEqual(q_values[0, 0], 2)

    def test_empty_road_ties_stabilize_and_preserve_value(self) -> None:
        model = driving_model()
        improved = policy_iteration(model)
        self.assertEqual(improved.iterations, 2)
        self.assertAlmostEqual(improved.values[START], 7.63804)
        np.testing.assert_array_equal(improved.policy[:, 0], np.ones(model.state_count))

    def test_frozen_lake_random_values_match_independent_linear_solution(self) -> None:
        for slippery in (False, True):
            with self.subTest(slippery=slippery), gym.make("FrozenLake-v1", is_slippery=slippery) as env:
                model = gymnasium_model(env)
                policy = uniform_policy(model)
                rewards = np.zeros(model.state_count)
                continuation = np.zeros((model.state_count, model.state_count))
                for state, row in enumerate(model.outcomes):
                    for action, outcomes in enumerate(row):
                        for item in outcomes:
                            weight = policy[state, action] * item.probability
                            rewards[state] += weight * item.reward
                            if not item.terminated:
                                continuation[state, item.next_state] += weight
                exact = np.linalg.solve(np.eye(model.state_count) - 0.9 * continuation, rewards)
                evaluated = policy_evaluation(model, policy)
                np.testing.assert_allclose(evaluated.values, exact, rtol=0, atol=1e-8)
                improved = policy_iteration(model)
                self.assertTrue(np.all(improved.values >= evaluated.values - 1e-8))
                self.assertLessEqual(improved.residual, 1e-9)
                if not slippery:
                    self.assertAlmostEqual(improved.values[0], 0.9 ** 5)

    def test_invalid_policy_and_iteration_caps_are_reported(self) -> None:
        model = driving_model()
        for invalid in (np.zeros((19, 3)), np.ones((19, 3)), np.full((19, 3), np.nan)):
            with self.assertRaises(ValueError):
                policy_evaluation(model, invalid)
        with self.assertRaises(RuntimeError):
            policy_evaluation(model, uniform_policy(model), max_sweeps=1)
        with self.assertRaises(RuntimeError):
            policy_iteration(model, max_iterations=1)
        with self.assertRaises(ValueError):
            policy_evaluation(model, uniform_policy(model), gamma=1.1)
        with self.assertRaises(ValueError):
            TeachingConfig(hazard_cell=(0, 1))
        with self.assertRaises(ValueError):
            TabularModel((((Transition(0.5, 0, 0, True),),),), (0,))


if __name__ == "__main__":
    unittest.main()
