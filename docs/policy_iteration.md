# Week 3: policy evaluation and policy iteration

Run from `Rogue_warrior`:

```powershell
uv run -m rogue_warrior dp-demo
uv run -m unittest discover -s tests -v
```

This experiment uses the finite teaching MDP and deterministic/slippery FrozenLake. It does not change the playable game's driving behavior. DP plans using a known transition/reward model; it is not a neural network trained from sampled episodes.

## Make lane choices meaningful

The original teaching model remains the default for `demo` and `play`. The DP experiment selects `TeachingConfig(hazard_cell=(3, 1), hazard_penalty=4)`.

Entering the center lane at progress row 3 incurs a penalty of 4, turning that step's reward from `+1` into `-3`. The rider still moves forward and can finish; this is a hazardous-cell cost, not a crash or traffic simulation. All 19 states and the three action IDs are preserved. The hazard's location is fixed and part of the known model, so no extra state variable is needed.

## Policy evaluation: how good is a policy?

A policy table stores the probabilities of choosing each action in each state. The initial random policy uses probability `1/3` for each action.

Starting from zero state values, perform synchronous Bellman expectation updates:

```text
V_new(s) = sum over actions a:
             pi(a|s) * sum over outcomes:
               probability * [reward + gamma * V(next_state)]
```

For terminal outcomes, the continuation value is zero. All updates in a sweep use the previous sweep's values. Stop when the Bellman residual, `max_s |V(s) - BellmanUpdate(V)(s)|`, is at most `1e-10`. A maximum sweep count prevents silent nonconvergence.

The random policy's start-state value is **6.55804** with `gamma=0.9`. This is its expected discounted future return, not an observed score from one episode.

## Policy improvement: choose better actions

Compute one-step lookahead values for every action:

```text
Q(s,a) = sum over outcomes:
           probability * [reward + gamma * V(next_state)]
```

Replace each state's policy with an action maximizing Q, evaluate that policy, and repeat until the action choices stop changing. This applies the policy improvement theorem: choosing greedily with respect to a policy's values yields a policy at least as good under the usual finite discounted-MDP assumptions.

If multiple actions tie, keep the existing deterministic choice when it is within `1e-12` of the maximum. Otherwise choose the first maximizing action column. This gives a stable, reproducible policy rather than cycling between equivalent actions.

At `(progress=2, lane=1)`, the improved policy selects left. Entering the hazardous center cell would have lookahead value `-3 + 0.9*6.76 = 3.084`; moving left has value `1 + 0.9*6.76 = 7.084`. Right is equally good, and the documented tie rule picks left.

The improved start value is **7.63804**, and its route is:

```text
(0,1) -> (1,1) -> (2,1) -> (3,0) -> (4,0) -> (5,0) -> FINISH
actions: 0, 0, 1, 0, 0, 0
rewards: 1, 1, 1, 1, 1, 6
```

The hazard-free discounted return matches the original empty-road example. A straight-only route through the hazard would return `7.63804 - 4*0.9^2 = 4.39804`.

The printed grid contains each state's improved action and V(s). A star marks the hazardous cell. Its state value does not include the arrival penalty already paid before reaching that state: values describe future rewards, not past ones.

## Validation and saved results

`experiments/policy_iteration_seed42.json` records the configuration, policy/value tables, tolerance, sweep counts, residuals, improved route, and FrozenLake evaluations.

The teaching policy stabilized in 2 iterations and 12 total evaluation sweeps with zero measured optimality residual. Its random and improved start values agree with exhaustive enumeration of all 729 six-action routes.

FrozenLake results use gamma 0.9 and the default 4x4 map. Each policy was evaluated on 200 episodes with environment seeds 42-241 and a separately seeded policy-sampling generator:

| Environment | Random V(start) | Improved V(start) | Random success | Improved success |
| --- | --- | --- | --- | --- |
| Deterministic FrozenLake | 0.004477 | 0.590490 | 1.5% | 100% |
| Slippery FrozenLake | 0.004477 | 0.068891 | 3.0% | 73.5% |

V(start) is the calculated discounted-model value. Success rate is an actual finite-sample evaluation metric, with a 100-step external episode cutoff. The DP model itself has no 100-step cutoff, so these are different quantities. The random values coincide because uniform random actions induce the same effective transition distribution in these two maps; sample success rates can still differ.

FrozenLake policy evaluation was also checked against an independently constructed linear-system solution. The deterministic optimal start value agrees with the known six-move route: `0.9^5`. The full suite has 38 passing tests, including stochastic outcomes, terminal handling, policy ties, and convergence caps.

To save another report, use a new filename:

```powershell
uv run -m rogue_warrior dp-demo --eval-episodes 200 --seed 100 --output experiments/policy_iteration_seed100.json
```

These findings demonstrate tabular planning on small known models. They do not establish learned motorcycle control, obstacle avoidance in the physical game, or real-world driving competence.
