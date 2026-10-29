# Small driving MDP: road, states, and actions

This finite teaching model is implemented in `rogue_warrior/driving_mdp.py` for practicing MDPs and dynamic programming. The agent chooses a driving action; the environment updates position and returns a reward and terminal flag. The full motorcycle game will add speed, traffic, combat, and other mechanics later.

## 1. Road, start, and finish

Use a straight road with three lanes and six progress positions before the finish.

| Item | Definition |
| --- | --- |
| Lanes | `0` = left, `1` = center, `2` = right |
| Progress | `0` through `5`; larger values are closer to the finish |
| Start | Center lane at progress `0` |
| Finish | A separate terminal state named `FINISH`, reached after the last progress position |
| Initial conditions | Empty road; no traffic, rivals, combat, police, or random spawning |
| Movement scale | A driving action will advance one progress position; steering can also change lane |

```text
                 Left     Center    Right
Finish              ------ FINISH ------
Progress 5        [     ] [     ] [     ]
Progress 4        [     ] [     ] [     ]
Progress 3        [     ] [     ] [     ]
Progress 2        [     ] [     ] [     ]
Progress 1        [     ] [     ] [     ]
Progress 0        [     ] [ BIKE] [     ]
```

These positions are logical cells, not pixels or distances in meters. The objective is to reach the finish. Rewards and the discount factor are defined below.

## 2. States

A nonterminal state is `(progress, lane)`.

For example, `(2, 0)` means the bike is in the left lane at progress position 2. The initial state is `(0, 1)`.

There are `6 * 3 = 18` nonterminal states and one shared terminal state, `FINISH`: **19 states total**. All finish lanes are merged because the episode has ended and the lane no longer affects future outcomes.

For tabular arrays, encode a nonterminal state as:

```text
state_id = 3 * progress + lane
```

This gives IDs `0` through `17`; reserve ID `18` for `FINISH`. The start state has ID `1`.

The model deliberately has no variable speed, health, or moving objects. Its next outcome will depend only on the current position and chosen action, so we do not need previous positions in the state. If we later add variable speed or moving traffic, we must reconsider the state definition to preserve the Markov property.

## 3. Actions

Use a three-action subset of the full game's centrally defined actions, with the same IDs.

| Action ID | Specification name | Meaning in this teaching model |
| --- | --- | --- |
| `0` | `ACCELERATE` | Advance along the current lane |
| `1` | `ACCELERATE_LEFT` | Advance and request one lane to the left |
| `2` | `ACCELERATE_RIGHT` | Advance and request one lane to the right |

In this small model, acceleration immediately supplies the fixed one-cell forward movement. It has no persistent speed dynamics. This simplification keeps the state small; the playable game's acceleration will use actual speed and physics.

All three actions are available in every nonterminal state. A request to steer beyond an outer lane keeps the bike in that outer lane while it advances. It does not create a fourth lane or an implicit crash outcome.

The model will reject action IDs outside `0`, `1`, and `2`. Braking, coasting, and combat remain outside this teaching model's action subset. Do not renumber the full game's action mapping when those actions are implemented.

`FINISH` ends the episode. The transition model represents it as absorbing for DP calculations, with its terminal value fixed at zero; the episode runner requires a reset before starting another episode.

## 4. Transition model

Let `delta(a)` be `0`, `-1`, or `+1` for actions `0`, `1`, or `2` respectively. For a nonterminal `(progress, lane)`:

```text
next_lane = clamp(lane + delta(a), 0, 2)
next_progress = progress + 1
next_state = FINISH if next_progress == 6
             otherwise (next_progress, next_lane)
```

There is exactly one outcome for each state/action pair, with probability `1`. All other outcomes have probability `0`, so every transition distribution sums to `1`. No seed is needed because there is no randomness.

Examples:

| Current state | Action | Next state | Reward | Terminal? |
| --- | --- | --- | --- | --- |
| `(0, 1)` | `0` | `(1, 1)` | `1` | No |
| `(0, 1)` | `1` | `(1, 0)` | `1` | No |
| `(0, 1)` | `2` | `(1, 2)` | `1` | No |
| `(0, 0)` | `1` | `(1, 0)` | `1` | No |
| `(5, 2)` | `0` | `FINISH` | `6` | Yes |
| `FINISH` | Any valid action in the DP model | `FINISH` | `0` | Yes |

The final row exists for mathematical completeness. `DrivingMDP.step()` rejects steps after termination so the caller must reset. The model exposes `transitions(state, action)` for future policy evaluation and iteration without needing to sample episodes.

## 5. Rewards and discount

- Each forward step earns `+1` for incremental progress.
- Entering `FINISH` adds a one-time `+5` finish bonus: that transition's total reward is `6`.
- The absorbing model's finish self-loop earns `0`, so it cannot repeat the finish bonus.
- Use `gamma = 0.9` for the teaching example.

These are teaching constants, not tuned rewards for the full game. No collision, steering, or boundary penalty is present. Every episode finishes after exactly six actions, so this model has no timeout or failure outcome.

## 6. Worked trajectory and return

A forward-only sequence follows:

```text
(0,1) -> (1,1) -> (2,1) -> (3,1) -> (4,1) -> (5,1) -> FINISH
actions: 0, 0, 0, 0, 0, 0
rewards: 1, 1, 1, 1, 1, 6
```

The undiscounted episode reward is `11`. The discounted return is:

```text
G_0 = r_1 + gamma*r_2 + gamma^2*r_3 + ... + gamma^5*r_6
    = 1 + 0.9 + 0.81 + 0.729 + 0.6561 + 0.59049*6
    = 7.63804
```

Reward is the feedback from a single transition. Return combines the rewards that follow a state. A bandit chooses an action for an immediate outcome; here the action changes the state from which later decisions and rewards follow.

## 7. Run and check

From the `Rogue_warrior` folder, using Python 3.11 or newer:

```powershell
uv run -m rogue_warrior demo
uv run -m rogue_warrior play
uv run -m rogue_warrior dp-demo
uv run -m unittest discover -s tests -v
```

`demo` drives straight and prints the worked return. `play` lets a human select actions in the terminal, including `q` to exit. These commands still use the original empty teaching model. `dp-demo` uses the configurable hazardous-cell variant described in policy_iteration.md. The graphical game and physical Gymnasium environment are separate from this finite model.

The tests check every state/action distribution, steering examples, all 729 six-action routes, terminal rewards, reset behavior, invalid inputs, and the return calculation.

## Current limitation

On the default empty road, every action advances equally and every route earns the same return. The Week 3 DP experiment now introduces a fixed hazardous cell at `(3, 1)` with an arrival penalty of 4 to make choices meaningful. It does not add obstacles to the graphical game. See policy_iteration.md for the Bellman updates, improved route, and actual results.
