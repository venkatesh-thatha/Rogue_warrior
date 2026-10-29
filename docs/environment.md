# Gymnasium driving environment

`rogue_warrior.env.DrivingEnv` connects the existing empty-road simulation to agents. It implements `reset`, `step`, `render`, and `close`. Each action advances one 1/60-second physical tick under the default configuration.

## Observation and action contract

The action space is `Discrete(5)`, preserving the shared driving IDs: 0 accelerate, 1 accelerate-left, 2 accelerate-right, 3 brake, and 4 coast. Combat actions remain outside this environment's current action space.

The observation is a `float32` vector with shape `(4,)`, bounded between 0 and 1:

| Index | Feature | Normalization |
| --- | --- | --- |
| 0 | Lateral bike position | x / road width |
| 1 | Speed | speed / maximum speed |
| 2 | Race progress | distance / race distance |
| 3 | Elapsed time | elapsed / maximum duration, capped at 1 |

The reset observation is `[0.5, 0, 0, 0]`. Elapsed time is included because the remaining time affects episode outcomes. In this empty-road model there are no hidden moving objects. Future traffic and combat will need an explicit observation-contract revision.

`step` returns `(observation, reward, terminated, truncated, info)`. Finishing is termination; reaching the configured time cutoff is truncation. Reset before the first action and after either outcome. Invalid actions are rejected without advancing simulation.

## Rewards

`RewardConfig` is independent of `GameConfig`. Default per-step components are:

```text
progress = (new_distance - previous_distance) / race_distance
finish   = +5 only when entering the finish
timeout  = -1 only when the time cutoff ends the episode
reward   = progress + finish + timeout
```

Progress components sum to 1 over a completed race, so the default undiscounted episode return is 6. The teaching MDP uses different reward units (one reward per logical cell); its return should not be compared directly with this physical environment's return.

Steering has no separate reward. A stationary bike earns no progress reward. The finish bonus cannot repeat because stepping after the outcome requires a reset. These are provisional rewards; obstacle avoidance and collision penalties will be introduced when obstacles exist.

`info` exposes the actual physical state summaries, episode return, termination reason, and separately named per-step reward components. Diagnostic information is separate from the four-feature policy observation.

## Use the environment

```python
from rogue_warrior.env import DrivingEnv

with DrivingEnv() as env:
    observation, info = env.reset(seed=42)
    env.action_space.seed(42)
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
```

The physical scenario itself is deterministic. Seed the action space separately to reproduce a random policy's chosen actions. This baseline does not update a policy or learn.

For rendering, construct `DrivingEnv(render_mode="rgb_array")`. `render()` returns a copied RGB `uint8` array shaped `(240, 320, 3)` with the same third-person view and separate HUD seen by a human. It does not open a window or advance physics. Image observations and frame stacking for a pixel policy are later work; this environment still supplies the four-feature structured observation.

## Commands and validation

From `Rogue_warrior`:

```powershell
uv run -m rogue_warrior random-agent --episodes 3 --seed 42
uv run -m rogue_warrior random-agent --watch --seed 42
uv run -m rogue_warrior game
uv run -m unittest discover -s tests -v
```

To save another experiment, add `--output experiments/random_seed100.json` and use the desired seed. Existing reports are not overwritten; select a new filename for another run.

`--watch` opens a real-time window for one random-agent race. The bike acts automatically; R replays the same seed and Escape exits. A random action is sampled for each physics tick, so display frame rate does not change the seeded action sequence. Watching uses the Gymnasium step method, including its rewards and outcome rules. Run the headless command to save reports. This viewer is for the random policy; the Week 3 DP policy belongs to the separate finite teaching model.

The Gymnasium environment checker passed. The 31-test suite covers spaces, finite observations, reward components, finish/reset, timeout, seeding, RGB rendering, physical movement, keyboard input, and the graphical race loop. Tested dependencies are recorded in uv.lock.

## First random baseline

The saved report is `experiments/random_baseline_seed42.json`. Seeds 42, 43, and 44 all finished the empty-road race in 2106, 2377, and 1981 physical steps respectively. The straight-accelerating physics demo finishes in 1012 steps. All three random runs earned an undiscounted return of 6, because the current reward has no elapsed-time penalty for successful races.

These results verify the interaction pipeline and show that the empty-road stage is forgiving. Three successful random episodes do not demonstrate learned skill, safety, or performance with traffic. Obstacles and stronger baselines are needed before making those claims.
