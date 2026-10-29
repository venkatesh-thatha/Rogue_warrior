# Empty-road driving simulation

`rogue_warrior/game.py` implements the first physical driving core, separate from the finite teaching MDP. It uses continuous-valued position and speed, with one fixed simulation tick per action. It has no rendering or RL dependencies.

## State and units

`GameConfig` holds the prototype settings. `GameWorld` owns the state and exposes immutable `BikeState` snapshots.

| Quantity | Default | Meaning |
| --- | --- | --- |
| Time step | 1/60 second | Exactly one tick per `step(action)` |
| Road width | 12 meters | Straight road; x increases from left to right |
| Bike width | 0.8 meter | The entire bike stays within the road |
| Initial x / speed | 6 meters / 0 m/s | Stationary start in the road center |
| Maximum speed | 35 m/s (126 km/h) | Speed cap |
| Engine acceleration | 8 m/s² | Applied for actions 0, 1, and 2 |
| Braking deceleration | 14 m/s² | Applied for action 3 |
| Passive drag | 1.2 m/s² | Constant prototype deceleration, including under throttle |
| Maximum lateral speed | 4.5 m/s | Scaled by forward speed |
| Race distance | 500 meters | Short empty-road race |
| Maximum duration | 60 simulated seconds | Idle driving eventually times out |

These values are initial design choices, not calibrated motorcycle physics. Time shown by the headless demo is simulated time; the computer can calculate a race much faster than real time.

## Actions and updates

The IDs are centralized in `rogue_warrior/actions.py`:

- `0`: accelerate straight.
- `1`: accelerate and steer left.
- `2`: accelerate and steer right.
- `3`: brake without steering.
- `4`: coast without steering.

Combat IDs are reserved in the shared enum but rejected by this prototype. The teaching MDP still accepts only 0, 1, and 2.

Each tick updates speed using acceleration minus drag, then clamps speed between zero and the configured maximum. Forward distance uses the average of the old and new speeds times the fixed time step. Steering uses that same average speed to scale lateral movement; a stopped bike does not move sideways without starting to drive.

The bike center is clamped between half the bike width and the road width minus half the bike width. Road-edge contact has no damage or reward yet. It is a simple boundary rule for this first prototype.

Race distance is capped at the finish. A finish sets `termination_reason="finish"` and `terminated=True`; reaching the time cutoff sets `termination_reason="timeout"` and `truncated=True`. Outcomes are detected on tick boundaries; finishing takes precedence if both thresholds are reached on the same tick. After either outcome, call `reset()` before stepping again.

No randomness is present. Given the same configuration and action sequence, snapshots repeat exactly. Reading state does not advance physics. Rewards and Gymnasium spaces will be introduced in the environment wrapper; the physical game rules do not calculate RL reward.

## Run and verify

From `Rogue_warrior`:

```powershell
uv sync
uv run -m rogue_warrior drive-demo
uv run -m unittest discover -s tests -v
```

The demo uses constant acceleration as a simple scripted driver, not a learned agent. Tests cover acceleration and speed limits, braking versus coasting, steering boundaries, idle timeout, finish/reset, repeatability, and invalid inputs.

The core was originally verified with Python 3.11.9 without graphics dependencies. The visual increment uses Pygame 2.6.1 and a uv-managed environment with versions recorded in uv.lock. The original 15 teaching/physics checks are included in the expanded suite.

## Play visually

```powershell
uv run -m rogue_warrior game
```

The 320x240 procedural scene is scaled to a 960x720 window using nearest-neighbor scaling. Following the learner's reference, the view is now a third-person camera behind the rider. A pseudo-3D road narrows toward a horizon with hills and roadside trees. Projection uses distance ahead and lateral world position relative to the bike; steering shifts the road while the rider stays centered. Road markings and scenery move with traveled distance. The physical road is still straight and flat; the hills and trees are decoration.

Following the learner's camera clarification, the HUD is a separate rectangular screen overlay showing speed, progress, and remaining time. It is not a cockpit. The rider remains visible from behind in the world view. All values come from the same world snapshot. Health, stamina, rival information, and race positions will be added when those mechanics exist. Graphics are procedural and do not use copied game sprites or HUD artwork.

The Gymnasium adapter now wraps this simulation; see `environment.md` for observations, actions, rewards, and baseline commands.

Up accelerates; left/right select accelerate-left/right even without holding up. Down takes priority and brakes. Opposite steering keys cancel steering but retain acceleration. Releasing the arrows coasts. R resets the race and pending time; Escape and closing the window exit cleanly.

The display loop accumulates wall time and consumes it in fixed physics ticks. Catch-up is capped at 0.25 seconds per display frame after a stall, so the race intentionally slows rather than jumping far ahead. Normal 30/60/120 FPS schedules produce identical physics. Rendering only reads snapshots.

The default straight-driving race finishes in 1012 ticks (about 16.87 simulated seconds). Automated SDL dummy-driver checks cover controls, render invariance, finish/restart, and window exit; a human play-through on the desktop remains a separate usability check.
