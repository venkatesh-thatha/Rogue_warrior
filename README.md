# Rogue Warrior

A personal reinforcement learning project: build a retro motorcycle racing and combat game, then use it to study decision-making agents.

Development follows my course progress, one small, verified increment per daily session.

Play a short empty-road motorcycle race with a retro third-person view, a separate speed/progress HUD, acceleration, braking, and steering. Run these commands inside this folder using `uv`:

```powershell
# Set up dependencies and open the game
uv sync
uv run -m rogue_warrior game

# Run a random-action baseline through Gymnasium
uv run -m rogue_warrior random-agent --episodes 3 --seed 42

# Watch the random agent drive
uv run -m rogue_warrior random-agent --watch --seed 42

# Practice policy evaluation and iteration on the teaching MDP
uv run -m rogue_warrior dp-demo

# Run all checks
uv run -m unittest discover -s tests -v
```

Up accelerates, left/right accelerate and steer, and down brakes. Release the arrows to coast. **R** restarts; **Esc** quits.

For the terminal learning model, use `uv run -m rogue_warrior play`. For a headless physics demo, use `uv run -m rogue_warrior drive-demo`.
