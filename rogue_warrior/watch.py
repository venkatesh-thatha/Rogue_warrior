"""Watch the random baseline drive through Gymnasium at real-time speed."""

import pygame

from .env import DrivingEnv
from .play import FixedStepDriver
from .visuals import Renderer


def watch_random(seed: int = 42) -> None:
    """Run one seeded race; R replays it and Escape closes the window."""
    if seed < 0:
        raise ValueError("Seed must be nonnegative.")
    env = DrivingEnv()
    env.reset(seed=seed)
    env.action_space.seed(seed)
    driver = FixedStepDriver(env.world, step=env.step)
    announced = False
    print(f"Watching the random agent (seed={seed}). R replays; Escape quits.")
    try:
        pygame.display.init()
        window = pygame.display.set_mode((960, 720))
        pygame.display.set_caption("Rogue Warrior - Random Agent")
        renderer = Renderer()
        clock = pygame.time.Clock()
        running = True
        while running:
            elapsed = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_r:
                        env.reset(seed=seed)
                        env.action_space.seed(seed)
                        driver.accumulator = 0.0
                        announced = False
                        elapsed = 0.0
            if not running:
                break
            # Sample once per physical tick, not once per display frame.
            driver.advance(elapsed, env.action_space.sample)
            state = env.world.state
            if state.termination_reason is not None and not announced:
                print(f"Outcome: {state.termination_reason}; steps: {state.steps}")
                announced = True
            scene = renderer.draw(
                state, env.world.config,
                controls_hint="RANDOM AGENT   R replay   ESC quit",
            )
            pygame.transform.scale(scene, window.get_size(), window)
            pygame.display.flip()
    finally:
        env.close()
        pygame.quit()
