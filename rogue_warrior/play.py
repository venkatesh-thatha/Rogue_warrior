"""Human input and display loop for the empty-road prototype."""

from collections.abc import Callable

import pygame

from .actions import Action
from .game import GameWorld
from .visuals import Renderer


def driving_action(*, up: bool, down: bool, left: bool, right: bool) -> Action:
    """Map keys to shared actions; braking wins and opposite steering cancels."""
    if down:
        return Action.BRAKE
    if left and not right:
        return Action.ACCELERATE_LEFT
    if right and not left:
        return Action.ACCELERATE_RIGHT
    if up or left or right:
        return Action.ACCELERATE
    return Action.COAST


class FixedStepDriver:
    """Accumulate wall time while advancing physics in fixed ticks."""

    def __init__(self, world: GameWorld, step: Callable[[int], object] | None = None) -> None:
        self.world = world
        self._step = step if step is not None else world.step
        self.accumulator = 0.0

    def advance(self, elapsed: float, action: Action | Callable[[], int]) -> None:
        """Limit catch-up after a long stall to prevent a huge sudden jump."""
        if self.world.state.termination_reason is not None:
            self.accumulator = 0.0
            return
        self.accumulator += min(max(elapsed, 0.0), 0.25)
        dt = self.world.config.time_step
        while self.accumulator + 1e-12 >= dt:
            self._step(action() if callable(action) else action)
            self.accumulator = max(0.0, self.accumulator - dt)
            if self.world.state.termination_reason is not None:
                self.accumulator = 0.0
                break

    def reset(self) -> None:
        self.world.reset()
        self.accumulator = 0.0


def play() -> None:
    """Open the window and run until Escape or a window-close event."""
    pygame.display.init()
    try:
        window = pygame.display.set_mode((960, 720))
        pygame.display.set_caption("Rogue Warrior - Empty Road")
        renderer = Renderer()
        world = GameWorld()
        driver = FixedStepDriver(world)
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
                        driver.reset()
                        elapsed = 0.0
            if not running:
                break
            keys = pygame.key.get_pressed()
            action = driving_action(
                up=keys[pygame.K_UP], down=keys[pygame.K_DOWN],
                left=keys[pygame.K_LEFT], right=keys[pygame.K_RIGHT],
            )
            driver.advance(elapsed, action)
            scene = renderer.draw(world.state, world.config)
            pygame.transform.scale(scene, window.get_size(), window)
            pygame.display.flip()
    finally:
        pygame.quit()
