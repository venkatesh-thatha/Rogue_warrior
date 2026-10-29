"""Original third-person motorcycle graphics with a separate HUD overlay."""

from math import pi

import pygame

from .game import BikeState, GameConfig


class Renderer:
    """Draw a retro chase-camera scene; never advance or modify physics."""

    SIZE = (320, 240)
    HORIZON = 66
    ROAD_BOTTOM = 184

    def __init__(self) -> None:
        pygame.font.init()
        self.surface = pygame.Surface(self.SIZE)
        self.font = pygame.font.Font(None, 16)
        self.small_font = pygame.font.Font(None, 12)
        self.title_font = pygame.font.Font(None, 26)

    def text(self, text: str, position: tuple[int, int], color=(240, 232, 205)) -> None:
        self.surface.blit(self.font.render(text, False, color), position)

    def centered(self, text: str, center: tuple[int, int], font=None, color=(240, 232, 205)) -> None:
        image = (font or self.font).render(text, False, color)
        self.surface.blit(image, image.get_rect(center=center))

    def project(self, ahead: float, lateral: float, state: BikeState, config: GameConfig) -> tuple[int, int]:
        """Project meters ahead and lateral road coordinates onto the scene."""
        scale = 40.0 / (max(ahead, 0.0) + 40.0)
        x = 160 + (lateral - state.x) * (460 / config.road_width) * scale
        y = self.HORIZON + (self.ROAD_BOTTOM - self.HORIZON) * scale
        return round(x), round(y)

    def background(self) -> None:
        screen = self.surface
        screen.fill((73, 112, 58))
        for y in range(self.HORIZON):
            pygame.draw.line(screen, (106 + y // 3, 151 + y // 4, 208 + y // 6), (0, y), (320, y))
        for x, y, width in ((32, 19, 44), (179, 12, 52), (265, 36, 35)):
            pygame.draw.ellipse(screen, (224, 231, 226), (x, y, width, 7))
            pygame.draw.ellipse(screen, (235, 237, 225), (x + 9, y - 4, width // 2, 10))
        pygame.draw.polygon(screen, (94, 124, 129), (
            (0, 72), (0, 60), (38, 48), (75, 63), (108, 52), (155, 68),
            (198, 56), (242, 63), (285, 43), (320, 60), (320, 90)
        ))
        pygame.draw.polygon(screen, (61, 107, 64), (
            (0, 77), (0, 67), (45, 70), (90, 64), (154, 76),
            (215, 64), (262, 58), (320, 67), (320, 94)
        ))

    def road(self, state: BikeState, config: GameConfig) -> None:
        screen = self.surface
        pygame.draw.rect(screen, (73, 112, 58), (0, 74, 320, 112))
        # Draw distant strips first so nearer geometry occludes them.
        for far in range(800, 0, -5):
            near = far - 5
            left_far = self.project(far, 0, state, config)
            right_far = self.project(far, config.road_width, state, config)
            left_near = self.project(near, 0, state, config)
            right_near = self.project(near, config.road_width, state, config)
            band = int((state.distance + near) / 10) % 2
            grass = (73, 112, 58) if band else (78, 117, 61)
            pygame.draw.rect(screen, grass, (0, left_far[1], 320, max(1, left_near[1] - left_far[1])))
            asphalt = (83, 84, 88) if band else (79, 81, 85)
            pygame.draw.polygon(screen, asphalt, (left_far, right_far, right_near, left_near))
            for edge in (0.15, config.road_width - 0.15):
                pygame.draw.polygon(screen, (226, 224, 207), (
                    self.project(far, edge - 0.04, state, config),
                    self.project(far, edge + 0.04, state, config),
                    self.project(near, edge + 0.04, state, config),
                    self.project(near, edge - 0.04, state, config),
                ))
            if int((state.distance + near) / 5) % 4 < 2:
                for lane in (config.road_width / 3, config.road_width * 2 / 3):
                    pygame.draw.polygon(screen, (233, 216, 146), (
                        self.project(far, lane - 0.035, state, config),
                        self.project(far, lane + 0.035, state, config),
                        self.project(near, lane + 0.035, state, config),
                        self.project(near, lane - 0.035, state, config),
                    ))
        left_bottom = self.project(0, 0, state, config)
        right_bottom = self.project(0, config.road_width, state, config)
        pygame.draw.polygon(screen, (79, 81, 85), (
            left_bottom, right_bottom, (right_bottom[0], 240), (left_bottom[0], 240)
        ))
        # Roadside decoration has no collision or reward role.
        offset = state.distance % 32
        for ahead in range(480, 15, -32):
            z = ahead - offset
            scale = 40 / (z + 40)
            for lateral in (-2.5, config.road_width + 2.5):
                x, y = self.project(z, lateral, state, config)
                height = max(3, round(70 * scale))
                width = max(2, round(25 * scale))
                pygame.draw.rect(screen, (99, 67, 42), (x - 1, y - height // 3, 3, height // 3))
                pygame.draw.polygon(screen, (33, 71, 39), (
                    (x, y - height), (x - width, y - height // 5), (x + width, y - height // 5)
                ))
        remaining = config.race_distance - state.distance
        if remaining < 180:
            for row in range(2):
                for column in range(12):
                    color = (236, 229, 208) if (row + column) % 2 else (28, 30, 35)
                    pygame.draw.polygon(screen, color, (
                        self.project(remaining + row * 1.5, column * config.road_width / 12, state, config),
                        self.project(remaining + row * 1.5, (column + 1) * config.road_width / 12, state, config),
                        self.project(remaining + (row + 1) * 1.5, (column + 1) * config.road_width / 12, state, config),
                        self.project(remaining + (row + 1) * 1.5, column * config.road_width / 12, state, config),
                    ))

    def bike(self) -> None:
        """A rider and motorcycle seen from behind, independent of the HUD."""
        screen = self.surface
        pygame.draw.ellipse(screen, (48, 49, 52), (143, 173, 36, 8))
        pygame.draw.rect(screen, (20, 22, 28), (155, 160, 10, 21))
        pygame.draw.rect(screen, (117, 123, 122), (147, 164, 5, 15))
        pygame.draw.rect(screen, (117, 123, 122), (168, 164, 5, 15))
        pygame.draw.polygon(screen, (195, 86, 40), ((145, 156), (175, 156), (169, 174), (151, 174)))
        pygame.draw.rect(screen, (237, 178, 71), (149, 169, 22, 4))
        pygame.draw.rect(screen, (160, 43, 33), (154, 173, 12, 4))
        pygame.draw.line(screen, (22, 26, 34), (150, 152), (145, 165), 5)
        pygame.draw.line(screen, (22, 26, 34), (170, 152), (175, 165), 5)
        pygame.draw.polygon(screen, (44, 52, 77), ((152, 137), (168, 137), (172, 156), (148, 156)))
        pygame.draw.line(screen, (168, 180, 171), (150, 137), (142, 149), 4)
        pygame.draw.line(screen, (168, 180, 171), (170, 137), (178, 149), 4)
        pygame.draw.circle(screen, (193, 53, 40), (160, 131), 9)
        pygame.draw.arc(screen, (246, 193, 91), (152, 123, 16, 15), 0, pi, 2)
        pygame.draw.rect(screen, (44, 42, 45), (157, 137, 6, 2))

    def hud(self, state: BikeState, config: GameConfig, controls_hint: str) -> None:
        """Screen-space race information; it is not a bike cockpit."""
        panel = pygame.Surface((308, 43), pygame.SRCALPHA)
        panel.fill((16, 23, 30, 220))
        self.surface.blit(panel, (6, 191))
        pygame.draw.rect(self.surface, (148, 145, 115), (6, 191, 308, 43), 1)
        self.centered(f"{state.speed * 3.6:03.0f}", (38, 206), self.title_font)
        self.centered("KM/H", (38, 222), self.small_font, (224, 183, 99))
        self.centered(f"{state.distance:.0f} / {config.race_distance:g} M", (159, 202))
        progress = state.distance / config.race_distance
        pygame.draw.rect(self.surface, (66, 70, 72), (87, 214, 145, 5))
        pygame.draw.rect(self.surface, (218, 163, 65), (87, 214, round(145 * progress), 5))
        self.centered("RACE PROGRESS", (159, 226), self.small_font)
        self.centered(f"{max(0, config.max_duration - state.elapsed):04.1f}", (278, 204))
        self.centered("SEC LEFT", (278, 222), self.small_font, (224, 183, 99))
        self.centered(controls_hint, (160, 7), self.small_font)

    def draw(
        self, state: BikeState, config: GameConfig, *,
        controls_hint: str = "ARROWS drive   R restart   ESC quit",
    ) -> pygame.Surface:
        self.background()
        self.road(state, config)
        self.bike()
        self.hud(state, config, controls_hint)
        if state.termination_reason is not None:
            pygame.draw.rect(self.surface, (20, 25, 31), (36, 83, 248, 43))
            pygame.draw.rect(self.surface, (191, 157, 57), (36, 83, 248, 43), 1)
            self.centered("RACE COMPLETE" if state.terminated else "TIME'S UP", (160, 97), self.title_font)
            self.centered(f"{state.elapsed:.2f}s - Press R to race again", (160, 115))
        return self.surface

