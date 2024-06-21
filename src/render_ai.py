from pendulum import Pendulum
from geometry import Vec
import pygame.freetype
import pygame
import math
import ai


WIDTH = 540
HEIGHT = 675

WHITE = (200, 200, 200)
GRAY = (100, 100, 100)
BLACK = (0, 0, 0)


class Window:
    def __init__(self):
        self.pendulum = Pendulum()
        self.agent = ai.Agent.load()

        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.freetype.SysFont(None, 14)

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.MOUSEWHEEL:
                acceleration = min(20, max(-20, event.precise_y * 3))
                self.pendulum.apply_acceleration(Vec(acceleration, 0))

        self.window.fill(BLACK)

        # Draw rail
        center = Vec(WIDTH // 2, HEIGHT // 2)
        unit_length = abs(WIDTH // 8 - WIDTH // 2)
        rail_start = center + Vec(unit_length, 0)
        rail_end = center - Vec(unit_length, 0)
        pygame.draw.line(self.window, GRAY, rail_start.tolist(), rail_end.tolist(), 2)
        rail_sections = 6

        for i in range(rail_sections + 1):
            x = rail_start.x * i / rail_sections + rail_end.x * (1 - i / rail_sections)
            rail_top = (x, rail_start.y + 3)
            rail_bottom = (x, rail_start.y + -3)
            pygame.draw.line(self.window, WHITE, rail_top, rail_bottom, 1)

        # Draw pendulum
        cart = center + Vec(self.pendulum.x * unit_length, 0)
        weight = Vec.from_angle(self.pendulum.angle)
        weight = weight * unit_length * self.pendulum.radius + cart
        pygame.draw.line(self.window, GRAY, cart.tolist(), weight.tolist(), 3)
        pygame.draw.circle(self.window, WHITE, cart.tolist(), 3)
        pygame.draw.circle(self.window, WHITE, weight.tolist(), 8)

        # Run ai
        output = self.agent.run(
            self.pendulum.x,
            self.pendulum.horizontal_velocity,
            math.cos(self.pendulum.angle),
            math.sin(self.pendulum.angle),
            self.pendulum.angular_velocity,
        )

        acceleration = Vec(output[0] * 30, 0)
        self.pendulum.apply_acceleration(acceleration)
        self.pendulum.update()

        pygame.display.flip()
        self.clock.tick(60) / 1000.0


if __name__ == "__main__":
    window = Window()
    while True:
        window.update()
