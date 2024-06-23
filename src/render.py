from pendulum import Pendulum
from geometry import Vec
import pygame.freetype
import math
import pygame


WIDTH = 540
HEIGHT = 675

WHITE = (200, 200, 200)
GRAY = (100, 100, 100)
BLACK = (0, 0, 0)


class Window:
    def __init__(self):
        self.pendulum = Pendulum()

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
        rail_sections = 4

        for i in range(rail_sections + 1):
            x = rail_start.x * i / rail_sections + rail_end.x * (1 - i / rail_sections)
            rail_top = (x, rail_start.y + 3)
            rail_bottom = (x, rail_start.y + -3)
            pygame.draw.line(self.window, WHITE, rail_top, rail_bottom, 1)

        # Draw pendulum
        cart = center + Vec(self.pendulum.x * unit_length, 0)
        bob = Vec.from_angle(self.pendulum.angle)
        bob = bob * unit_length * self.pendulum.radius + cart
        pygame.draw.line(self.window, GRAY, cart.tolist(), bob.tolist(), 3)
        pygame.draw.circle(self.window, WHITE, cart.tolist(), 3)
        pygame.draw.circle(self.window, WHITE, bob.tolist(), 8)

        pygame.display.flip()
        self.clock.tick(60) / 1000.0
        self.pendulum.update()


if __name__ == "__main__":
    window = Window()
    while True:
        window.update()
