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
        self.pendulum.angle = 0.5

        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()

        self.center = Vec(WIDTH // 2, HEIGHT // 2)
        self.unit_length = abs(WIDTH // 8 - WIDTH // 2)

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            elif event.type == pygame.MOUSEWHEEL:
                acceleration = min(20, max(-20, event.precise_y * 3))
                self.pendulum.apply_acceleration(Vec(acceleration, 0))
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:  # Reset
                    self.pendulum = Pendulum()

        self.draw()
        self.pendulum.update()

        pygame.display.flip()
        dt = self.clock.tick(60)
        self.window.fill(BLACK)

    def draw(self):
        # Draw rail
        self.draw_rail()

        # Draw pendulum
        self.draw_pendulum()

    def draw_rail(self):
        rail_start = self.center + Vec(self.unit_length, 0)
        rail_end = self.center - Vec(self.unit_length, 0)
        pygame.draw.line(self.window, GRAY, rail_start.tolist(), rail_end.tolist(), 2)
        rail_sections = 4

        for i in range(rail_sections + 1):
            x = rail_start.x * i / rail_sections + rail_end.x * (1 - i / rail_sections)
            rail_top = (x, rail_start.y + 3)
            rail_bottom = (x, rail_start.y + -3)
            pygame.draw.line(self.window, WHITE, rail_top, rail_bottom, 1)

    def draw_pendulum(self):
        cart = self.center + Vec(self.pendulum.x * self.unit_length, 0)
        bob = Vec.from_angle(self.pendulum.angle)
        bob = bob * self.unit_length * self.pendulum.radius + cart

        for i in range(-1, 2):
            offset = Vec(
                math.sin(self.pendulum.angle) * i, -math.cos(self.pendulum.angle) * i
            )
            pygame.draw.aaline(
                self.window, GRAY, (cart + offset).tolist(), (bob + offset).tolist()
            )
        pygame.draw.line(self.window, GRAY, cart.tolist(), bob.tolist(), 1)

        pygame.draw.circle(self.window, WHITE, cart.tolist(), 3)
        pygame.draw.circle(self.window, WHITE, bob.tolist(), 8)


if __name__ == "__main__":
    window = Window()
    while True:
        window.update()
