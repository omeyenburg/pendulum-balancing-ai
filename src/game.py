from geometry import Vec
import math


class Pendulum:
    def __init__(self):
        self.x = 0
        self.angle = math.pi / 2
        self.angular_velocity = 0
        self.horizontal_velocity = 0
        self.radius = 0.5
        self.mass = 10
        self.damping = 0.1
        self.gravity = 9.81

    def apply_acceleration(self, acceleration: Vec, delta_time: float):
        if acceleration.x < 0 and self.x <= -1 or acceleration.x > 0 and self.x >= 1:
            acceleration.x = 0

        # Rotational movement
        force: Vec = acceleration * self.mass
        moment_of_inertia = self.mass * self.radius**2
        angular_acceleration = (
            force.cross(Vec.from_angle(self.angle) * self.radius) / moment_of_inertia
        )

        angular_acceleration -= self.damping * self.angular_velocity
        self.angular_velocity += angular_acceleration * delta_time

        # Horizontal movement
        acceleration.x -= self.damping * self.horizontal_velocity
        self.horizontal_velocity += acceleration.x * delta_time

    def update_velocity(self, delta_time: float):
        # Rotational movement
        self.angle += self.angular_velocity * delta_time

        # Horizontal movement
        self.x += self.horizontal_velocity * delta_time
        if self.x < -1:
            self.x = -1
            self.horizontal_velocity = 0
        elif self.x > 1:
            self.x = 1
            self.horizontal_velocity = 0

    def update(self):
        delta_time = 0.016666666666667
        self.apply_acceleration(Vec(0, -self.gravity), delta_time)
        self.update_velocity(delta_time)


import pygame.freetype
import pygame


WHITE = (200, 200, 200)
GRAY = (100, 100, 100)
BACKGROUND = (0, 0, 0)


class Game:
    def __init__(self):
        self.pendulum = Pendulum()

        self.width = 540
        self.height = 675

        pygame.init()
        self.window = pygame.display.set_mode(
            (self.width, self.height), pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()
        self.delta_time = 1.0
        self.font = pygame.freetype.SysFont(None, 14)

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.MOUSEWHEEL:
                self.pendulum.apply_acceleration(
                    Vec(event.precise_y, 0), self.delta_time
                )

        self.window.fill(BACKGROUND)

        # Draw rail
        center = Vec(self.width // 2, self.height // 2)
        unit_length = abs(self.width // 8 - self.width // 2)
        rail_start = center + Vec(unit_length, 0)
        rail_end = center - Vec(unit_length, 0)
        pygame.draw.line(self.window, GRAY, rail_start.tolist(), rail_end.tolist(), 2)
        rail_sections = 6

        for i in range(rail_sections + 1):
            rail_top = rail_start * i / rail_sections + rail_end * (
                1 - i / rail_sections
            )
            rail_top = rail_top + Vec(0, 3)
            rail_bottom = rail_top + Vec(0, -6)
            pygame.draw.line(
                self.window, WHITE, rail_top.tolist(), rail_bottom.tolist(), 1
            )

        # Draw pendulum
        cart = center + Vec(self.pendulum.x * unit_length, 0)
        weight = (
            cart
            + Vec.from_angle(self.pendulum.angle) * unit_length * self.pendulum.radius
        )
        pygame.draw.line(self.window, GRAY, cart.tolist(), weight.tolist(), 3)
        pygame.draw.circle(self.window, WHITE, cart.tolist(), 3)
        pygame.draw.circle(self.window, WHITE, weight.tolist(), 8)

        self.font.render_to(
            self.window, (0, 0), str(self.pendulum.angular_velocity), WHITE
        )

        pygame.display.flip()
        self.delta_time = self.clock.tick(60) / 1000.0
        self.pendulum.update()


if __name__ == "__main__":
    g = Game()
    while True:
        g.update()
