from util import Vec
import math


DELTA_TIME = 1 / 60  # Uncontrolled delta_time leads to uninterntional behaviour


class Pendulum:
    def __init__(self):
        self.x = 0
        self.angle = math.pi / 2
        self.angular_velocity = 0
        self.horizontal_velocity = 0
        self.radius = 0.5
        self.mass = 1

        self.angular_damping = 0.1
        self.horizontal_daming = 0.3
        self.gravity = 9.81

    def apply_acceleration(self, acceleration: Vec):
        if acceleration.x < 0 and self.x <= -1 or acceleration.x > 0 and self.x >= 1:
            acceleration.x = 0

        # Rotational movement
        force: Vec = acceleration * self.mass
        moment_of_inertia = self.mass * self.radius**2
        angular_acceleration = (
            force.cross(Vec.from_angle(self.angle) * self.radius) / moment_of_inertia
        )

        angular_acceleration -= self.angular_damping * self.angular_velocity
        self.angular_velocity += angular_acceleration * DELTA_TIME

        # Horizontal movement
        acceleration.x -= self.horizontal_damping * self.horizontal_velocity
        self.horizontal_velocity += acceleration.x * DELTA_TIME

    def update_velocity(self):
        # Rotational movement
        self.angle += self.angular_velocity * DELTA_TIME

        # Horizontal movement
        self.x += self.horizontal_velocity * DELTA_TIME
        if self.x < -1:
            self.x = -1
            self.horizontal_velocity = 0
        elif self.x > 1:
            self.x = 1
            self.horizontal_velocity = 0

    def update(self):
        self.apply_acceleration(Vec(0, -self.gravity))
        self.update_velocity()
