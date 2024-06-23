from geometry import Vec
import math


# Set fixed delta_time; uncontrolled delta_time leads to uninterntional behaviour
DELTA_TIME = 1 / 60
ANGULAR_DAMPING = 0.7
HORIZONTAL_DAMPING = 0.3
GRAVITY = 9.81


class Pendulum:
    def __init__(self):
        self.x = 0
        self.angle = math.pi / 2
        self.angular_velocity = 0
        self.horizontal_velocity = 0
        self.radius = 0.5
        self.mass = 1

    def apply_acceleration(self, acceleration: Vec):
        if acceleration.x < 0 and self.x <= -1 or acceleration.x > 0 and self.x >= 1:
            acceleration.x = 0

        # Rotational movement
        force: Vec = acceleration * self.mass
        moment_of_inertia = self.mass * self.radius**2
        angular_acceleration = (
            force.cross(Vec.from_angle(self.angle) * self.radius) / moment_of_inertia
        )

        angular_acceleration -= ANGULAR_DAMPING * self.angular_velocity
        self.angular_velocity += angular_acceleration * DELTA_TIME

        # Horizontal movement
        acceleration.x -= HORIZONTAL_DAMPING * self.horizontal_velocity
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
        self.apply_acceleration(Vec(0, -GRAVITY))
        self.update_velocity()
