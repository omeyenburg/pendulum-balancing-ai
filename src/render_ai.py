from pendulum import Pendulum
from geometry import Vec
import pygame.freetype
import pygame.gfxdraw
import pygame
import math
import ai


GENERATION = ai.argv("gen", -1)

WIDTH = 540
HEIGHT = 675

WHITE = (200, 200, 200)
GRAY = (100, 100, 100)
BLACK = (0, 0, 0)


class Window:
    def __init__(self):
        self.agent: ai.Agent = ai.Agent.load(GENERATION)
        self.pendulum = Pendulum()
        self.ai_enabled = True

        pygame.init()
        self.window = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.freetype.SysFont(None, 14)

        self.real_time = ai.seconds_to_str(self.agent.time)
        self.virtual_time = ai.seconds_to_str(self.agent.ticks / 60)

        self.center = Vec(WIDTH // 2, HEIGHT * 2 // 3)
        self.unit_length = abs(WIDTH // 8 - WIDTH // 2)

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit
            if event.type == pygame.MOUSEWHEEL:  # Accelerate with mouse wheel
                self.pendulum.apply_acceleration(Vec(event.y * 10, 0))
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Reset with left click
                    self.pendulum = Pendulum()
                elif event.button == 3:  # Toggle ai with right click
                    self.ai_enabled = not self.ai_enabled

        self.draw()
        self.forward_ai()
        self.pendulum.update()

        pygame.display.flip()
        self.clock.tick(60)
        self.window.fill(BLACK)

    def draw(self):
        # Draw rail
        self.draw_rail()

        # Draw pendulum
        self.draw_pendulum()

        # Draw neurons
        neuron_positions = self.get_neuron_positions()
        for i, layer in enumerate(self.agent.layers):
            for j in range(layer):
                self.draw_neuron(i, j, neuron_positions)

        # Draw info texts
        self.draw_info()

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

    def get_neuron_positions(self):
        w = WIDTH // (len(self.agent.layers) + 1)
        h = max(self.agent.layers)

        positions = [
            [(w + w * x, 20 + 30 * (h / 2 - layer / 2 + y)) for y in range(layer)]
            for x, layer in enumerate(self.agent.layers)
        ]
        # for x, layer in enumerate(self.agent.layers):
        #     positions.append([])
        #     for y in range(layer):
        #         positions[-1].append((w + w * x, 20 + 30 * (h / 2 - layer / 2 + y)))

        return positions

    def draw_neuron(self, i, j, positions):
        node_size = 10
        pos = positions[i][j]

        # Draw weight lines
        if i + 1 < len(self.agent.layers):
            for k, pos2 in enumerate(positions[i + 1]):
                weight = abs(self.agent.weights[i][j][k])
                color = (
                    200 - 50 * weight,
                    200 - 50 * weight,
                    200 - 50 * weight,
                )
                for y in range(round(weight * 3)):
                    pygame.draw.aaline(
                        self.window,
                        color,
                        (pos[0], pos[1] + y * 0.8 - weight),
                        (pos2[0], pos2[1] + y * 0.8 - weight),
                    )

        # Draw neuron background
        pygame.draw.circle(self.window, BLACK, pos, node_size)

        # Draw neuron value
        value = self.agent.values[i][j]
        neuron_size = round(node_size * max(min(abs(value), 1), 0.1))
        value = ai.ActivationFunction.tanh(value * -5)

        if value > 0:
            color = (100, 100 + int(150 * (1 - value)), 100 + int(150 * value))
        else:
            color = (100 + int(150 * -value), 100 + int(150 * (1 + value)), 100)

        pygame.draw.circle(self.window, color, pos, neuron_size)

        # Draw neuron border
        pygame.draw.circle(self.window, WHITE, pos, node_size, 1)

        # Draw neuron labels
        if i == 0:
            self.font.render_to(
                self.window,
                (10, pos[1] - 5),
                ("cart.x", "cart.vel", "bob.x", "bob.y", "bob.vel")[j],
                WHITE,
                size=14,
            )
        elif i + 1 == len(self.agent.layers):
            self.font.render_to(
                self.window,
                (WIDTH - 90, pos[1] - 5),
                "acceleration",
                WHITE,
                size=14,
            )

    def draw_info(self):
        texts = (
            "Real training time: " + self.real_time,
            "Simulated training time: " + self.virtual_time,
            "Generation: " + str(self.agent.generation),
        )

        for i, text in enumerate(texts):
            self.font.render_to(
                self.window,
                (10, HEIGHT - 110 - 30 * i),
                text,
                WHITE,
                size=14,
            )

    def forward_ai(self):
        if not self.ai_enabled:
            return

        output = self.agent.run(
            self.pendulum.x,
            self.pendulum.horizontal_velocity,
            math.cos(self.pendulum.angle),
            math.sin(self.pendulum.angle),
            self.pendulum.angular_velocity,
        )

        acceleration = Vec(output[0] * 30, 0)
        self.pendulum.apply_acceleration(acceleration)


if __name__ == "__main__":
    window = Window()
    while True:
        window.update()
