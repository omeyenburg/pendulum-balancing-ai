# original assests: https://flappybird.fandom.com/wiki/Sprites
import pygame.freetype
import pygame
import game
import ai
import os


speedup = 1


images = {
    "background": [0, 0, 144, 256],
    "ground": [146, 0, 154, 56],
    "logo": [146, 173, 96, 22],
    "game_over": [146, 199, 94, 19],
    "get_ready": [87, 22, 146, 221],
    "pipe_top": [302, 0, 26, 135],
    "pipe_bottom": [330, 0, 26, 121],
    "tap": [153, 41, 58, 30],
    "0": [288, 100, 7, 10],
    "1": [289, 118, 7, 10],
    "2": [289, 134, 7, 10],
    "3": [289, 150, 7, 10],
    "4": [287, 173, 7, 10],
    "5": [287, 185, 7, 10],
    "6": [165, 245, 7, 10],
    "7": [175, 245, 7, 10],
    "8": [185, 245, 7, 10],
    "9": [195, 245, 7, 10],
    "+": [190, 231, 5, 5],
    "bird_0": [223, 124, 17, 12],
    "bird_1": [264, 90, 17, 12],
    "bird_2": [264, 64, 17, 12],
    "medal_bronze": [302, 137, 22, 22],
    "medal_silver": [220, 144, 22, 22],
    "medal_gold": [242, 229, 22, 22],
    "medal_platinum": [266, 229, 22, 22],
}


class Window:
    def __init__(self):
        self.scale = 2
        self.width = 144
        self.height = 256

        pygame.init()
        pygame.display.set_caption("Flappy Bird")
        self.window = pygame.display.set_mode(
            (self.width * self.scale, self.height * self.scale)
        )
        self.pressed = False

        sprites_path = os.path.join(os.path.split(__file__)[0], "assets", "sprites.png")
        self.atlas = pygame.image.load(sprites_path).convert_alpha()
        self.clock = pygame.time.Clock()
        self.time = 0
        self.font = pygame.freetype.SysFont(None, 1)

        self.game = game.Game()
        self.agent = ai.Agent.load()
        self.real_time = ai.seconds_to_str(self.agent.time)
        self.virtual_time = ai.seconds_to_str(self.agent.ticks / 60)

    def draw(self, image, pos=(0, 0), angle=0.0):
        surf = pygame.Surface(images[image][2:], pygame.SRCALPHA)
        surf.blit(self.atlas, (0, 0), images[image])

        surf = pygame.transform.scale2x(surf)
        if angle:
            surf = pygame.transform.rotate(surf, -angle)

        self.window.blit(
            surf,
            (pos[0] * self.scale, pos[1] * self.scale),
        )

    def update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self.game.space()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.game.space()

        # Background
        self.draw("background", (0, 0))
        x_offset = self.width // 3

        # Pipes
        for pipe in self.game.pipes:
            self.draw(
                "pipe_top",
                (x_offset + pipe.x - self.game.bird.x, self.height - pipe.y - 182),
            )
            self.draw(
                "pipe_bottom",
                (x_offset + pipe.x - self.game.bird.x, self.height - pipe.y),
            )

        # Bird
        if self.game.bird.yvel > -2:
            bird_angle = -20
        else:
            bird_angle = min(90, -(self.game.bird.yvel + 2) * 60 - 20)

        if self.game.bird.dead:
            self.draw(
                "bird_1",
                (x_offset, self.height - self.game.bird.y - images["ground"][3]),
                bird_angle,
            )
        else:
            self.draw(
                "bird_" + str(int(self.time * 6) % 3),
                (x_offset, self.height - self.game.bird.y - images["ground"][3]),
                bird_angle,
            )

        # Ground
        ground_x = -self.game.bird.x % images["ground"][3]
        self.draw("ground", (ground_x, self.height - images["ground"][3]))
        self.draw(
            "ground",
            (ground_x - images["ground"][2], self.height - images["ground"][3]),
        )

        # Score
        score = str(self.game.score)
        width = (images["0"][2] + 1) * len(score) - 1
        right = self.width // 2 - width // 2
        for s in score:
            self.draw(s, (right, self.height // 4))
            right += images[s][2] + 1

        # Ai
        for pipe in self.game.pipes:
            if pipe.x - self.game.bird.x > -pipe.width:
                next_pipe = pipe
                break

        inputs = [
            self.game.bird.y,
            self.game.bird.yvel,
            next_pipe.y,
        ]
        output = self.agent.run(inputs)
        if output[0] > 0.5:
            self.game.jump()

        w = self.width * self.scale // (len(self.agent.layers) + 1)
        h = max(self.agent.layers)
        node_size = 10
        positions = []
        for x, layer in enumerate(self.agent.layers):
            positions.append([])
            for y in range(layer):
                positions[-1].append((w + w * x, 20 + 30 * (h / 2 - layer / 2 + y)))

        for i, layer in enumerate(self.agent.layers):
            for j, pos in enumerate(positions[i]):
                if i + 1 < len(self.agent.layers):
                    for k, pos2 in enumerate(positions[i + 1]):
                        weight = abs(self.agent.weights[i][j][k]) / 2
                        color = (50 - 20 * weight, 50 - 20 * weight, 50 - 30 * weight)
                        pygame.draw.line(
                            self.window, color, pos, pos2, round(5 * weight)
                        )

                value = self.agent.values[i][j]
                if i + 1 == len(self.agent.layers) and value > 0.49:
                    pygame.draw.circle(self.window, (100, 250, 100), pos, node_size)
                else:
                    pygame.draw.circle(self.window, (250, 250, 250), pos, node_size)
                if value > 1:
                    value = min(1, value / 200)
                pygame.draw.circle(
                    self.window, (200, 50, 0), pos, round(node_size * value)
                )
                pygame.draw.circle(self.window, (0, 0, 0), pos, node_size, 1)

                if i == 0:
                    text = ("y", "velocity", "y-pipe")[j]
                    self.font.render_to(
                        self.window, (3, pos[1] - 5), text, (0, 0, 0), size=12
                    )
                elif layer == 1:
                    self.font.render_to(
                        self.window,
                        (self.width * self.scale - 35, pos[1] - 5),
                        "jump",
                        (0, 0, 0),
                        size=12,
                    )

        texts = (
            "Real time: " + self.real_time,
            "Virtual time: " + self.virtual_time,
            "Generation: " + str(self.agent.generation),
        )

        for i, text in enumerate(texts):
            self.font.render_to(
                self.window,
                (5, self.height * self.scale - 55 - 20 * i),
                text,
                (0, 0, 0),
                size=14,
            )

        # Update
        self.game.tick(1 / 60)
        pygame.display.flip()
        self.time += self.clock.tick(60 * speedup) / 1000


if __name__ == "__main__":
    window = Window()

    while True:
        window.update()
