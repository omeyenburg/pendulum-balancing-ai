# original assets: https://flappybird.fandom.com/wiki/Sprites
import pygame
import game
import os


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

        self.game = game.Game()

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
        self.pressed = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit()
            elif (
                event.type == pygame.MOUSEBUTTONDOWN
                or event.type == pygame.KEYDOWN
                and event.key == pygame.K_SPACE
            ):
                self.pressed = True

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
            self.draw(s, (right, self.height // 6))
            right += images[s][2] + 1

        if self.pressed:
            self.game.space()
        self.game.tick(1 / 60)

        pygame.display.flip()
        self.time += self.clock.tick(60) / 1000


if __name__ == "__main__":
    window = Window()

    while True:
        window.update()
