import game
import ai


def train(agent):
    g = game.Game()
    score = 0

    while g.score < 1000 and not g.bird.dead:
        for pipe in g.pipes:
            if pipe.x - g.bird.x > -pipe.width:
                next_pipe = pipe
                break

        # score -= 0.01 * abs(next_pipe.y - g.bird.y)
        if not (
            g.bird.y + g.bird.height / 2 > next_pipe.y
            or g.bird.y < next_pipe.y - next_pipe.height
        ):
            score += 1

        inputs = [g.bird.y, g.bird.yvel, next_pipe.y]
        output = agent.run(inputs)

        if output[0] > 0.5:
            g.jump()

        g.tick(1 / 60)

    return score


def main():
    rlm = ai.ReinforcementLearningModel(
        train, 30, ["y", "yvel", "ypipe"], ["jump"], [4, 4], "sigmoid"
    )
    rlm.train()


if __name__ == "__main__":
    main()
