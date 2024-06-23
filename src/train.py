from pendulum import Pendulum
from geometry import Vec
import random
import math
import ai


AGENT_TIME = ai.argv("time", 60.0) * 60
RANDOM_START = ai.argv("random-start", False)
DISTRACTIONS = ai.argv("distract", False)


def train(agent: ai.Agent):
    pendulum = Pendulum()
    random.seed(agent.generation)

    last_acceleration = 0

    if RANDOM_START and agent.generation % 2 == 1:
        pendulum.x = random.uniform(-0.1, 0.1)
        pendulum.angle = -math.pi / 2 + random.uniform(-0.3, 0.3)
        # pendulum.angular_velocity = random.uniform(-3, 3)
        # pendulum.horizontal_velocity = random.uniform(-3, 3)

    if DISTRACTIONS:
        distraction_time = random.randint(0, AGENT_TIME)
        distraction_strength = random.uniform(-50, 50)
    else:
        distraction_time = -1

    score = 0

    while agent.ticks < AGENT_TIME:
        output = agent.run(
            pendulum.x,
            pendulum.horizontal_velocity,
            math.cos(pendulum.angle),
            math.sin(pendulum.angle),
            pendulum.angular_velocity,
        )

        acceleration = Vec(output[0] * 30, 0)
        pendulum.apply_acceleration(acceleration)
        pendulum.update()

        if distraction_time == agent.ticks:
            pendulum.apply_acceleration(Vec(distraction_strength, 0))

        # Gain score while bob of the pendulum is above the x-axis close to x=0
        y = -math.sin(pendulum.angle)
        if y > 0:
            score += y * (1 - abs(pendulum.x))

        # Loose score close to edges
        score -= abs(pendulum.x) * 10

        if -0.01 <= pendulum.x <= 0.01:
            score += 30

        score -= abs(output[0]) * 5

        # Loose score for fast acceleration changes
        score -= abs(output[0] - last_acceleration) * 5
        last_acceleration = output[0]

        # Loose score for accelerating away from center after 5 seconds
        if agent.ticks > 300:
            if pendulum.x > 0 and output[0] > 0:
                score -= 3
            if pendulum.x < 0 and output[0] < 0:
                score -= 3

    return score


def main():
    rlm = ai.ReinforcementLearningModel(
        func=train,
        num_agents=50,
        inputs=["cart.x", "cart.vel", "bob.x", "bob.y", "bob.vel"],
        outputs=["acceleration"],
        hidden=[10, 10],
        hidden_activation="tanh",
        output_activation="tanh",
    )
    rlm.train()


if __name__ == "__main__":
    main()
