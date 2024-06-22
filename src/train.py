from pendulum import Pendulum
from geometry import Vec
import random
import math
import ai


AGENT_TIME = 3600
RANDOM_START = True
DISTRACTIONS = True


def train(agent: ai.Agent):
    pendulum = Pendulum()
    random.seed(agent.generation)

    last_acceleration = 0

    if RANDOM_START and agent.generation % 2 == 1:
        pendulum.x = random.uniform(-0.7, 0.7)
        pendulum.angle = math.pi / 2 + random.uniform(-math.pi, math.pi)
        pendulum.angular_velocity = random.uniform(-3, 3)
        pendulum.horizontal_velocity = random.uniform(-3, 3)

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
        score -= abs(pendulum.x) ** 3 * 5

        # Loose score for fast acceleration changes
        score -= abs(output[0] - last_acceleration) * 0.1
        last_acceleration = output[0]

        # Loose score for accelerating away from center after 10 seconds
        if agent.ticks > 1200:
            if pendulum.x > 0 and output[0] > 0:
                score -= 1
            if pendulum.x < 0 and output[0] < 0:
                score -= 1

        # score -= math.sin(pendulum.angle) * 0.1
        # score -= abs(pendulum.x) * 10
        # score -= abs(pendulum.angular_velocity) * 0.1
        # score -= abs(pendulum.horizontal_velocity) * 0.1

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
