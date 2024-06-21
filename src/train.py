from pendulum import Pendulum
from geometry import Vec
import random
import math
import ai


AGENT_TIME = 3600 * 3


def train(agent: ai.Agent):
    pendulum = Pendulum()
    # pendulum.x = random.uniform(-0.5, 0.5)
    # pendulum.horizontal_velocity = random.uniform(-1, 1)
    # pendulum.angle = random.uniform(-math.pi, math.pi)
    # pendulum.angular_velocity = random.uniform(-0.5, 0.5)
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

        y = -math.sin(pendulum.angle)
        if y > 0:
            score += y * (1 - abs(pendulum.x))
        score -= abs(pendulum.x) ** 3 * 5

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
