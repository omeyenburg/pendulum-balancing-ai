# Pendulum stabilisation AI

This Python program simulates a pendulum on a cart that can move within the range of -1 ≤ x ≤ 1. An AI, trained using reinforcement learning, is capable of stabilizing the pendulum in an inverted position.

## Dependencies

- [pygame](https://www.pygame.org)
- [numpy](https://numpy.org)
- [psutil](https://pypi.org/project/psutil/) (only used for training; optional)
- [pytest](https://docs.pytest.org) (required for testing, otherwise optional)

## File overview

- src/render.py - run to simulate the pendulum <b>without</b> the AI
- src/render_ai.py - run to simulate the pendulum <b>with</b> the AI
- src/train.py - run to train the AI
- src/ai - core of the AI with the classes "Agent" and "ReinforcementLearningModel"
- src/pendulum.py - pendulum simulation
- src/geometry.py - two dimensional vector class `Vec`
- tests/\* - run using `pytest`
- src/gen/\* - files containing the best weights and biases of individual generations

## Usage

### Running program

To simulate the pendulum without the AI: `python3 src/render.py`
To simulate the pendulum with the AI: `python3 src/render_ai.py`
To train the AI: `python3 src/train.py`

### Clean up

To remove all generations: `rm src/gen/*`
To remove unnecessary generations: `find src/gen -regex "gen.*[1-9]\.json" -delete`
