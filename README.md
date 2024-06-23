# Pendulum Balancing AI

This Python program simulates a pendulum on a cart that can move within the range of -1 ≤ x ≤ 1. An AI, trained using reinforcement learning, balances the pendulum upside down.

AI inputs:

- horizontal position of the cart
- horizontal velocity of the cart
- horizontal position of the bob
- vertical position of the bob
- angular velocity of the bob

AI output:

- horizontal acceleration

## Dependencies

- [`pygame`](https://www.pygame.org): used for rendering
- [`numpy`](https://numpy.org): used for ai training and execution
- [`psutil`](https://pypi.org/project/psutil/): only used for training; optional
- [`pytest`](https://docs.pytest.org): required for testing, otherwise optional

## File overview

- `src/render.py` - run to simulate the pendulum <b>without</b> the AI
- `src/render_ai.py` - run to simulate the pendulum <b>with</b> the AI
- `src/train.py` - run to train the AI
- `src/ai` - core of the AI with the classes "Agent" and "ReinforcementLearningModel"
- `src/pendulum.py` - pendulum simulation
- `src/geometry.py` - two dimensional vector class `Vec`
- `tests/*` - run using `pytest`
- `src/gen/*` - files containing the best weights and biases of individual generations
- `showcase/*` - example videos

## Usage

### Running program

Simulate the pendulum without the AI:
`python3 src/render.py`

Simulate the pendulum with the AI:
`python3 src/render_ai.py`

> Optional arguments:
>
> - `--gen [int]`

Train the AI:
`python3 src/train.py`

> Optional arguments:
>
> - `--time [float]`
> - `--random-start [bool]`
> - `--distract [bool]`

## Generations

The training process has saved the state of each generation in the src/gen/ directory. Early generations, up until generation 12157, were trained with progressively increased gravity to help the AI gradually adapt to the final gravity value of 9.81 m/s².

One of the best-performing generations is generation 12170. To run the simulation using this generation's parameters, execute the following command:
`python3 src/render_ai.py --gen 12170`
