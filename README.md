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
- `src/util.py` - two dimensional vector class `Vec`
- `tests/*` - run using `pytest`
- `src/gen/*` - files containing the best weights and biases of individual generations
- `showcase/*` - example videos

## Usage

### Running program

Simulate the pendulum without the AI:
`python3 src/render.py`

> Optional arguments:
>
> - `--angular-damping [float]` (default: 0.1)
> - `--horizontal-damping [float]` (default: 0.3)
> - `--gravity [float]` (default: 9.81)

> User inputs:
>
> - `r` - reset pendulum
> - `scroll` - accelerate the pendulum to the left or right

Simulate the pendulum with the AI:
`python3 src/render_ai.py`

> Optional arguments:
>
> - `--gen [int]` (default: -1)
> - `--angular-damping [float]` (default: 0.1)
> - `--horizontal-damping [float]` (default: 0.3)
> - `--gravity [float]` (default: 9.81)

> User inputs:
>
> - `r` - reset pendulum
> - `t` - toggle ai
> - `scroll` - accelerate the pendulum to the left or right

Train the AI:
`python3 src/train.py`

> Optional arguments:
>
> - `--time [float]` (default: 60)
> - `--random-start [bool]` (default: False)
> - `--distract [bool]` (default: False)

## Generations

The training process has saved the state of each generation in the `src/gen/` directory. Early generations, up until generation 12157, were trained with progressively increased gravity to help the AI gradually adapt to the final gravity value of 9.81 m/s². Similarly the damping values for horizontal and angular movement were reduced.

One of the best-performing generations is generation 12170. To run the simulation using this generation's parameters, execute the following command:
`python3 src/render_ai.py --gen 12170`

### Gravity and damping values

- Generation 1388

  > ```
  > angular_damping = 0.7
  > horizontal_damping = 0.48
  > gravity = 0.65
  > ```

- Generations 2157 to 12157:

  > ```
  > angular_damping = 0.64624 + (0.1 - 0.64624) * (generation - 2157) / 10000
  > horizontal_damping = 0.463872 + (0.3 - 0.463872) * (generation - 2157) / 10000
  > gravity = 1.470736 + (9.81 - 1.470736) * (generation - 2157) / 10000
  > ```

- Generations past 12157:
  > ```
  > angular_damping = 0.1
  > horizontal_damping = 0.3
  > gravity = 9.81
  > ```
