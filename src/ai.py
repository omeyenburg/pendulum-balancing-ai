from __future__ import annotations
import concurrent.futures
import threading
import random
import numpy
import util
import time
import json
import sys
import os

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ModuleNotFoundError:
    PSUTIL_AVAILABLE = False


GENERATION_DIRECTORY: str = util.abspath("gen")
NUM_WORKERS: int = 8  # Number of workers/processes, not agents
WEIGHT_CHANGE_STRENGTH: float = 0.01
BIAS_CHANGE_STRENGTH: float = 0.005
PRINT_RESULTS: bool = True
SESSION_GENERATIONS: int = 2000
TRANSFER_BEST_PERCENT: float = 0.5


def set_niceness(niceness):
    """
    Set the priority of the current process. Lower values mean higher priority.
    -20 <= niceness <= 20
    """
    if not PSUTIL_AVAILABLE:
        return

    try:
        p = psutil.Process(os.getpid())
        p.nice(niceness)
    except psutil.AccessDenied:
        # Permission denied to set niceness.
        # Fixed by running as root or adjusting system permissions.
        return


def get_newest_generation(files: list[str]):
    """
    Returns the number of the newest generation, starting with 0.
    Returns -1 if no file name is valid.
    """
    newest = -1

    for name in files:
        try:
            generation = int(name[3:-5])  # Extract number from "gen0.json"
        except ValueError:
            continue

        newest = max(newest, generation)

    return newest


def save_generation_data(data: dict):
    """
    Save data to file.
    """

    def save_thread(data: dict):
        os.path.isdir(GENERATION_DIRECTORY) or os.makedirs(GENERATION_DIRECTORY)
        generation = data["generation"]
        file_name = os.path.join(
            GENERATION_DIRECTORY, "gen" + str(generation) + ".json"
        )

        data["layers"] = data["layers"].tolist()
        data["weights"] = data["weights"].tolist()
        data["biases"] = data["biases"].tolist()

        with open(file_name, "w") as fp:
            json.dump(data, fp)

    thread = threading.Thread(target=save_thread, args=(data,), daemon=False)
    thread.start()


def _load_generation(generation=-1):
    """
    Returns the data of the specified generation save file.
    Returns the most recent file if generation is set to -1.
    Returns None if no save file is found.
    """
    if generation == -1:
        os.path.isdir(GENERATION_DIRECTORY) or os.makedirs(GENERATION_DIRECTORY)
        generation = get_newest_generation(os.listdir(GENERATION_DIRECTORY))
        if generation == -1:
            return None

    file_name = os.path.join(GENERATION_DIRECTORY, "gen" + str(generation) + ".json")
    try:
        with open(file_name, "r") as fp:
            data = json.load(fp)
    except FileNotFoundError:
        return None

    data["layers"] = numpy.array(data["layers"])

    return data


def seconds_to_str(t):
    seconds = int(t) % 60
    minutes = int(t / 60) % 60
    hours = int(t / 3600) % 24
    days = int(t / 3600 / 24) % 356
    years = int(t / 3600 / 24 / 356)

    def join_plural(number, unit):
        if number == 1:
            return str(number) + " " + unit
        return str(number) + " " + unit + "s"

    if years:
        string = join_plural(years, "year")
        if days:
            string += ", " + join_plural(days, "day")
    elif days:
        string = join_plural(days, "day")
        if hours:
            string += ", " + join_plural(hours, "hour")
    elif hours:
        string = join_plural(hours, "hour")
        if minutes:
            string += ", " + join_plural(minutes, "minute")
    elif minutes:
        string = join_plural(minutes, "minute")
        if seconds:
            string += ", " + join_plural(seconds, "second")
    else:
        string = join_plural(seconds, "second")

    return string


def _worker_process(func, *args):
    set_niceness(-10)
    agent = Agent(*args)

    score = func(agent)
    ticks = agent.ticks

    return score, ticks


def _activation_function(name: str):
    """
    Returns a activation function from a name.
    """
    return ActivationFunction.__dict__[name]


class ActivationFunction:
    def sigmoid(z):
        return 1 / (1 + numpy.exp(-z))

    def tanh(z):
        return numpy.tanh(z)

    def relu(z):
        return numpy.maximum(0, z)


class Agent:
    def __init__(
        self, layers, weights, biases, hidden_activation, output_activation, generation
    ):
        self.layers = layers
        self.weights = weights
        self.biases = biases
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.generation = generation
        self.ticks = 0
        self._initialize_arrays(weights, biases)

    def _initialize_arrays(self, weights, biases):
        """
        Construct arrays for values, biases and weights for each layer
        """
        self.values = []
        self.biases = []
        self.weights = []

        bias_index = weight_index = 0
        for i, layer in enumerate(self.layers):
            self.values.append(numpy.zeros(layer))

            if i + 1 == len(self.layers):
                continue
            next_layer = self.layers[i + 1]

            layer_biases = biases[bias_index : bias_index + next_layer]
            self.biases.append(layer_biases)
            bias_index += next_layer

            layer_weights = weights[
                weight_index : weight_index + layer * next_layer
            ].reshape((layer, next_layer))
            self.weights.append(layer_weights)
            weight_index += layer * next_layer

    @staticmethod
    def load(generation=-1) -> "Agent":
        """
        Load the specified generation and return the agent.
        Set generation to -1 to load the latest generation.
        """
        generation = _load_generation(generation)
        assert generation

        agent = Agent(
            layers=numpy.array(generation["layers"]),
            weights=numpy.array(generation["weights"]),
            biases=numpy.array(generation["biases"]),
            hidden_activation=_activation_function(generation["hidden_activation"]),
            output_activation=_activation_function(generation["output_activation"]),
            generation=generation["generation"],
        )

        agent.ticks = generation["ticks"]
        agent.time = generation["time"]
        agent.inputs = generation["inputs"]
        agent.outputs = generation["outputs"]

        return agent

    def run(self, *inputs: float):
        """
        Run a single iteration through the network.
        """
        self.ticks += 1
        self.values[0][:] = inputs[:]

        for i in range(len(self.layers) - 2):
            self.values[i + 1] = self.hidden_activation(
                numpy.dot(self.values[i], self.weights[i]) + self.biases[i]
            )

        self.values[-1] = self.output_activation(
            numpy.dot(self.values[-2], self.weights[-1]) + self.biases[-1]
        )

        return self.values[-1]


class ReinforcementLearningModel:
    def __init__(
        self,
        func,
        num_agents: int = 25,  # Number of agents
        inputs: list[str] = [],  # Input names
        outputs: list[str] = [],  # Output names
        hidden: list[int] = [],  # Number of neurons per hidden layer
        hidden_activation: str = "relu",  # Activation function for hidden layers
        output_activation: str = "sigmoid",  # Activation function for output layer
    ):
        self.func = func
        self.num_agents = num_agents
        self._load_data(inputs, outputs, hidden, hidden_activation, output_activation)

    def _load_data(self, *args):
        self.weights = []
        self.biases = []
        self.data = _load_generation()

        if self.data:
            weights = numpy.array(self.data["weights"])
            biases = numpy.array(self.data["biases"])
            self.weights = weights[numpy.newaxis, :].repeat(self.num_agents, 0)
            self.biases = biases[numpy.newaxis, :].repeat(self.num_agents, 0)
        else:
            self._default_data(*args)

        self.hidden_activation = _activation_function(self.data["hidden_activation"])
        self.output_activation = _activation_function(self.data["output_activation"])

    def _default_data(
        self, inputs, outputs, hidden, hidden_activation, output_activation
    ):
        self.data = {
            "generation": -1,
            "inputs": inputs,
            "outputs": outputs,
            "layers": numpy.array([len(inputs), *hidden, len(outputs)]),
            "hidden_activation": hidden_activation,
            "output_activation": output_activation,
            "ticks": 0,
            "time": 0,
        }

        num_weights = sum(self.data["layers"][1:] * self.data["layers"][:-1])
        num_biases = sum(self.data["layers"][1:])

        self.weights = numpy.random.uniform(-1, 1, (self.num_agents, num_weights))
        self.biases = numpy.zeros((self.num_agents, num_biases)) * 2 - 1

    def train(self):
        set_niceness(-10)
        last_time = time.time()

        with concurrent.futures.ProcessPoolExecutor(NUM_WORKERS) as executor:
            for _ in range(SESSION_GENERATIONS):
                self.data["generation"] += 1
                self._iterate(executor)

                t = time.time()
                self.data["time"] += t - last_time
                last_time = t

    def _iterate(self, executor):
        self._adjust_weights()
        self._adjust_weights()
        self._adjust_weights()

        workers = [
            executor.submit(
                _worker_process,
                self.func,
                self.data["layers"],
                self.weights[i],
                self.biases[i],
                self.hidden_activation,
                self.output_activation,
                self.data["generation"],
            )
            for i in range(self.num_agents)
        ]

        results = sorted(
            [(i, *worker.result()) for i, worker in enumerate(workers)],
            key=lambda n: n[1],
            reverse=True,
        )

        self.data["ticks"] += sum([results[i][2] for i in range(len(results))])

        # Save generation data to file
        generation_data = dict(self.data)
        generation_data["weights"] = self.weights[results[0][0]]
        generation_data["biases"] = self.biases[results[0][0]]
        save_generation_data(generation_data)

        # Use best weights and biases
        new_agents = [results[0][0]] * int(self.num_agents * TRANSFER_BEST_PERCENT)
        new_agents.extend([results[1][0]] * ((self.num_agents - len(new_agents)) // 3))
        new_agents.extend(
            random.choices(range(self.num_agents), k=self.num_agents - len(new_agents))
        )

        for i, agent in enumerate(new_agents):
            self.weights[i] = self.weights[agent].copy()
            self.biases[i] = self.biases[agent].copy()

        # Print results
        self._print_results(results)

    def _adjust_weights(self):
        """
        Adjusts the weights randomly, except for the first agent.
        """
        size = self.num_agents
        strength = WEIGHT_CHANGE_STRENGTH
        weight_indices = numpy.random.randint(0, self.weights.shape[1], size=size - 1)
        weight_changes = numpy.random.uniform(-strength, strength, size=size - 1)
        self.weights[numpy.arange(1, size), weight_indices] += weight_changes

        strength = BIAS_CHANGE_STRENGTH
        bias_indices = numpy.random.randint(0, self.biases.shape[1], size=size - 1)
        bias_changes = numpy.random.uniform(-strength, strength, size=size - 1)
        self.biases[numpy.arange(1, size), bias_indices] += bias_changes

    def _print_results(self, results):
        if PRINT_RESULTS:
            gen = self.data["generation"]
            best_score = results[0][1]
            tot_time = round(self.data["ticks"] / 60)
            time = list(map(lambda r: r[2], results))
            gen_time = sum(time)
            min_time = min(time)
            max_time = max(time)

            string = (
                f"Generation: {gen}; Best Score: {best_score}; Total Time: {tot_time}"
            )

            if not min_time == max_time == gen_time // self.num_agents:
                string += f"; Gen Time: {gen_time}; Min Time: {min_time}; Max Time: {max_time}"
            print(string)
