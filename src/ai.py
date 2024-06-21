import concurrent.futures
import threading
import random
import numpy
import time
import json
import os


GENERATION_DIRECTORY: str = os.path.join(os.path.split(__file__)[0], "gen")
NUM_WORKERS: int = 8  # Number of workers, not agents
WEIGHT_CHANGE_STRENGTH: float = 0.003
BIAS_CHANGE_STRENGTH: float = 0.001
PRINT_RESULTS: bool = True
SESSION_GENERATIONS: int = 500


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


def load():
    """
    Returns the data of the most recent generation save file.
    Returns None if no save file is found.
    """
    os.path.isdir(GENERATION_DIRECTORY) or os.makedirs(GENERATION_DIRECTORY)
    generation = get_newest_generation(os.listdir(GENERATION_DIRECTORY))
    if generation == -1:
        return None

    file_name = os.path.join(GENERATION_DIRECTORY, "gen" + str(generation) + ".json")
    with open(file_name, "r") as fp:
        data = json.load(fp)

    data["layers"] = numpy.array(data["layers"])

    return data


def seconds_to_str(t):
    seconds = round(t) % 60
    minutes = round(t / 60) % 60
    hours = round(t / 3600) % 24
    days = round(t / 3600 / 24)

    def join_plural(number, unit):
        if number == 1:
            return str(number) + " " + unit
        return str(number) + " " + unit + "s"

    if days:
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


def activation_function(name: str):
    """
    Returns a activation function from a name.
    """
    return ActivationFunction.__dict__[name]


class ActivationFunction:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + numpy.exp(-z))

    @staticmethod
    def tanh(z):
        return numpy.tanh(z)

    @staticmethod
    def relu(z):
        return numpy.maximum(0, z)


class Agent:
    def __init__(self, layers, weights, biases, hidden_activation, output_activation):
        self.layers = layers
        self.weights = weights
        self.biases = biases
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
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
    def load():
        """
        Load the latest generation and return the agent.
        """
        generation = load()
        assert generation

        layers = numpy.array(generation["layers"])
        weights = numpy.array(generation["weights"])
        biases = numpy.array(generation["biases"])
        hidden_activation = activation_function(generation["hidden_activation"])
        output_activation = activation_function(generation["output_activation"])

        agent = Agent(layers, weights, biases, hidden_activation, output_activation)
        agent.ticks = generation["ticks"]
        agent.time = generation["time"]
        agent.generation = generation["generation"]
        agent.inputs = generation["inputs"]
        agent.outputs = generation["outputs"]

        return agent

    def run(self, *inputs: float):
        """
        Run a single iteration through the network.
        """
        self.ticks += 1
        self.values[0][:] = inputs[:]

        for i in range(len(self.layers) - 1):
            self.values[i + 1] = self.hidden_activation(
                numpy.dot(self.values[i], self.weights[i]) + self.biases[i]
            )

        # self.values[-1] = self.output_activation(
        #     numpy.dot(self.values[-2], self.weights[-1]) + self.biases[-1]
        # )

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
        self.data = load()

        if self.data:
            weights = numpy.array(self.data["weights"])
            biases = numpy.array(self.data["biases"])
            self.weights = weights[numpy.newaxis, :].repeat(self.num_agents, 0)
            self.biases = biases[numpy.newaxis, :].repeat(self.num_agents, 0)
        else:
            self._default_data(*args)

        self.hidden_activation = activation_function(self.data["hidden_activation"])
        self.output_activation = activation_function(self.data["output_activation"])

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
        last_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            for _ in range(SESSION_GENERATIONS):
                self._iterate(executor)
                t = time.time()
                self.data["time"] += t - last_time
                last_time = t

    def _iterate(self, executor):
        self.data["generation"] += 1
        self._adjust_weights()

        workers = [
            executor.submit(
                self._worker_process,
                self.func,
                self.data["layers"],
                self.weights[i],
                self.biases[i],
                self.hidden_activation,
                self.output_activation,
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
        new_agents = [results[0][0]] * self.num_agents
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

            string = f"Generation: {gen}; Best Score: {best_score}; Total Time: {tot_time}; Gen Time: {gen_time}; Min Time: {min_time}; Max Time: {max_time}"
            print(string)

    @staticmethod
    def _worker_process(func, *args):
        agent = Agent(*args)

        score = func(agent)
        ticks = agent.ticks

        return score, ticks
