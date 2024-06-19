import concurrent.futures
import threading
import random
import numpy
import time
import json
import os


generation_directory: str = os.path.join(
    os.path.split(__file__)[0], "gen"
)  # directory name of generation backups
num_workers: int = 8  # number of workers, not agents
changes: int = 1  # number of changes to the weights and biases
change_strength: float = 0.05  # amount of the changes


def get_newest_generation(files: list[str]):
    """
    Returns the number of the newest generation, starting with 0, based on a list of file names.
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
        os.path.isdir(generation_directory) or os.makedirs(generation_directory)
        generation = data["generation"]
        file_name = os.path.join(
            generation_directory, "gen" + str(generation) + ".json"
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
    os.path.isdir(generation_directory) or os.makedirs(generation_directory)
    generation = get_newest_generation(os.listdir(generation_directory))
    if generation == -1:
        return None

    file_name = os.path.join(generation_directory, "gen" + str(generation) + ".json")
    with open(file_name, "r") as fp:
        data = json.load(fp)

    data["layers"] = numpy.array(data["layers"])

    return data


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
    def __init__(self, layers, weights, biases, activation):
        self.layers = layers
        self.weights = weights
        self.biases = biases
        self.activation = activation
        self.ticks = self.time = self.generation = 0

        # Construct arrays for values, biases and weights for each layer
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
        generation_dict = load()
        assert generation_dict

        layers = numpy.array(generation_dict["layers"])
        weights = numpy.array(generation_dict["weights"])
        biases = numpy.array(generation_dict["biases"])
        activation = ActivationFunction.__dict__[generation_dict["activation"]]

        agent = Agent(layers, weights, biases, activation)
        agent.ticks = generation_dict["ticks"]
        agent.time = generation_dict["time"]
        agent.generation = generation_dict["generation"]

        return agent

    def run(self, inputs: list[float]):
        self.ticks += 1
        self.values[0] = inputs

        for i in range(len(self.layers) - 1):
            self.values[i + 1] = self.activation(
                numpy.dot(self.values[i], self.weights[i]) + self.biases[i]
            )

        return self.values[-1]


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


def worker_process(func, layers, weights, biases, activation):
    agent = Agent(layers, weights, biases, activation)

    score = func(agent)
    ticks = agent.ticks

    return (score, ticks)


class ReinforcementLearningModel:
    def __init__(
        self,
        func,
        num_agents: int,  # number of agents
        inputs: list[str],  # input names
        outputs: list[str],  # output names
        hidden: list[int],  # number of neurons per hidden layer
        activation,  # method of ActicationFunction
    ):
        self.func = func
        self.num_agents = num_agents

        self.data = load() or {
            "generation": -1,
            "inputs": inputs,
            "outputs": outputs,
            "layers": numpy.array([len(inputs), *hidden, len(outputs)]),
            "activation": activation,
            "ticks": 0,
            "time": 0,
        }

        self.weights = []
        self.biases = []

        if "weights" in self.data:
            for _ in range(num_agents):
                self.weights.append(numpy.array(self.data["weights"]))
                self.biases.append(numpy.array(self.data["biases"]))
        else:
            num_weights = sum(self.data["layers"][1:] * self.data["layers"][:-1])
            num_biases = sum(self.data["layers"][1:])

            for _ in range(num_agents):
                self.weights.append(numpy.random.random(num_weights) * 2 - 1)
                self.biases.append(numpy.zeros(num_biases))

        self.activation = ActivationFunction.__dict__[self.data["activation"]]

    def train(self):
        last_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            while True:
                self.data["generation"] += 1

                workers = []
                for i in range(self.num_agents):
                    if i > 0:
                        for _ in range(changes):
                            j = random.randint(0, len(self.weights[i]) - 1)
                            self.weights[i][j] += (
                                random.random() - 0.5
                            ) * change_strength

                            j = random.randint(0, len(self.biases[i]) - 1)
                            self.biases[i][j] += (
                                random.random() - 0.5
                            ) * change_strength

                    worker = executor.submit(
                        worker_process,
                        self.func,
                        self.data["layers"],
                        self.weights[i],
                        self.biases[i],
                        self.activation,
                    )
                    workers.append(worker)

                results = sorted(
                    [(i, *worker.result()) for i, worker in enumerate(workers)],
                    key=lambda n: n[1],
                    reverse=True,
                )

                self.data["ticks"] += sum([results[i][2] for i in range(len(results))])
                print(
                    "Generation: "
                    + str(self.data["generation"])
                    + "; Best score: "
                    + str(results[0][1])
                )

                # TODO: save generation data to file
                generation_data = dict(self.data)
                generation_data["weights"] = self.weights[results[0][0]]
                generation_data["biases"] = self.biases[results[0][0]]
                save_generation_data(generation_data)

                # TODO: adjust weights and biases
                # new_agents = [results[1][0], results[2][0]]
                # new_agents.extend([results[0][0]] * (self.num_agents // 2))
                # new_agents.extend([results[1][0]] * (self.num_agents - len(new_agents)))
                new_agents = [results[0][0]] * self.num_agents
                self.weights = [self.weights[agent].copy() for agent in new_agents]
                self.biases = [self.biases[agent].copy() for agent in new_agents]

                t = time.time()
                self.data["time"] += t - last_time
                last_time = t
