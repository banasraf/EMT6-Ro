import numpy as np


def get_rand_population(population_size=120):
    MIN_DOSE = 0.25
    MAX_DOSE = 10
    return np.clip(np.random.random((population_size, 21)) * 10, MIN_DOSE, MAX_DOSE)


class ConvertRepresentation:
    NUMBER_OF_DAYS = 5

    def __init__(self, hour_steps, protocol_resolution):
        self.hour_steps = hour_steps
        self.protocol_resolution = protocol_resolution

    def calculate_number_of_steps(self):
        return self.NUMBER_OF_DAYS * 24 * self.hour_steps // self.protocol_resolution

    def convert_pairs_to_list(self, protocol):
        list_protocol = np.zeros(self.calculate_number_of_steps())
        for (timestep, value) in protocol:
            list_protocol[int(timestep // self.protocol_resolution)] = value
        return list_protocol

    def convert_list_to_pairs(self, protocol):
        indices = np.where(protocol != 0)[0]
        pair_protocol = [
            (index * self.protocol_resolution, protocol[index])
            for index in indices
        ]
        return pair_protocol
