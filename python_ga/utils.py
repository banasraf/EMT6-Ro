import numpy as np


def get_rand_population(population_size=120):
    return np.clip(np.random.random((population_size, 21)) * 10, 0.25, 10)


class ConvertRepresentation:
    def __init__(self, hour_steps, dosage_frequency):
        self.hour_steps = hour_steps
        self.dosage_frequency = dosage_frequency

    def convert_pairs_to_list(self):
        pass

    def convert_list_to_pairs(self):
        pass
