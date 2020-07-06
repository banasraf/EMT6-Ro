import numpy as np
import os


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
        non_zero_indices = [i for i, value in enumerate(protocol) if value != 0]
        pair_protocol = [
            tuple((index * self.protocol_resolution, protocol[index]))
            for index in non_zero_indices
        ]
        return pair_protocol


class ModelWrapper:
    def __init__(self, model, converter):
        self.model = model
        self.converter = converter

    def predict(self, population):
        converted = [self.converter.convert_list_to_pairs(p) for p in population]
        return self.model.predict(converted)


def save_output(file: any, file_name: str, path: str = 'results', extension: str = 'csv'):
    saving_directory = os.path.join(os.getcwd(), f'python_ga/{path}')
    saving_path = f'{saving_directory}/{file_name}.{extension}'

    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)

    if extension == 'csv':
        file.to_csv(saving_path, index=False)
    else:
        with open(saving_path, 'w') as f:
            f.write(str(file))
