import numpy as np
import os
import yaml


def get_random_protocol(
    min_dose: float = 0.25,
    max_dose: float = 10.0,
    step_value: float = 0.25,
    max_dose_value: float = 2.5,
    time_steps: np.ndarray = None,
):
    protocol_dose = 0
    max_protocol_value = max_dose_value
    protocol_pair = []
    while protocol_dose < max_dose:
        if max_dose - protocol_dose < max_dose_value:
            max_protocol_value = max_dose - protocol_dose

        upper_value_boundary = int(round((max_protocol_value - min_dose) / step_value)) + 1
        new_dose_value = np.random.randint(0, upper_value_boundary) * step_value + min_dose
        protocol_pair.append(
            (np.random.choice(time_steps), new_dose_value)
        )
        protocol_dose += new_dose_value
    return protocol_pair


def get_rand_population(
        min_dose: float = 0.25,
        max_dose: float = 10.0,
        step_value: float = 0.25,
        num_protocols: int = 10,
        max_dose_value: float = 2.5,
        hour_steps: int = 600,
        protocol_resolution: int = 300,
        available_hours: int = 24 * 5,
):
    time_steps = np.arange(0, available_hours * hour_steps, protocol_resolution)
    protocol_pairs = []
    for protocol in range(num_protocols):
        protocol_pair = get_random_protocol(
            min_dose=min_dose,
            max_dose=max_dose,
            step_value=step_value,
            max_dose_value=max_dose_value,
            time_steps=time_steps,
        )
        protocol_pairs.append(protocol_pair)

    return protocol_pairs


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

    def convert_population_lists_to_pairs(self, protocols):
        return [
            self.convert_list_to_pairs(protocol=protocol)
            for protocol in protocols
        ]


class ModelWrapper:
    def __init__(self, model, converter):
        self.model = model
        self.converter = converter

    def predict(self, population):
        converted = [self.converter.convert_list_to_pairs(p) for p in population]
        return self.model.predict(converted)


def save_output(file: any, file_name: str, extension: str = 'csv', config=None):
    saving_directory = resolve_saving_path(config=config)
    saving_path = f'{saving_directory}/{file_name}.{extension}'

    if extension == 'csv':
        file.to_csv(saving_path, index=False)
    else:
        with open(saving_path, 'w') as f:
            f.write(str(file))


def resolve_saving_path(path: str = 'experiment_results', config=None):
    if config:
        sub_path = f'{os.path.basename(config["config_path"]).split(".")[0]}_{config["experiment_time"]}'
        path = os.path.join(path, sub_path)
    saving_directory = os.path.join(os.getcwd(), f'python_ga/{path}')

    if not os.path.exists(saving_directory):
        os.makedirs(saving_directory)

    return saving_directory


def read_config(config_path: str):
    with open(config_path, 'r') as file:
        config = yaml.load(file, yaml.FullLoader)
    return config
