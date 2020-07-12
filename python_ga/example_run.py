import numpy as np
import sys

from utils import ConvertRepresentation, get_rand_population, ModelWrapper, read_config
from genetic_algorithm import new_genetic_algorithm

from emt6ro.simulation import load_state, load_parameters
from emt6ro.ga_model import EMT6RoModel


class MockPredictionModel:
    def predict(self, x):
        return np.random.rand(len(x))


def main(config_path: str):
    config_path = config_path
    config = read_config(config_path)
    config['config_path'] = config_path


    num_gpus = config['num_gpus']
    num_protocols = config['num_protocols']
    num_tests = config['num_tests']
    params = load_parameters("data/default-parameters.json")
    tumors = [load_state("data/tumor-lib/tumor-{}.txt".format(i), params) for i in range(1, 11)]
    model_emt = EMT6RoModel(params, tumors, num_protocols, num_tests, num_gpus)


    hour_steps = config['hour_steps']
    protocol_resolution = config['protocol_resolution']
    #model = MockPredictionModel()
    converter = ConvertRepresentation(hour_steps=hour_steps, protocol_resolution=protocol_resolution)


    pair_protocols = [
        # Initial
        [(hour_steps * 12, 1.25), (hour_steps * 36, 3.0)],
        [(hour_steps * 12, 1.25), (hour_steps * 24, 1.5), (hour_steps * 36, 1.5)],

        # Sample
        [(hour_steps * 12, 1.25), (hour_steps * 36, 3.0)],
        [(hour_steps * 8, 1.25), (hour_steps * 30, 1.25), (hour_steps * 48, 1.75)],
        [(hour_steps * 24, 4.25)],
        [(hour_steps * 12, 2), (hour_steps * 36, 2.25)],
        [(hour_steps * 6, 1), (hour_steps * 18, 1), (hour_steps * 30, 2.25)],
        [(hour_steps * 36, 2), (hour_steps * 54, 2.25)],
        [(hour_steps * 6, 1), (hour_steps * 18, 1), (hour_steps * 30, 8)],
        [(hour_steps * 36, 4), (hour_steps * 54, 6)],
    ]


    list_protocols = [
        converter.convert_pairs_to_list(protocol=protocol)
        for protocol in pair_protocols
    ]

    new_genetic_algorithm(population=list_protocols, model=model_emt, config=config, converter=converter)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        raise Exception(f'Please specify path to yaml config file.\n\nFor example:\n'
                        f'python python_ga/example_run.py '
                        f'python_ga/experiment_config/base__tournament__dose_value_time_value.yaml')

    config_path = sys.argv[1]
    print(f'Using config file: {config_path}')

    main(config_path=config_path)
