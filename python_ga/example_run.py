import numpy as np

from utils import get_rand_population
from utils import ConvertRepresentation
from genetic_algorithm import new_genetic_algorithm

# from emt6ro.simulation import load_state, load_parameters
# from emt6ro.ga_model import EMT6RoModel


class MockPredictionModel:
    def predict(self, x):
        return np.asarray([[np.random.random()] for i in range(len(x))])


# params = load_parameters("data/default-parameters.json")
# tumors = [load_state("data/tumor-lib/tumor-{}.txt".format(i), params) for i in range(1, 11)]

# num_gpus = 2
# num_protocols = 2
# num_tests = 20
# model = EMT6RoModel(params, tumors, num_protocols, num_tests, num_gpus)


hour_steps = 600
protocol_resolution = 300
model = MockPredictionModel()
converter = ConvertRepresentation(hour_steps=hour_steps, protocol_resolution=protocol_resolution)


sample_protocols = [
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


list_protocols = np.asarray([
    converter.convert_pairs_to_list(protocol=protocol)
    for protocol in sample_protocols
])


pair_protocols = np.asarray([
    converter.convert_list_to_pairs(protocol=protocol)
    for protocol in list_protocols
])


config = {
    'cross_type': 'normalized_crossover',
    # 'selection': {'type': 'simple_selection'},
    'selection': {'type': 'tournament_selection'},
    # 'selection': {'type': 'roulette_selection'},

    'mutations': {
        'mut_swap': {'mut_prob': 0.03},
        'mut_dose_value': {'mut_prob': 0.05, 'min_value': 0.25, 'max_value': 10, 'step_value': 0.25},
        'mut_time_value': {'mut_prob': 0.05},
        'mutate_merge': {'mut_prob': 0.05},
        'mutate_split': {'mut_prob': 0.05},
    },
    'step_value': 0.25,
    'max_value': 10,
    'select_n': 0.5,
    'max_iter': 100,
    'stop_fitness': -0.5,
    'normalization': {'min_value': 0.25, 'max_value': 10, 'step_value': 0.25},
}


# population = get_rand_population(8)
new_genetic_algorithm(list_protocols, model, config)
