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


model = MockPredictionModel()
converter = ConvertRepresentation()
# model = EMT6RoModel(params, tumors, num_protocols, num_tests, num_gpus)
num_gpus = 2
num_protocols = 2
num_tests = 20


hour_steps = 600
prot1 = [(hour_steps * 12, 1.25), (hour_steps * 36, 3.0)]
prot2 = [(hour_steps * 12, 1.25), (hour_steps * 24, 1.5), (hour_steps * 36, 1.5)]


prots = [prot1, prot2]





config = {
    'cross_type': 'proximity_base_crossover',
    'selection': {'type': 'simple_selection'},
    'mutations': {
        'mut_swap': {'mut_prob': 0.03},
        'mut_random': {'mut_prob': 0.09, 'max_value': 118},
        'mutate_merge': None,
    },
    'select_n': 0.25,
    'max_iter': 100,
    'stop_fitness': -0.5,
}


population = get_rand_population(8)
new_genetic_algorithm(population, model, config)
