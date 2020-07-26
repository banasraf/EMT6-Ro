import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import neptune


from datetime import datetime
from itertools import cycle

from utils import save_output, resolve_saving_path
from emt6ro.simulation import load_state, load_parameters
from emt6ro.ga_model import EMT6RoModel



hour_steps = 600

params = load_parameters("data/default-parameters.json")
protocols = [
    [[(hour_steps * 12, 1.25), (hour_steps * 36, 3.0)]],
    [[(hour_steps * 12, 1.25), (hour_steps * 24, 1.5), (hour_steps * 36, 1.5)]],
    [[(hour_steps * 12, 1.25), (hour_steps * 36, 3.0)]],
    [[(hour_steps * 8, 1.25), (hour_steps * 30, 1.25), (hour_steps * 48, 1.75)]],
    [[(hour_steps * 24, 4.25)]],
    [[(hour_steps * 12, 2), (hour_steps * 36, 2.25)]],
    [[(hour_steps * 6, 1), (hour_steps * 18, 1), (hour_steps * 30, 2.25)]],
    [[(hour_steps * 36, 2), (hour_steps * 54, 2.25)]],
    [[(hour_steps * 6, 1), (hour_steps * 18, 1), (hour_steps * 30, 8)]],
    [[(hour_steps * 36, 4), (hour_steps * 54, 6)]],
]


num_gpus = 4
num_protocols = 1
params = load_parameters("data/default-parameters.json")
tumors = [load_state("data/tumor-lib/tumor-{}.txt".format(i), params) for i in range(1, 11)]

for p in protocols:
    test_name = "Test_convergence_protocol_protocol_" + str(p)
    print("Protocol: " + str(p))
    neptune.init('TensorCell/cancertreatment')
    neptune.create_experiment(name=test_name, params={'protocol': p, 'num_gpus': num_gpus, 'num_protocols': num_protocols})
    neptune.append_tag(test_name)
    for i in range(10):
        num_tests = 2**i
        print("num_tests: " + str(num_tests))
        model = EMT6RoModel(params, tumors, num_protocols, num_tests, num_gpus)
        date1 = datetime.now()
        pop_fitness = model.predict(p)
        neptune.log_metric("Fitness", pop_fitness)
        date2 = date1
        date1 = datetime.now()
        print("Time: " + str(date1 - date2))
        print(pop_fitness)
    neptune.stop()
