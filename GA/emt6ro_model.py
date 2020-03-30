import py_emt6ro as sim
import numpy as np


class EMT6RoModel:
    def __init__(self, params, tumors, num_protocols, num_tests, num_gpus=1):
        self.num_steps = 144000
        self.protocol_resolution = 300
        self.params = params
        self.tumors = tumors
        self.num_protocols = num_protocols
        self.num_tests = num_tests
        self.num_gpus = num_gpus
        self.experiments = [sim.Experiment(params, tumors, self.num_tests, self.num_protocols, self.num_steps,
                                      self.protocol_resolution, g) for g in range(self.num_gpus)]

    def predict(self, protocols):
        assert len(protocols) == self.num_protocols
        for experiment in self.experiments:
            experiment.run(protocols)
        results = [experiment.results().reshape((1, self.num_protocols, -1)) for experiment in self.experiments]
        results = np.concatenate(results)
        results = results.mean(0).mean(1)
        fitness = 1000. / results
        return fitness