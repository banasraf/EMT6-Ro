from .. import simulation as sim
import numpy as np


class EMT6RoModel:
    """
    Model calculating fitness of irradiation protocols.
    """
    def __init__(self, params, tumors, num_protocols, num_tests, num_gpus=1):
        """
        Parameters
        ----------
        params : simulation.Parameters
            Parameters of the simulation.
        tumors : list of simulation.TumorState
            List of tumors to be simulated.
        num_protocols : int
            Number of protocols to test.
        num_tests : int
            Number of simulations for each protocols, tumor and gpu. So if we want
            to test a protocol 100 times and we have 4 gpus and 5 test tumors, 
            num_tests should be equal to 5.
        num_gpus : int
            Number of available gpus.
        """
        self.num_steps = 144000
        self.protocol_resolution = 300
        self.params = params
        self.tumors = tumors
        self.num_protocols = num_protocols
        self.num_tests = num_tests
        self.num_gpus = num_gpus
        self.experiments = [sim.Experiment(params, tumors, self.num_tests, self.num_protocols, g, self.num_steps,
                                           self.protocol_resolution) for g in range(self.num_gpus)]

    def predict(self, protocols):
        """
        Returns fitness of each protocol.

        Parameters
        ----------
        protocols : list of lists of pairs (simulation step, irradiation dose)
        """
        assert len(protocols) == self.num_protocols
        for experiment in self.experiments:
            experiment.add_irradiations(protocols)
            experiment.run(self.num_steps)
        results = [experiment.get_results().reshape((1, self.num_protocols, -1)) for experiment in self.experiments]
        results = np.concatenate(results)
        results = results.mean(0).mean(1)
        fitness = 1500 - results
        for experiment in self.experiments:
            experiment.reset()
        return fitness
