import emt6ro.simulation as sim
import numpy as np
params = sim.load_parameters("../data/default-parameters.json")
state = sim.load_state("../data/test_tumor.txt", params)
state2 = sim.load_state("../data/test_tumor.txt", params)
protocols = [[(0, 5.), (42 * 600, 2.5), (66 * 600, 2.5)], [(0, 2.5), (42 * 600, 1), (66 * 600, 1.5)]]
experiment = sim.Experiment(params, [state, state2], 200, len(protocols))  

experiment.run(protocols)
res = experiment.get_results()
print("Protocol 1 result")
print(np.mean(res[0]))
print("Protocol 2 result")
print(np.mean(res[1]))

