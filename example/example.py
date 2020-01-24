import py_emt6ro as sim
import numpy as np
params = sim.load_parameters("../data/default-parameters.json")
state = sim.load_state("../data/test_tumor.txt", params)
state2 = sim.load_state("../data/test_tumor.txt", params)
sim.something([state, state2, state])
experiment = sim.Experiment(params, [state, state2],
                            500,  # tests number per protocol and tumor
                            1,  # protocols number
                            144000,  # simulation steps number
                            300,  # protocol resolution
                            0)  # GPU id
experiment.run([[(0, 5.), (42 * 600, 2.5), (66 * 600, 2.5)]])
res = experiment.results()
print(res)
print(np.mean(res))
print(np.var(res))