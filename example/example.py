import emt6ro.simulation as sim
import numpy as np
params = sim.load_parameters("../data/default-parameters.json")
state = sim.load_state("../data/test_tumor.txt", params)
state2 = sim.load_state("../data/test_tumor.txt", params)
protocols = [[(44700, 1.0), (54000, 1.0), (54900, 2.0), (56100, 1.75), (56400, 1.75), (56700, 1.5), (57000, 1.0)], [(0, 2.5), (42 * 600, 1), (66 * 600, 1.5)]]
experiment = sim.Experiment(params, [state, state2], 250, len(protocols))

experiment.add_irradiations(protocols)
experiment.run(144000)
res = experiment.get_results()
print("Protocol 1 result")
print(np.mean(res[0]))
print("Protocol 2 result")
print(np.mean(res[1]))

states = experiment.state()
p1_states = states[0:500]
p2_states = states[500:1000]

p1_states = filter(lambda state: (state.occupancy() == 1).any(), p1_states)
p2_states = filter(lambda state: (state.occupancy() == 1).any(), p2_states)

mR = sum(map(lambda state: np.mean(state.irradiation()[state.occupancy() == 1]), p1_states)) / 500
print("Protocol 1 mean irradiation: ", mR)

mR = sum(map(lambda state: np.mean(state.irradiation()[state.occupancy() == 1]), p2_states)) / 500
print("Protocol 2 mean irradiation: ", mR)
