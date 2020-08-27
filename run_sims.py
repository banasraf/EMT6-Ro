import emt6ro.simulation as sim
import numpy as np
import time
import sys
params = sim.load_parameters("data/default-parameters.json")
states = [sim.load_state("data/tumor-lib/tumor-{}.txt".format(i), params) for i in range(1, 11)]
protocols = []
with open('protocols_3007.txt') as pfile:
    for line in pfile:
        doses = [float(d) for d in line.strip().split()]
        times = [int(t) - 1 for t in pfile.readline().strip().split()]
        protocol = list(zip(times, doses))
        protocols.append(protocol)

       
experiment = sim.Experiment(params, states, 100, 1)  

for prot in protocols:
    start = time.time()
    experiment.run([prot])
    res = experiment.get_results()
    end = time.time()
    print('time: ', end - start, file=sys.stderr)
    for t in range(10):
        for r in range(100):
            print(res[0, t, r], end=' ')
    print(' ')


