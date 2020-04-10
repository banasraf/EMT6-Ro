from emt6ro.simulation import load_state, load_parameters
from emt6ro.ga_model import EMT6RoModel

params = load_parameters("data/default-parameters.json")
tumors = [load_state("data/tumor-lib/tumor-{}.txt".format(i), params) for i in range(1, 11)]

num_gpus = 2
num_protocols = 2
num_tests = 20

model = EMT6RoModel(params, tumors, num_protocols, num_tests, num_gpus) 

hour_steps = 600
prot1 = [(hour_steps * 12, 1.25), (hour_steps * 36, 3.0)]
prot2 = [(hour_steps * 12, 1.25), (hour_steps * 24, 1.5), (hour_steps * 36, 1.5)]

prots = [prot1, prot2]

print(model.predict(prots))

