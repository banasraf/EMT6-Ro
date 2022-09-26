from .backend import _Experiment, load_parameters, load_state
import numpy as np

class Experiment:
	"""
	Reusable simulation object with fixes initial tumors states.
	"""
	def __init__(self, params, tumors, runs, protocols_num, gpu_id=0, simulation_steps=144000, protocol_resolution=300):
		"""
		Parameters
		----------
		params 
			Simulation parameters
		tumors
			Initial tumors states
		runs
			Number of simulation runs for each protocol and tumor
		protocols_num
			Number of protocols to test
		gpu_id
		
		simulation_steps
			Length of simulation in steps
		protocol_resolution
			Time resolution of irradiation protocols
		"""
		self.result_shape = (protocols_num, len(tumors), runs)
		self._experiment = _Experiment(params, tumors, runs, protocols_num, simulation_steps, protocol_resolution, gpu_id)

	def run(self, nsteps):
		"""
		Run nsteps steps of the simulation
		
		Parameters
		----------
		nsteps - number of simulation steps
		"""
		self._experiment.run(nsteps)

	def get_results(self):
		"""
		Waits for the end of previously started simulation and returns
		the final number of living cells for each run.
		
		Returns a 3-dimensional numpy array with dimensions being:
		(protocol, tumor, run)
		"""
		res = np.array(self._experiment.results())
		return res.reshape(self.result_shape)

	def add_irradiations(self, protocols):
		"""
		Add irradiations.

		Parameters
		----------
		protocols - list of pairs (time, dose)
		"""
		self._experiment.add_irradiations(protocols)

	def reset(self):
		"""
		Restore the initial states of the simulations
		"""
		self._experiment.reset()

	def state(self):
		"""
		Get all of the simlation grids
		"""
		return self._experiment.state()
