import numpy as np


class Dataset:
	def __init__(self, histories, cmds, phys):
		self.histories = histories
		self.cmds = cmds
		self.phys = phys
		self.input_dim = self.histories.shape[1]
		self.output_dim = self.cmds.shape[1]

	def __len__(self):
		return self.histories.shape[0]

	def sample_batch(self, batch_size=None, ret_indices=False):
		if batch_size is None:
			if not ret_indices:
				return self.histories, self.cmds, self.phys
			else:
				return self.histories, self.cmds, self.phys, np.arange(len(self.histories))
		else:
			indices = np.random.choice(len(self), batch_size)
			if ret_indices:
				return self.histories[indices], self.cmds[indices], self.phys[indices], indices
			else:
				return self.histories[indices], self.cmds[indices], self.phys[indices]

	def __call__(self, batch_size=None):
		return self.sample_batch(batch_size)
