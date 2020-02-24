import numpy as np

class Dataset:
	def __init__(self, inputs, outputs):
		self.inputs = inputs
		self.outputs = outputs
		self.input_dim = self.inputs.shape[1]
		self.output_dim = self.outputs.shape[1]

	def __len__(self):
		return self.inputs.shape[0]

	def sample_batch(self, batch_size=None, ret_indices=False):
		if batch_size is None:
			if not ret_indices:
				return self.inputs, self.outputs
			else:
				return self.inputs, self.outputs, np.arange(len(self.inputs))
		else:
			indices = np.random.choice(len(self), batch_size)
			if ret_indices:
				return self.inputs[indices], self.outputs[indices], indices
			else:
				return self.inputs[indices], self.outputs[indices]

	def __call__(self, batch_size=None, ret_indices=False):
		return self.sample_batch(batch_size, ret_indices)
	
