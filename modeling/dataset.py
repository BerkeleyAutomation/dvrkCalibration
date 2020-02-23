import numpy as np

class Dataset:
	def __init__(self, inputs, outputs):
		self.inputs = inputs
		self.outputs = outputs
		self.joined_data = np.hstack((self.inputs, self.outputs))
		self.input_dim = self.inputs.shape[1]
		self.output_dim = self.outputs.shape[1]

	def __len__(self):
		return self.inputs.shape[0]

	def sample_batch(self, batch_size=None):
		if batch_size == None:
			return self.inputs, self.outputs
		indices = np.random.choice(len(self), batch_size)
		batch = self.joined_data[indices]
		return batch[:,:self.input_dim], batch[:,self.input_dim:]

	def __call__(self, batch_size=None):
		return self.sample_batch(batch_size)
	
