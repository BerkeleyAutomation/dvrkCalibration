import os
import os.path as osp
import pickle

import datetime
from dotmap import DotMap
import matplotlib.pyplot as plt
import numpy as np
import pprint
import torch
from torch.optim import Adam
import torch.nn.functional as F

from .dataset import Dataset
from .models import CalibrationModel, CalibrationLSTM
from .utils import load_data, join_data, flatten_data, incorporate_history


class Experiment:

	def __init__(self, config):
		self.config = config
		self.device = torch.device(config.device)

		self.peg_data_full = load_data(config.peg_data)
		self.random_data_full = load_data(config.random_data)
		self.data_full = join_data(self.peg_data_full, self.random_data_full)
		self.training_inputs, self.training_outputs = incorporate_history(
			flatten_data(self.data_full, config.input_src),
			flatten_data(self.data_full, config.output_src),
			config.history, rnn=config.rnn)
		self.original_preds = torch.FloatTensor(flatten_data(self.data_full, config.original_preds)).to(self.device)
		self.original_loss = F.mse_loss(
			self.original_preds,
			torch.FloatTensor(flatten_data(self.data_full, config.output_src)).to(self.device)
			).detach().cpu().numpy()
		self.use_original_preds = config.use_original_preds

		self.validation_size = int(config.validation_prob * len(self.training_inputs))
		self.training_inputs = self.training_inputs[self.validation_size:]
		self.training_outputs = self.training_outputs[self.validation_size:]
		self.val_inputs = self.training_inputs[:self.validation_size]
		self.val_outputs = self.training_outputs[:self.validation_size]
		self.config.input_dim = self.training_inputs.shape[2] if config.rnn else self.training_inputs.shape[1]
		self.config.output_dim = self.training_outputs.shape[1]

		self.dataset = Dataset(self.training_inputs, self.training_outputs)
		self.val_dataset = Dataset(self.val_inputs, self.val_outputs)

		self.batch_size = config.batch_size
		self.training_iterations = config.training_iterations
		self.save_dir = config.save_dir
		self.save_freq = config.save_freq
		self.log_freq = config.log_freq

		if config.rnn:
			self.model = CalibrationLSTM(self.training_inputs.shape[2], self.training_outputs.shape[1])
		else:
			self.model = CalibrationModel(self.training_inputs.shape[1], self.training_outputs.shape[1])
		self.optimizer = Adam(self.model.parameters(), lr=config.lr)
		self.rnn = config.rnn

		self.training_losses = []
		self.validation_losses = []

		if not osp.exists(self.save_dir):
			os.makedirs(self.save_dir)
		self.config.save_dir = self.save_dir = osp.join(self.save_dir, datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))
		os.makedirs(self.save_dir)
		with open(osp.join(self.save_dir, "config.pkl"), "wb") as f:
			pickle.dump(self.config, f)
		with open(osp.join(self.save_dir, "config.txt"), "w") as f:
			f.write(pprint.pformat(self.config.toDict()))

	def reset(self):
		self.training_losses = []
		self.validation_losses = []

	def run(self):
		for i in range(self.training_iterations):
			self.optimizer.zero_grad()
			batch_inputs, batch_outputs, batch_indices = self.dataset(self.batch_size, ret_indices=True)
			batch_inputs = torch.FloatTensor(batch_inputs).to(self.device)
			batch_outputs = torch.FloatTensor(batch_outputs).to(self.device)
			if self.rnn:
				preds = []
				# TODO: vectorize the following code
				for j in range(self.batch_size):
					input_seq = batch_inputs[j]
					self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size),
							torch.zeros(1, 1, self.model.hidden_layer_size))
					preds.append(self.model(input_seq))
				preds = torch.stack(preds)
			else:
				preds = self.model(batch_inputs)
			if self.use_original_preds:
				preds = preds + self.original_preds[batch_indices + self.validation_size]
			loss = F.mse_loss(preds, batch_outputs)
			loss.backward()
			self.optimizer.step()

			val_batch_inputs, val_batch_outputs, val_batch_indices = self.val_dataset(ret_indices=True)
			val_batch_inputs = torch.FloatTensor(val_batch_inputs).to(self.device)
			val_batch_outputs = torch.FloatTensor(val_batch_outputs).to(self.device)
			with torch.no_grad():
				if self.rnn:
					val_preds = []
					for j in range(len(val_batch_inputs)):
						input_seq = val_batch_inputs[j]
						self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size),
								torch.zeros(1, 1, self.model.hidden_layer_size))
						val_preds.append(self.model(input_seq))
					val_preds = torch.stack(val_preds)
				else:
					val_preds = self.model(val_batch_inputs)
				if self.use_original_preds:
					val_preds = val_preds + self.original_preds[val_batch_indices]
				val_loss = F.mse_loss(val_preds, val_batch_outputs)

			self.training_losses.append(loss.detach().cpu().numpy())
			self.validation_losses.append(val_loss.detach().cpu().numpy())

			if i % self.log_freq == 0:
				print("Iteration %d. Training Loss: %8.5f. Validation Loss: %8.5f"
					%(i, self.training_losses[-1], self.validation_losses[-1]))
			if i % self.save_freq == 0:
				self.save("model_iter%d"%i)
		self.save("model")
		self.plot(save=True, show=False)

	def save(self, fname):
		torch.save(self.model.state_dict(), osp.join(self.save_dir, fname))

	def plot(self, save=False, show=True):
		plt.clf()
		plt.plot(np.ones_like(self.training_losses) * self.original_loss, c="g", label="Original Loss (No correction)")
		plt.plot(self.training_losses, c="r", label="Training Loss")
		plt.plot(self.validation_losses, c="b", label="Validation Loss")
		plt.legend(loc="upper right")
		plt.title("Loss Curve")
		plt.xlabel("Iteration")
		plt.ylabel("MSE Loss")
		plt.ylim(0, 0.05)
		if save:
			plt.savefig(osp.join(self.save_dir, "losses.png"))
		if show:
			plt.show()


def create_config():
	config = DotMap()
	config.peg_data = "training_dataset_brijen/peg_transfer"
	config.random_data = "training_dataset_brijen/random"
	config.input_src = ["joint_desired", "quaternion_desired", "joint_actual"]
	config.output_src = ["joint_actual"]
	config.original_preds = ["joint_desired"]
	config.use_original_preds = False # this overfits for some reason...
	config.history = 5
	config.batch_size = 100
	config.training_iterations = 5000
	config.lr = 1e-3
	config.device = "cpu"
	config.save_dir = "log"
	config.save_freq = 10
	config.log_freq = 10
	config.validation_prob = 0.1
	config.rnn = False
	return config

if __name__ == '__main__':
	config = create_config()
	experiment = Experiment(config)
	experiment.reset()
	experiment.run()
	experiment.plot()
