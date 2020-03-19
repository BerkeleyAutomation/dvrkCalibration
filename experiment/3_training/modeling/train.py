import os
import os.path as osp
import pickle
import pprint

import datetime
from dotmap import DotMap
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim import Adam
import torch.nn.functional as F

from .dataset import Dataset
from .models import CalibrationModel, CalibrationLSTM


def load_data(data_dir):
	joint_actual = np.load(osp.join(data_dir, "joint_act.npy"))
	joint_desired = np.load(osp.join(data_dir, "joint_des.npy"))
	position_actual = np.load(osp.join(data_dir, "position_act.npy"))
	position_desired = np.load(osp.join(data_dir, "position_des.npy"))
	quaternion_actual = np.load(osp.join(data_dir, "quaternion_act.npy"))
	quaternion_desired = np.load(osp.join(data_dir, "quaternion_des.npy"))

	return {
		"joint_actual": joint_actual,
		"position_actual": position_actual,
		"quaternion_actual": quaternion_actual,
		"joint_desired": joint_desired,
		"position_desired": position_desired,
		"quaternion_desired": quaternion_desired
	}


def format_data(H, fname, use_actual_inputs=False, rnn=False):
	"""
	Formats histories from oldest to newest.
	"""
	data = load_data(fname)
	desired = data["joint_desired"][:,3:]
	actual = data["joint_actual"][:,3:]
	histories = []
	for i in range(H):
		if use_actual_inputs:
			histories.append(actual[i:len(desired) - H + i])
		else:	
			histories.append(desired[i:len(desired) - H + i])
	cmds = desired[H:]
	phys = actual[H:]
	if rnn:
		histories = np.stack(histories, axis=1)
	else:
		histories = np.hstack(histories)
	return histories, cmds, phys

def compute_standard_loss(model, batch_histories, batch_cmds, batch_phys, is_forward, is_rnn, device):
	if is_forward:
		if is_rnn:
			batch_inputs = np.hstack((batch_histories, batch_cmds[:,np.newaxis,:]))
			batch_outputs = batch_phys
		else:
			batch_inputs = np.hstack((batch_histories, batch_cmds))
			batch_outputs = batch_phys
	else:
		if is_rnn:
			batch_inputs = np.hstack((batch_histories, batch_phys[:,np.newaxis,:]))
			batch_outputs = batch_cmds
		else:
			batch_inputs = np.hstack((batch_histories, batch_phys))
			batch_outputs = batch_cmds
	batch_inputs = torch.FloatTensor(batch_inputs).to(device)
	batch_outputs = torch.FloatTensor(batch_outputs).to(device)

	preds = model(batch_inputs)
	return F.mse_loss(preds, batch_outputs)

def compute_cyclic_losses(forward_model, inverse_model, batch_histories, batch_cmds, batch_phys, is_rnn, device):
	if is_rnn:
		batch_forward_inputs = np.hstack((batch_histories, batch_cmds[:,np.newaxis,:]))
		batch_inverse_inputs = np.hstack((batch_histories, batch_phys[:,np.newaxis,:]))
	else:
		batch_forward_inputs = np.hstack((batch_histories, batch_cmds))
		batch_inverse_inputs = np.hstack((batch_histories, batch_phys))
	batch_forward_inputs = torch.FloatTensor(batch_forward_inputs).to(device)
	batch_inverse_inputs = torch.FloatTensor(batch_inverse_inputs).to(device)
	batch_histories = torch.FloatTensor(batch_histories).to(device)
	batch_cmds = torch.FloatTensor(batch_cmds).to(device)
	batch_phys = torch.FloatTensor(batch_phys).to(device)

	pred_states = forward_model(batch_forward_inputs)
	pred_actions = inverse_model(batch_inverse_inputs)

	if is_rnn:
		reconstructed_actions = inverse_model(torch.cat((batch_histories, pred_states.unsqueeze(1)), 1))
		reconstructed_states = forward_model(torch.cat((batch_histories, pred_actions.unsqueeze(1)), 1))
	else:
		reconstructed_actions = inverse_model(torch.cat((batch_histories, pred_states), 1))
		reconstructed_states = forward_model(torch.cat((batch_histories, pred_actions), 1))

	action_loss = F.mse_loss(reconstructed_actions, batch_cmds)
	state_loss = F.mse_loss(reconstructed_states, batch_phys)
	return action_loss + state_loss

untorchify = lambda x: x.detach().cpu().numpy()


class Experiment:

	def __init__(self, config):
		self.config = config
		self.device = torch.device(config.device)

		# Load and format data		
		histories, cmds, phys = format_data(config.history, config.training_data, config.actual_inputs, config.rnn)
		self.validation_size = int(config.validation_prob * len(histories))
		self.training_histories = histories[self.validation_size:]
		self.training_cmds = cmds[self.validation_size:]
		self.training_phys = phys[self.validation_size:]
		# TODO: use other trajectory for validation
		self.val_histories = histories[:self.validation_size]
		self.val_cmds = cmds[:self.validation_size]
		self.val_phys = phys[:self.validation_size]
		self.config.input_dim = self.training_histories.shape[2] if config.rnn else self.training_histories.shape[1] + self.training_phys.shape[1]
		self.config.output_dim = self.training_phys.shape[1]

		# Wrap training and validation datasets
		self.dataset = Dataset(self.training_histories, self.training_cmds, self.training_phys)
		self.val_dataset = Dataset(self.val_histories, self.val_cmds, self.val_phys)

		self.batch_size = config.batch_size
		self.training_iterations = config.training_iterations
		self.save_dir = config.save_dir
		self.save_freq = config.save_freq
		self.log_freq = config.log_freq

		# Set up models and optimizers
		if config.rnn:
			self.forward_model = CalibrationLSTM(self.config.input_dim, self.config.output_dim)
			self.inverse_model = CalibrationLSTM(self.config.input_dim, self.config.output_dim)
		else:
			self.forward_model = CalibrationModel(self.config.input_dim, self.config.output_dim)
			self.inverse_model = CalibrationModel(self.config.input_dim, self.config.output_dim)
		self.optimizer = Adam(list(self.forward_model.parameters()) + list(self.inverse_model.parameters()), lr=config.lr)
		self.rnn = config.rnn
		self.consistency_weight = config.consistency_weight

		# Initializes data structures for logging.
		self.reset()

		# Create a directory for saving models and logs
		if not osp.exists(self.save_dir):
			os.makedirs(self.save_dir)
		self.config.save_dir = self.save_dir = osp.join(self.save_dir, datetime.datetime.now().strftime("%Y-%m-%d--%H:%M:%S"))
		os.makedirs(self.save_dir)
		with open(osp.join(self.save_dir, "config.pkl"), "wb") as f:
			pickle.dump(self.config, f)
		with open(osp.join(self.save_dir, "config.txt"), "w") as f:
			f.write(pprint.pformat(self.config.toDict()))


	def reset(self):
		"""
		Resets logging data structures
		"""
		self.logs = DotMap()
		self.logs.forward.training_losses = []
		self.logs.forward.validation_losses = []
		self.logs.inverse.training_losses = []
		self.logs.inverse.validation_losses = []
		self.logs.consistency.training_losses = []
		self.logs.consistency.validation_losses = []


	def run(self):
		for i in range(self.training_iterations):
			# Sample training batch, compute loss, then take a step with Adam
			self.optimizer.zero_grad()
			batch_histories, batch_cmds, batch_phys = self.dataset(self.batch_size)
			fwd_loss = compute_standard_loss(self.forward_model, batch_histories, batch_cmds, batch_phys, True, self.rnn, self.device)
			inv_loss = compute_standard_loss(self.inverse_model, batch_histories, batch_cmds, batch_phys, False, self.rnn, self.device)
			consistency_loss = compute_cyclic_losses(self.forward_model, self.inverse_model, batch_histories, batch_cmds, batch_phys, self.rnn, self.device)

			loss = fwd_loss + inv_loss + self.consistency_weight * consistency_loss
			loss.backward()
			self.optimizer.step()

			# Compute validation losses
			val_batch_histories, val_batch_cmds, val_batch_phys = self.val_dataset()
			with torch.no_grad():
				fwd_loss_val = compute_standard_loss(self.forward_model, val_batch_histories, val_batch_cmds, val_batch_phys, True, self.rnn, self.device)
				inv_loss_val = compute_standard_loss(self.inverse_model, val_batch_histories, val_batch_cmds, val_batch_phys, False, self.rnn, self.device)
				consistency_loss_val = compute_cyclic_losses(self.forward_model, self.inverse_model, val_batch_histories, val_batch_cmds, val_batch_phys, self.rnn, self.device)

			# Log loss data
			self.logs.forward.training_losses.append(untorchify(fwd_loss))
			self.logs.forward.validation_losses.append(untorchify(fwd_loss_val))
			self.logs.inverse.training_losses.append(untorchify(inv_loss))
			self.logs.inverse.validation_losses.append(untorchify(inv_loss_val))
			self.logs.consistency.training_losses.append(untorchify(consistency_loss))
			self.logs.consistency.validation_losses.append(untorchify(consistency_loss_val))

			# Print losses and save models periodically
			if i % self.log_freq == 0:
				print("Iteration %d. Forward Training Loss: %8.5f Forward Validation Loss: %8.5f Inverse Training Loss %8.5f Inverse Validation Loss %8.5f Consistency Training Loss %8.5f Consistency Validation Loss %8.5f"\
						%(i, untorchify(fwd_loss), untorchify(fwd_loss_val), untorchify(inv_loss),
						untorchify(inv_loss_val), untorchify(consistency_loss), untorchify(consistency_loss_val)))
			if i % self.save_freq == 0:
				self.save("model_iter%d.out"%i)

		# Save final model
		self.save("model.out")
		# Save plots
		self.plot(save=True, show=False)

	def save(self, fname):
		torch.save(self.forward_model.state_dict(), osp.join(self.save_dir, fname))

	def plot(self, save=False, show=True):
		plt.clf()
		val_batch_histories, val_batch_cmds, val_batch_phys = self.val_dataset()
		original_loss = np.square(val_batch_phys - val_batch_cmds).mean()

		# Plot Forward Losses
		plt.plot(np.ones_like(self.logs.forward.training_losses) * original_loss, c="g", label="Original Loss (No correction)")
		plt.plot(self.logs.forward.training_losses, c="r", label="Training Loss")
		plt.plot(self.logs.forward.validation_losses, c="b", label="Validation Loss")
		plt.legend(loc="upper right")
		plt.title("Forward Model Loss Curve")
		plt.xlabel("Iteration")
		plt.ylabel("MSE Loss")
		plt.ylim(0, 0.05)
		if save:
			plt.savefig(osp.join(self.save_dir, "forward_losses.png"))
		if show:
			plt.show()

		# Plot Inverse Losses
		plt.plot(np.ones_like(self.logs.forward.training_losses) * original_loss, c="g", label="Original Loss (No correction)")
		plt.plot(self.logs.inverse.training_losses, c="r", label="Training Loss")
		plt.plot(self.logs.inverse.validation_losses, c="b", label="Validation Loss")
		plt.legend(loc="upper right")
		plt.title("Inverse Model Loss Curve")
		plt.xlabel("Iteration")
		plt.ylabel("MSE Loss")
		plt.ylim(0, 0.05)
		if save:
			plt.savefig(osp.join(self.save_dir, "inverse_losses.png"))
		if show:
			plt.show()

		# Plot Consistency Losses
		plt.plot(self.logs.consistency.training_losses, c="r", label="Training Loss")
		plt.plot(self.logs.consistency.validation_losses, c="b", label="Validation Loss")
		plt.legend(loc="upper right")
		plt.title("Consistency Loss Curve")
		plt.xlabel("Iteration")
		plt.ylabel("MSE Loss")
		plt.ylim(0, 0.05)
		if save:
			plt.savefig(osp.join(self.save_dir, "consistency_losses.png"))
		if show:
			plt.show()


def create_config():
	"""
	Set up experimental parameters here.
	"""
	config = DotMap()
	config.peg_data = "training_dataset_brijen/peg_transfer"
	config.random_data = "training_dataset_brijen/random"
	config.training_data = config.random_data # which dataset to train on
	config.actual_inputs = False # whether to use actual as input
	config.history = 10
	config.batch_size = 100
	config.training_iterations = 5000
	config.lr = 1e-3
	config.device = "cpu" # switch to cuda if desired
	config.save_dir = "log" # where to save logs and models
	config.save_freq = 10 # how often to save model
	config.log_freq = 10 # how often to print progress
	config.validation_prob = 0.1
	config.rnn = False
	config.consistency_weight = 1. # set to 0 to not use cyclic consistency loss
	return config


if __name__ == '__main__':
	config = create_config()
	experiment = Experiment(config)
	experiment.run()
	experiment.plot()
