import os.path as osp
import pickle

import torch

from .models import CalibrationModel

def load_model(model_dir, model_name=None):
	if model_name is not None:
		model_file = osp.join(model_dir, model_name)
	else:
		model_file = osp.join(model_dir, "model")

	config_file = osp.join(model_dir, "config.pkl")
	with open(config_file, "rb") as f:
		config = pickle.load(f)

	model = CalibrationModel(config.input_dim, config.output_dim)
	model.load_state_dict(torch.load(model_file))
	model.eval()

	return model


if __name__ == '__main__':
	model = load_model(osp.join("log", "2020-02-23--17:42:16"))
