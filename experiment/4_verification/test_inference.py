import pickle
import os
import sys
current_filepath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_filepath,'../3_training/modeling'))
from models import CalibrationModel, CalibrationLSTM
import torch
import numpy as np
#TODO: Update forward model, inverse model, and config filenames
forward_model_filename = "/home/davinci/dvrkCalibration/experiment/3_training/modeling/log/2024-10-23--03:27:36/model_prime_forward.out"
inverse_model_filename = "/home/davinci/dvrkCalibration/experiment/3_training/modeling/log/2024-10-23--03:27:36/model_prime_inverse.out"
config_filename = "/home/davinci/dvrkCalibration/experiment/3_training/modeling/log/2024-10-23--03:27:36/config.pkl"
sample_desired_data = np.load('/home/davinci/dvrkCalibration/old_data/psm2_q_des_raw_801.npy')
sample_actual_data = np.load('/home/davinci/dvrkCalibration/old_data/psm2_q_act_raw_801.npy')
forward_rnn_model_ = None
inverse_rnn_model_ = None
with open(config_filename, "rb") as f:
    config = pickle.load(f)
if config.rnn:
    forward_rnn_model_ = CalibrationLSTM(config.input_dim, config.output_dim)
    inverse_rnn_model_ = CalibrationLSTM(config.input_dim, config.output_dim)
else:
    forward_rnn_model_ = CalibrationModel(config.input_dim, config.output_dim)
    inverse_rnn_model_ = CalibrationModel(config.input_dim, config.output_dim)
inverse_rnn_model_.load_state_dict(torch.load(inverse_model_filename))
inverse_rnn_model_.eval()
forward_rnn_model_.load_state_dict(torch.load(forward_model_filename))
forward_rnn_model_.eval()
joint_history_size_ = config.history
joint_history_ = []

def forward_inference(input):
    if len(joint_history_) == joint_history_size_:
        joint_history = np.array(joint_history_)
        desired_joint_cmd = np.array(input)
        model_input_np = np.vstack((joint_history, desired_joint_cmd[np.newaxis, :]))
        model_input_np = model_input_np[np.newaxis, :, :]
        model_input = torch.FloatTensor(model_input_np)
        model_output = forward_rnn_model_(model_input)
        model_output_np = model_output.detach().numpy()[0]
        return model_output_np

def inverse_inference():
    pass

i = 0
while (i < joint_history_size_):
    joint_history_.append(sample_desired_data[i])
    i += 1
while(i < sample_desired_data.shape[0]):
    forward_input = sample_desired_data[i]
    gt_output = sample_actual_data[i]
    forward_output = forward_inference(forward_input)
    print("No NN error: " + str(np.linalg.norm(gt_output - forward_input)))
    print("NN error: " + str(np.linalg.norm(gt_output - forward_output)))
    import pdb
    pdb.set_trace()
    joint_history_.pop(0)
    joint_history_.append(forward_input)
    i += 1