from dvrk.vision import vision_constants as cst
from dvrk.vision.cameras.Camera import Camera
import cv2
import numpy as np
import time
from old_motion.dvrkArmNew import dvrkArm
import os
import utils.CmnUtil as U
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import torch
pos_act = np.load(
    "/home/davinci/dvrkCalibration/experiment/0_trajectory_extraction/calibration_outputs/psm1_pos_act.npy"
)
pos_des = np.load(
    "/home/davinci/dvrkCalibration/experiment/0_trajectory_extraction/calibration_outputs/psm1_pos_des.npy"
)
psm1_to_zivid_initial = U.get_rigid_transform(np.array(pos_act), np.array(pos_des))
print("Initial psm1 to zivid")
print(psm1_to_zivid_initial)
def loss(x):
    pos = x[0:3]
    quat = x[3:]
    rotation_matrix = R.from_quat(quat).as_matrix()
    psm1_to_zivid = np.eye(4)
    psm1_to_zivid[:3,:3] = rotation_matrix
    psm1_to_zivid[:3,3] = pos
    errors = []
    preds = []
    targets = []
    for psm_pos, cam_pos in zip(pos_des, pos_act):
        cam_pos_homogenous = np.append(cam_pos, 1).reshape(-1, 1)
        estimated_psm_pos_homogenous = psm1_to_zivid @ cam_pos_homogenous
        estimated_psm_pos = estimated_psm_pos_homogenous[:-1] / estimated_psm_pos_homogenous[-1]
        estimated_psm_pos = estimated_psm_pos.reshape(
            -1,
        )
        preds.append(estimated_psm_pos)
        targets.append(psm_pos)
        error = np.linalg.norm(psm_pos - estimated_psm_pos)
        errors.append(error)
    np_errors = np.array(errors)
    torch_preds = torch.tensor(np.array(preds))
    torch_targets = torch.tensor(np.array(targets))
    torch_loss = torch.nn.MSELoss()
    loss_value = torch_loss(torch_preds,torch_targets).item()
    print(loss_value)
    return loss_value

initial_rotation_matrix = psm1_to_zivid_initial[:3,:3]
initial_pos = psm1_to_zivid_initial[:3,3]
initial_quat = R.from_matrix(initial_rotation_matrix).as_quat()
x0 = np.concatenate((initial_pos, initial_quat))
res = minimize(loss, x0,method='Newton-CG')
print(res)
pos = res.x[0:3]
quat = res.x[3:]
rotation_matrix = R.from_quat(quat).as_matrix()
psm1_to_zivid = np.eye(4)
psm1_to_zivid[:3,:3] = rotation_matrix
psm1_to_zivid[:3,3] = pos
errors = []
for psm_pos, cam_pos in zip(pos_des, pos_act):
    cam_pos_homogenous = np.append(cam_pos, 1).reshape(-1, 1)
    estimated_psm_pos_homogenous = psm1_to_zivid @ cam_pos_homogenous
    estimated_psm_pos = estimated_psm_pos_homogenous[:-1] / estimated_psm_pos_homogenous[-1]
    estimated_psm_pos = estimated_psm_pos.reshape(
        -1,
    )
    error = np.linalg.norm(psm_pos - estimated_psm_pos)
    errors.append(error)
np_errors = np.array(errors)
mean_error = np.mean(np_errors)
std_error = np.std(np_errors)
print("Mean error is " + str(mean_error * 1000) + " mm.")
print("Std error is " + str(std_error * 1000) + " mm.")
