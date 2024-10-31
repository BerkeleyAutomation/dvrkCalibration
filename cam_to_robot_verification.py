from dvrk.vision import vision_constants as cst
from dvrk.vision.cameras.Camera import Camera
import cv2
import numpy as np
import time
from old_motion.dvrkArmNew import dvrkArm
import os
import utils.CmnUtil as U

pos_actual = np.load(
    "/home/davinci/dvrkCalibration/experiment/0_trajectory_extraction/calibration_outputs/psm1_pos_act.npy"
)
pos_des = np.load(
    "/home/davinci/dvrkCalibration/experiment/0_trajectory_extraction/calibration_outputs/psm1_pos_des.npy"
)
psm1_to_zivid = np.load(
    "/home/davinci/dvrkCalibration/experiment/0_trajectory_extraction/calibration_outputs/psm1_robot_to_zivid.npy"
)
errors = []
for psm_pos, cam_pos in zip(pos_des, pos_actual):
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
