import cv2
import numpy as np
import time
import os
import utils.CmnUtil as U

psm_number_input = input("Which PSM are you calibrating. Pick 1 or 2 ")
if not psm_number_input == "1" and not psm_number_input == "2":
    print("Please select 1 or 2")
    exit()

root_path = os.path.dirname(os.path.abspath(__file__))
calibration_output_path = os.path.join(
    root_path, "experiment/0_trajectory_extraction/shallow_and_deep_calibration_outputs"
)
pos_actual = np.load(calibration_output_path + "/psm" + psm_number_input + "_pos_act.npy")
pos_des = np.load(calibration_output_path + "/psm" + psm_number_input + "_pos_des.npy")
psm_to_zivid = np.load(calibration_output_path + "/psm" + psm_number_input + "_robot_to_zivid.npy")

errors = []
for psm_pos, cam_pos in zip(pos_des, pos_actual):
    cam_pos_homogenous = np.append(cam_pos, 1).reshape(-1, 1)
    estimated_psm_pos_homogenous = psm_to_zivid @ cam_pos_homogenous
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
