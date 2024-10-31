from dvrk.vision import vision_constants as cst
from dvrk.vision.cameras.Camera import Camera
import cv2
import numpy as np
import time
from old_motion.dvrkArmNew import dvrkArm
import os
import utils.CmnUtil as U

# PRINT DOESN'T WORK
NUM_POINTS = 30
JAW_OPEN_ANGLE = [np.pi / 2]
JAW_CLOSE_ANGLE = [-0.3]  # Angle for closing jaw to grasp
pos_des = []
pos_act = []


def click_event(event, u, v, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel_value = param["img"][v, u]
        print(f"Pixel coordinates: (u={u}, v={v}) - Pixel value: {pixel_value}")

        # Display the coordinates and pixel value on the image
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(
            param["img"],
            f"({u},{v})",
            (u, v),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            param["img"],
            str(pixel_value),
            (u, v + 20),
            font,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        time.sleep(0.5)
        param["u"] = u
        param["v"] = v


def load_trajectory(filename):
    joint = np.load(filename)
    indices = np.random.choice(joint.shape[0], size=NUM_POINTS * 4, replace=False)
    # Use these indices to create your subsampled (30, 6) array
    subsampled_joints = joint[indices]
    subsampled_joints[:, -3:] = 0
    return subsampled_joints


# Uncomment this line in ZividCapture.py for dvrk_2024 so you get depth readings at the gripper self.settings.processing.filters.reflection.removal.enabled = False
zivid_cam = Camera(cst.ZIVID, zivid_capture_type="3d")

psm_number = input("Which PSM are you calibrating. Pick 1 or 2 ")
psm_string = ""
if psm_number == "1":
    psm_string = "/PSM1"
elif psm_number == "2":
    psm_string = "/PSM2"
else:
    print("Please select 1 or 2")
    exit()
psm = dvrkArm(psm_string, use_rnn=False)
if psm_string == "1":
    # Home position
    psm.set_joint(joint=[0.21441395551996928, -0.2575145276910163, 0.12146408823000001, 0.0, 0.0, 0.0])
    psm.set_jaw(jaw=JAW_CLOSE_ANGLE)
    time.sleep(1)
root_path = os.path.dirname(os.path.abspath(__file__))
calibration_output_path = os.path.join(root_path, "experiment/0_trajectory_extraction/calibration_outputs")
filename = os.path.join(calibration_output_path, "prime_psm" + psm_number + "_random_sampled.npy")
joints = load_trajectory(filename)
i = 0
for joint in joints:
    psm.set_joint(joint)
    time.sleep(1)
    zivid_image, zivid_depth, zivid_pcl, intrinsics_matrix, distortion_coefficients = zivid_cam.capture()
    K_zivid = zivid_cam.cam.intrinsics_
    D_zivid = zivid_cam.cam.distortion_coefficients_

    cv2.imshow("Pick pixel.", zivid_image)
    params = {"img": zivid_image.copy(), "u": None, "v": None}
    cv2.setMouseCallback("Pick pixel.", click_event, param=params)

    # Wait until a key is pressed and the coordinates are set
    while params["u"] is None or params["v"] is None:
        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break
    cv2.destroyAllWindows()
    # Retrieve the u, v pixel coordinates from the callback
    pixel_u, pixel_v = params["u"], params["v"]
    is_nan = np.isnan(zivid_pcl[pixel_v, pixel_u]).any()
    depth_u, depth_v = pixel_u, pixel_v
    if is_nan:
        print("No depth value here")
    else:
        camera_point = zivid_pcl[depth_v, depth_u] / 1000
        robot_point, _ = psm.get_current_pose()
        pos_des.append(camera_point)
        pos_act.append(robot_point)
    if len(pos_des) >= NUM_POINTS:
        break
T = U.get_rigid_transform(np.array(pos_des), np.array(pos_act))
np.save(os.path.join(calibration_output_path, "prime_psm" + str(psm_number) + "_robot_to_zivid"), T)
