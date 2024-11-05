import os

from datetime import datetime
import numpy as np
import cv2
import time

from dvrk.vision import vision_constants as cst
from dvrk.vision.cameras.Camera import Camera
from dvrk.motion.dvrkArm import dvrkArm
from autolab_core import RigidTransform
from dvrk.motion.dvrkTypes import dvrkTypes
from dvrk.vision.cameras.AlliedVisionUtils import AlliedVisionUtils
import dvrk.utils.CmnUtil as utils

JAW_OPEN_ANGLE = [np.pi / 2]
JAW_CLOSE_ANGLE = [-0.3]  # Angle for closing jaw to grasp

home_joints = None


def random_quaternion():
    """
    TODO: Consider refactoring. New function (fully tested, ready for copy-paste) in psm_random_dance.py
    """
    q = None
    while q is None:
        # Note: comment out first line of each pair for horizontal positions, second line for vertical ones

        # quaternion = utils.euler_to_quaternion(np.random.uniform([np.pi/2 - .1, 0, np.pi/4], [np.pi/2 + .1, np.pi*2, 7*np.pi/4]))
        quaternion = utils.euler_to_quaternion(np.random.uniform([0, 0, 0], [np.pi * 2, np.pi * 2, np.pi * 2]))

        R = utils.quaternion_to_R(quaternion)

        # ax = 2
        ax = 1

        proj = np.array([[0], [0], [0]])
        proj[ax] = 1
        dir = np.dot(R, proj)

        # if dir[ax] > min_z:
        if dir[2] > 0.85:
            q = quaternion

    return q


def plot_psm_pos_on_img(img, psm_p_robot, tf_cam_psm, K, D):
    rvec_cam_psm, _ = cv2.Rodrigues(tf_cam_psm.rotation)

    psm_p_cam, _ = cv2.projectPoints(psm_p_robot, rvec_cam_psm, tf_cam_psm.translation, K, D)
    psm_p_cam = psm_p_cam.squeeze().astype(int)
    img = cv2.circle(img, (psm_p_cam[0], psm_p_cam[1]), 5, (255, 0, 0), -1)
    return img


def single_psm_warmup(psm_number, psm):
    psm.set_jaw(JAW_CLOSE_ANGLE)
    psm_warmup_points = []
    psm_warmup_quats = []
    if psm_number == "1":
        psm_warmup_points = [
            np.array([[0.01], [0.0], [-0.12], [1]]),
            np.array([[0.01], [0.05], [-0.14], [1]]),
            np.array([[0.05], [0.0], [-0.12], [1]]),
            np.array([[0.05], [0.05], [-0.14], [1]]),
        ]
        psm_warmup_quats = [
            np.array([0.32743764, 0.79021532, 0.43636955, 0.27915223]),
            np.array([0.1098817, 0.54847876, 0.7428689, 0.3677538]),
            np.array([0.06069057, 0.49024802, 0.86942407, 0.00867864]),
            np.array([0.63309931, -0.3134932, -0.31293999, 0.63480379]),
        ]
    elif psm_number == "2":
        print("Not implemented yet")
        import pdb

        pdb.set_trace()
        psm_warmup_points = [
            np.array([[-0.01], [0.0], [-0.1], [1]]),
            np.array([[-0.01], [0.05], [-0.12], [1]]),
            np.array([[-0.05], [0.0], [-0.1], [1]]),
            np.array([[-0.05], [0.05], [-0.12], [1]]),
        ]
        psm_warmup_quats = [
            np.array([0.61942243, 0.34586668, 0.06921689, 0.70135662]),
            np.array([0.69046505, -0.40335585, -0.27076941, 0.53595335]),
            np.array([0.59117077, -0.50339768, -0.40847555, 0.47984958]),
            np.array([-0.0902323, 0.76810494, 0.61100639, 0.16895008]),
        ]
    else:
        print("Please select PSM1 or PSM2")
        exit()
    for p1, q1 in zip(psm_warmup_points, psm_warmup_quats):
        psm1_pos = np.array([p1[0, 0] / p1[3, 0], p1[1, 0] / p1[3, 0], p1[2, 0] / p1[3, 0]])
        psm1_quat = q1
        psm.set_pose(psm1_pos, psm1_quat)
        time.sleep(1)
    psm_pose = np.array([0, 0, -0.13]), np.array([0, 0, 0, 1])
    psm.set_pose(*psm_pose)
    curr_joints = psm.get_current_joint()
    curr_joints[3:] = [0, 0, 0]
    home_joints = np.copy(curr_joints)
    psm.set_joint(joint=curr_joints)
    psm.set_jaw(jaw=JAW_CLOSE_ANGLE)
    return home_joints


root_path = os.path.dirname(os.path.abspath(__file__))

left_P_matrix_fname = os.path.join(
    root_path,
    "experiment/0_trajectory_extraction/allied_vision_calibration_outputs/allied_vision_matrices/left_P_mat.npy",
)
right_P_matrix_fname = os.path.join(
    root_path,
    "experiment/0_trajectory_extraction/allied_vision_calibration_outputs/allied_vision_matrices/right_P_mat.npy",
)
P_av_left = np.load(left_P_matrix_fname)
P_av_right = np.load(right_P_matrix_fname)
D_undistorted = np.zeros((4,), np.float64)

tf_av_left_zivid = RigidTransform.load(
    os.path.join(
        root_path,
        "experiment/0_trajectory_extraction/allied_vision_to_zivid_calibration_outputs/av_zivid_output_matrices/av_left_to_zivid.tf",
    )
)
tf_av_right_zivid = RigidTransform.load(
    os.path.join(
        root_path,
        "experiment/0_trajectory_extraction/allied_vision_to_zivid_calibration_outputs/av_zivid_output_matrices/av_right_to_zivid.tf",
    )
)


psm_number_input = input("Which PSM are you using? (1 or 2)")
tf_zivid_psm = None
psm_points = []

if psm_number_input == "1":
    tf_zivid_psm = np.load(
        os.path.join(
            root_path, "experiment/0_trajectory_extraction/shallow_and_deep_calibration_outputs/psm1_robot_to_zivid.npy"
        )
    )
    psm_points = np.array(
        [
            [0.08, 0.05, -0.14, 1],
            [0.12, 0.05, -0.14, 1],
            [0.08, 0.09, -0.14, 1],
            [0.12, 0.09, -0.14, 1],
            [0.08, 0.05, -0.14, 1],
            [0.12, 0.05, -0.12, 1],
            [0.08, 0.09, -0.12, 1],
            [0.12, 0.09, -0.12, 1],
        ]
    )

elif psm_number_input == "2":
    print("Not implemented yet")
    import pdb

    pdb.set_trace()
else:
    print("Please select 1 or 2")
    exit()
dvrk_type_input = input(
    "Which PSM driver type are you calibrating. Large SutureCut Needle Driver (1) or Large Needle Driver (2)"
)
dvrk_type = None
if dvrk_type_input == "1":
    dvrk_type = dvrkTypes.LRG_SUTURECUT_NEEDLE_DRIVER
elif dvrk_type_input == "2":
    dvrk_type = dvrkTypes.LARGE_NEEDLE_DRIVER
else:
    print("Please select 1 or 2")
    exit()

psm = dvrkArm("/PSM" + psm_number_input, dvrk_type=dvrk_type, use_rnn=True)

home_joints = single_psm_warmup(psm_number=psm_number_input, psm=psm)
points_in_psm_frame = np.array(psm_points)

tf_zivid_psm = np.linalg.inv(tf_zivid_psm)
tf_zivid_psm = RigidTransform(
    rotation=tf_zivid_psm[:3, :3],
    translation=tf_zivid_psm[:3, 3],
    from_frame="psm" + str(psm_number_input),
    to_frame="zivid",
)

# Compute desired transforms
tf_av_left_psm = tf_av_left_zivid * tf_zivid_psm
tf_av_right_psm = tf_av_right_zivid * tf_zivid_psm

points_in_av_left_frame_homogenous = tf_av_left_psm.matrix @ psm_points.T
points_in_av_left_frame_homogenous = points_in_av_left_frame_homogenous.T
points_in_av_left_frame = points_in_av_left_frame_homogenous[:, :3] / points_in_av_left_frame_homogenous[:, 3].reshape(
    -1, 1
)
pixels, _ = cv2.projectPoints(
    points_in_av_left_frame,
    rvec=np.array([0.0, 0.0, 0.0]),
    tvec=np.array([0.0, 0.0, 0.0]),
    cameraMatrix=P_av_left[:, :3],
    distCoeffs=D_undistorted,
)
pixels = np.round(pixels).astype(int).squeeze()
av_util = AlliedVisionUtils()
av_cam = Camera(cst.ALLIED_VISION, zivid_cam_choice="inclined", zivid_capture_type="2d", rectify_img=False)
time.sleep(2)
print(" ")
print("AV Stereo ready!")
psm_point_index = 0
while True:
    img_av_left, img_av_right = av_cam.capture()

    if len(img_av_left) == 0 or len(img_av_right) == 0:
        time.sleep(0.1)
        continue
    img_av_left = av_util.rectify_single(img_av_left, is_left=True)
    img_av_right = av_util.rectify_single(img_av_right, is_left=False)
    psm1_p_robot, _ = psm.get_current_pose()

    if psm_point_index - 1 >= 0:
        img_av_left = cv2.circle(img_av_left, pixels[psm_point_index - 1], 5, (0, 255, 0), -1)
    img_av_left = plot_psm_pos_on_img(img_av_left, psm1_p_robot, tf_av_left_psm, P_av_left[:, :-1], D_undistorted)
    cv2.imshow("End Effector Tracking AV_LEFT. Press q to exit.", img_av_left)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        av_cam.stop()
        break
    elif key == ord(" "):

        psm.set_joint(joint=np.copy(home_joints))
        time.sleep(1)
        if psm_point_index >= len(psm_points):
            av_cam.stop()
            exit()
        psm_point_homogenous = psm_points[psm_point_index]
        psm_point = psm_point_homogenous[:3] / psm_point_homogenous[3]
        psm_rand_quat = random_quaternion()
        psm_pose = psm_point, psm_rand_quat
        psm.set_pose(*psm_pose)
        psm_point_index += 1
