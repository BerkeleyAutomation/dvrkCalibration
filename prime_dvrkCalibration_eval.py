import numpy as np
from dvrk.motion.dvrkArm import dvrkArm
from autolab_core import RigidTransform
from dvrk.motion.dvrkTypes import dvrkTypes
from dvrk.motion.dvrkKinematics import dvrkKinematics
import os
import cv2
import time
from dvrk.vision import vision_constants as cst
from dvrk.vision.cameras.Camera import Camera


def plot_psm_pos_on_img(img, psm_p_robot, tf_cam_psm, K, D):
    rvec_cam_psm, _ = cv2.Rodrigues(tf_cam_psm.rotation)

    psm_p_cam, _ = cv2.projectPoints(psm_p_robot, rvec_cam_psm, tf_cam_psm.translation, K, D)
    psm_p_cam = psm_p_cam.squeeze().astype(int)
    img = cv2.circle(img, (psm_p_cam[0], psm_p_cam[1]), 5, (255, 0, 0), -1)
    return img


joint_traj = np.load(
    "/home/davinci/dvrkCalibration/experiment/0_trajectory_extraction/shallow_and_deep_calibration_outputs/prime_psm1_random_joint_eval_in_distribution.npy"
)

root_path = os.path.dirname(os.path.abspath(__file__))
psm_number_input = input("Which PSM are you using? (1 or 2)")
tf_zivid_psm = np.load(
    os.path.join(
        root_path,
        "experiment/0_trajectory_extraction/shallow_and_deep_calibration_outputs/psm"
        + psm_number_input
        + "_robot_to_zivid.npy",
    )
)
psm = dvrkArm("/PSM" + str(psm_number_input), dvrk_type=dvrkTypes.LRG_SUTURECUT_NEEDLE_DRIVER, use_rnn=True)
pose_lists = [
    dvrkKinematics.joint_to_pose(
        row,
        L1=psm.l_rcc_,
        L2=psm.l_tool_,
        L3=psm.l_pitch_2_yaw_,
        L4=psm.l_yaw_2_ctrl_pnt_,
    )
    for row in joint_traj
]
psm_points = np.array([pose[0] for pose in pose_lists])
points_in_psm_frame = np.array(psm_points)
tf_zivid_psm = np.linalg.inv(tf_zivid_psm)
tf_zivid_psm = RigidTransform(
    rotation=tf_zivid_psm[:3, :3],
    translation=tf_zivid_psm[:3, 3],
    from_frame="psm" + str(psm_number_input),
    to_frame="zivid",
)
psm_points_homogenous = [np.append(position, 1) for position in psm_points]
psm_points_homogenous = np.array(psm_points_homogenous)
points_in_zivid_frame_homogenous = tf_zivid_psm.matrix @ psm_points_homogenous.T
points_in_zivid_frame_homogenous = points_in_zivid_frame_homogenous.T
points_in_zivid_frame = points_in_zivid_frame_homogenous[:, :3] / points_in_zivid_frame_homogenous[:, 3].reshape(-1, 1)
zivid_cam = Camera(cst.ZIVID)
K_zivid = zivid_cam.cam.intrinsics_
D_zivid = zivid_cam.cam.distortion_coefficients_
time.sleep(2)
print(" ")
print("Zivid ready!")
pixels, _ = cv2.projectPoints(
    points_in_zivid_frame,
    rvec=np.array([0.0, 0.0, 0.0]),
    tvec=np.array([0.0, 0.0, 0.0]),
    cameraMatrix=K_zivid,
    distCoeffs=D_zivid,
)
pixels = np.round(pixels).astype(int).squeeze()
for i in range(len(pose_lists)):
    psm.set_pose(*pose_lists[i])
    time.sleep(1)
    img_zivid = zivid_cam.capture()

    psm1_p_robot, _ = psm.get_current_pose()

    img_zivid = cv2.circle(img_zivid, pixels[i], 5, (0, 255, 0), -1)
    img_zivid = plot_psm_pos_on_img(img_zivid, psm1_p_robot, tf_zivid_psm, K_zivid, D_zivid)
    cv2.imshow("End Effector Tracking ZIVID. Press q to exit.", img_zivid)

    key = cv2.waitKey(0) & 0xFF

    if key == ord("q"):
        break
    elif key == ord(" "):
        if i >= len(psm_points):
            exit()
