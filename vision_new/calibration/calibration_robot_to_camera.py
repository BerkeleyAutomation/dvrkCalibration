import os

from datetime import datetime
import numpy as np
import cv2
import time

from dvrk.vision import vision_constants as cst
from dvrk.vision.cameras.Camera import Camera
from dvrk.motion.dvrkArm import dvrkArm
from autolab_core import RigidTransform
from dvrk.vision.cameras.AlliedVisionUtils import AlliedVisionUtils


CALIB_MATS_DIR_ZIVID = "calibration_matrices_zivid"
CALIB_MATS_PATH_ZIVID = os.path.join(cst.DATA_PATH, CALIB_MATS_DIR_ZIVID)
CALIB_MATS_DIR_AV_LEFT = "calibration_matrices_av_left"
CALIB_MATS_PATH_AV_LEFT = os.path.join(cst.DATA_PATH, CALIB_MATS_DIR_AV_LEFT)
CALIB_MATS_DIR_AV_RIGHT = "calibration_matrices_av_right"
CALIB_MATS_PATH_AV_RIGHT = os.path.join(cst.DATA_PATH, CALIB_MATS_DIR_AV_RIGHT)

POSE_SAVE_DIR = "/home/davinci/dvrk/data/aruco_calib_results_rect_v3"  # os.path.join(cst.DATA_PATH, "aruco_calib_results_rect_10_5_2023")
TF_ZIVID_PSM1 = "tf_zivid_psm1_old.tf"
TF_ZIVID_PSM2 = "tf_zivid_psm2.tf"
TF_AV_LEFT_PSM1 = "tf_av_left_psm1.tf"
TF_AV_LEFT_PSM2 = "tf_av_left_psm2.tf"
TF_AV_RIGHT_PSM1 = "tf_av_right_psm1.tf"
TF_AV_RIGHT_PSM2 = "tf_av_right_psm2.tf"
TF_AV_LEFT_ZIVID_FNAME = f"tf_av_left_zivid.tf"
TF_AV_RIGHT_ZIVID_FNAME = f"tf_av_right_zivid.tf"


def np_to_rigid_transform(np_tf_mat: np.ndarray, from_frame: str, to_frame: str) -> RigidTransform:
    rigid_transform = RigidTransform(
        rotation=np_tf_mat[0:3, 0:3], translation=np_tf_mat[0:3, 3], from_frame=from_frame, to_frame=to_frame
    )
    return rigid_transform


def project_robot_pt_to_img(current_end_effector_pos, robot2cam_tf, k_mat):
    current_end_effector_pos = np.append(current_end_effector_pos, 1.0)
    current_end_effector_pos = np.expand_dims(current_end_effector_pos, axis=1)
    current_end_effector_pos_in_cam_frame = np.matmul(robot2cam_tf.matrix, current_end_effector_pos)
    end_effector_cam_point_hom = np.matmul(k_mat, current_end_effector_pos_in_cam_frame[:-1, :])
    end_effector_cam_point_x = int(end_effector_cam_point_hom[0] / end_effector_cam_point_hom[2] + 0.5)
    end_effector_cam_point_y = int(end_effector_cam_point_hom[1] / end_effector_cam_point_hom[2] + 0.5)
    return end_effector_cam_point_x, end_effector_cam_point_y


def plot_psm_pos_on_img(img, psm_p_robot, tf_cam_psm, K, D):
    rvec_cam_psm, _ = cv2.Rodrigues(tf_cam_psm.rotation)

    psm_p_cam, _ = cv2.projectPoints(psm_p_robot, rvec_cam_psm, tf_cam_psm.translation, K, D)
    psm_p_cam = psm_p_cam.squeeze().astype(int)
    img = cv2.circle(img, (psm_p_cam[0], psm_p_cam[1]), 5, (0, 255, 0), -1)
    return img


def main():
    # Load calibration parameters
    K_zivid = np.load(os.path.join(CALIB_MATS_PATH_ZIVID, cst.K_MAT_FNAME))
    D_zivid = np.load(os.path.join(CALIB_MATS_PATH_ZIVID, cst.D_MAT_FNAME))
    K_av_left = np.load(os.path.join(CALIB_MATS_PATH_AV_LEFT, cst.K_MAT_FNAME))
    D_av_left = np.load(os.path.join(CALIB_MATS_PATH_AV_LEFT, cst.D_MAT_FNAME))
    K_av_right = np.load(os.path.join(CALIB_MATS_PATH_AV_RIGHT, cst.K_MAT_FNAME))
    D_av_right = np.load(os.path.join(CALIB_MATS_PATH_AV_RIGHT, cst.D_MAT_FNAME))

    P_av_left = np.load(os.path.join(CALIB_MATS_PATH_AV_LEFT, cst.P_MAT_FNAME))
    P_av_right = np.load(os.path.join(CALIB_MATS_PATH_AV_RIGHT, cst.P_MAT_FNAME))
    D_undistorted = np.zeros((4,), np.float64)

    # Load precomputed rigid transforms
    # tf_zivid_psm1 = RigidTransform.load(os.path.join(POSE_SAVE_DIR, TF_ZIVID_PSM1))
    tf_zivid_psm1 = np.load("/home/davinci/Trc_inclined_PSM1.npy")
    tf_zivid_psm1 = np.linalg.inv(tf_zivid_psm1)
    tf_zivid_psm1 = RigidTransform(
        rotation=tf_zivid_psm1[:3, :3], translation=tf_zivid_psm1[:3, 3], from_frame="psm1", to_frame="zivid"
    )
    tf_zivid_psm2 = np.load("/home/davinci/Trc_inclined_PSM2.npy")
    tf_zivid_psm2 = np.linalg.inv(tf_zivid_psm2)
    tf_zivid_psm2 = RigidTransform(
        rotation=tf_zivid_psm2[:3, :3], translation=tf_zivid_psm2[:3, 3], from_frame="psm2", to_frame="zivid"
    )

    # tf_zivid_psm2 = RigidTransform.load(os.path.join(POSE_SAVE_DIR, TF_ZIVID_PSM2))
    tf_av_left_zivid = RigidTransform.load(os.path.join(POSE_SAVE_DIR, TF_AV_LEFT_ZIVID_FNAME))
    tf_av_right_zivid = RigidTransform.load(os.path.join(POSE_SAVE_DIR, TF_AV_RIGHT_ZIVID_FNAME))

    # Compute desired transforms
    tf_av_left_psm1 = tf_av_left_zivid * tf_zivid_psm1
    tf_av_right_psm1 = tf_av_right_zivid * tf_zivid_psm1
    tf_av_left_psm2 = tf_av_left_zivid * tf_zivid_psm2
    tf_av_right_psm2 = tf_av_right_zivid * tf_zivid_psm2

    # Prepare hardware
    zivid_cam = Camera(cst.ZIVID)
    # av_cam = Camera(cst.ALLIED_VISION, rectify_img=True)
    av_util = AlliedVisionUtils()
    av_cam = Camera(cst.ALLIED_VISION, zivid_cam_choice="inclined", zivid_capture_type="2d", rectify_img=False)
    psm1 = dvrkArm("/PSM1")
    psm2 = dvrkArm("/PSM2")
    print("HERE")
    while True:  # loop to show green poin on gripper wrist live
        psm1_p_robot, _ = psm1.get_current_pose()
        psm2_p_robot, _ = psm2.get_current_pose()

        img_zivid = zivid_cam.capture()
        img_zivid = plot_psm_pos_on_img(img_zivid, psm1_p_robot, tf_zivid_psm1, K_zivid, D_zivid)
        img_zivid = plot_psm_pos_on_img(img_zivid, psm2_p_robot, tf_zivid_psm2, K_zivid, D_zivid)
        cv2.imshow("End Effector Tracking ZIVID. Press q to exit.", img_zivid)

        img_av_left, img_av_right = av_cam.capture()

        if len(img_av_left) == 0 or len(img_av_right) == 0:
            time.sleep(0.1)
            continue
        img_av_left = av_util.rectify_single(img_av_left, is_left=True)
        img_av_right = av_util.rectify_single(img_av_right, is_left=False)
        # left_k = np.load(os.path.join('left_calibration_2024_05_25', cst.K_MAT_FNAME))
        # left_d = np.load(os.path.join('left_calibration_2024_05_25', cst.D_MAT_FNAME))
        # right_k = np.load(os.path.join('right_calibration_2024_05_25', cst.K_MAT_FNAME))
        # right_d = np.load(os.path.join('right_calibration_2024_05_25', cst.D_MAT_FNAME))
        img_av_left = plot_psm_pos_on_img(img_av_left, psm1_p_robot, tf_av_left_psm1, P_av_left[:, :-1], D_undistorted)
        img_av_left = plot_psm_pos_on_img(img_av_left, psm2_p_robot, tf_av_left_psm2, P_av_left[:, :-1], D_undistorted)
        img_av_right = plot_psm_pos_on_img(
            img_av_right, psm1_p_robot, tf_av_right_psm1, P_av_right[:, :-1], D_undistorted
        )
        img_av_right = plot_psm_pos_on_img(
            img_av_right, psm2_p_robot, tf_av_right_psm2, P_av_right[:, :-1], D_undistorted
        )
        img_stereo = np.hstack((img_av_left, img_av_right))
        img_stereo = cv2.resize(img_stereo, None, fx=0.5, fy=0.5)
        cv2.imshow("End Effector Tracking AV_RIGHT. Press q to exit.", img_stereo)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            av_cam.stop()
            break

    tf_av_left_psm1.save(os.path.join(POSE_SAVE_DIR, TF_AV_LEFT_PSM1))
    tf_av_left_psm2.save(os.path.join(POSE_SAVE_DIR, TF_AV_LEFT_PSM2))
    tf_av_right_psm1.save(os.path.join(POSE_SAVE_DIR, TF_AV_RIGHT_PSM1))
    tf_av_right_psm2.save(os.path.join(POSE_SAVE_DIR, TF_AV_RIGHT_PSM2))


if __name__ == "__main__":
    main()
