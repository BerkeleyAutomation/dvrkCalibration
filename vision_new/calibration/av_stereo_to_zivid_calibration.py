import os
from typing import Tuple
from datetime import datetime
import time

import numpy as np
import cv2
from autolab_core import RigidTransform
from dvrk.vision.cameras.AlliedVisionUtils import AlliedVisionUtils

from dvrk.vision.cameras.Camera import Camera
import dvrk.vision.vision_constants as cst

CALIB_MATS_DIR_AV_LEFT = "calibration_matrices_av_left"
CALIB_MATS_PATH_AV_LEFT = os.path.join(cst.DATA_PATH, CALIB_MATS_DIR_AV_LEFT)
CALIB_MATS_DIR_AV_RIGHT = "calibration_matrices_av_right"
CALIB_MATS_PATH_AV_RIGHT = os.path.join(cst.DATA_PATH, CALIB_MATS_DIR_AV_RIGHT)
CALIB_MATS_DIR_ZIVID = "calibration_matrices_zivid"
CALIB_MATS_PATH_ZIVID = os.path.join(cst.DATA_PATH, CALIB_MATS_DIR_ZIVID)

CAPTURE_IMGS = True
# N_IMGAGES = 9
N_IMGAGES = 100
IMG_SAVE_DIR = "aruco_imgs_capture_rect_05_25_2024"
# IMG_SAVE_DIR = "aruco_imgs_capture_rect"
IMG_SAVE_PATH = os.path.join(cst.DATA_PATH, IMG_SAVE_DIR)
AV_LEFT_IMG_FNAME = "img_av_left_"
AV_RIGHT_IMG_FNAME = "img_av_right_"
ZIVID_IMG_FNAME = "img_zivid_"


# now = datetime.now()
# TF_AV_LEFT_AV_RIGHT_FNAME = f"tf_av_left_av_right_{now.strftime('%Y-%m-%d_%H-%M-%S')}.tf"
# TF_AV_LEFT_ZIVID_FNAME = f"tf_av_left_zivid_{now.strftime('%Y-%m-%d_%H-%M-%S')}.tf"
# TF_AV_RIGHT_ZIVID_FNAME = f"tf_av_right_zivid_{now.strftime('%Y-%m-%d_%H-%M-%S')}.tf"
TF_AV_LEFT_AV_RIGHT_FNAME = f"tf_av_left_av_right.tf"
TF_AV_LEFT_ZIVID_FNAME = f"tf_av_left_zivid.tf"
TF_AV_RIGHT_ZIVID_FNAME = f"tf_av_right_zivid.tf"
TF_SAVE_PATH = os.path.join(cst.DATA_PATH, "aruco_calib_results_rect_v3")

TAG_SIZE = 0.07946  # 0.0705  edge size in m
ALL_TAG_IDS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}
av_util = AlliedVisionUtils()


def capture_camera_imgs(cam_av: Camera, cam_zivid: Camera) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:

    while True:
        img_left_av, img_right_av = cam_av.capture()
        if len(img_left_av) == 0 or len(img_right_av) == 0:
            time.sleep(0.1)
            continue
        img_left_av = av_util.rectify_single(img_left_av, is_left=True)
        img_right_av = av_util.rectify_single(img_right_av, is_left=False)
        img_stereo_av = np.hstack((img_left_av, img_right_av))
        img_stereo_av = cv2.resize(img_stereo_av, None, fx=0.5, fy=0.5)

        img_zivid = cam_zivid.capture()

        cv2.imshow("cam img: press c to capture img, q to quit", img_zivid)
        cv2.imshow("cam stereo img: press c to capture img, q to quit", img_stereo_av)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            cv2.destroyAllWindows()
            return True, img_left_av, img_right_av, img_zivid
        elif key == ord("q"):
            cv2.destroyAllWindows()
            return False, None, None, None


def rvec_tvec_to_transform(rvec, tvec):
    """
    convert translation and rotation to pose
    """
    if rvec is None or tvec is None:
        return None

    R = cv2.Rodrigues(rvec)[0]
    t = tvec
    return RigidTransform(R, t, from_frame="tag", to_frame="camera")


def detect_marker_corners(frame, aruco_dict_type):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, rejected_img_points = detector.detectMarkers(gray)
    return corners, ids


def get_tag2cam_pose(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients):
    """
    frame - Frame from the video stream
    matrix_coefficients - Intrinsic matrix of the calibrated camera
    distortion_coefficients - Distortion coefficients associated with your camera

    return:-
    frame - The frame with the axis drawn on it
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dictionary = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    corners, ids, rejected_img_points = detector.detectMarkers(gray)

    rvecs, tvecs = [], []
    object_points = np.zeros((4, 3), np.float32)
    object_points[0, :] = np.array([-TAG_SIZE / 2, TAG_SIZE / 2, 0])
    object_points[1, :] = np.array([TAG_SIZE / 2, TAG_SIZE / 2, 0])
    object_points[2, :] = np.array([TAG_SIZE / 2, -TAG_SIZE / 2, 0])
    object_points[3, :] = np.array([-TAG_SIZE / 2, -TAG_SIZE / 2, 0])

    if len(corners) > 0:
        for i in range(0, len(ids)):
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            ret, rvec, tvec = cv2.solvePnP(
                object_points, corners[i], matrix_coefficients, distortion_coefficients, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            rvecs.append(rvec)
            tvecs.append(tvec)
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
    return frame, rvecs, tvecs, ids, corners


# def get_tag_to_camera_tf(
#     image: np.ndarray, camera_intrinsics: np.ndarray, distortion_coefficients: np.ndarray, tag_size: float
# ) -> Tuple[RigidTransform, np.ndarray]:
#     aruco_dict_type = ARUCO_DICT["DICT_ARUCO_ORIGINAL"]
#     output_img, rvec, tvec = pose_estimation(
#         image, aruco_dict_type, camera_intrinsics, distortion_coefficients, tag_size
#     )
#     print(f"cam to aruco center dist is {np.linalg.norm(tvec)} m.")
#     pose = rvec_tvec_to_transform(rvec, tvec)

#     return pose, output_img


def get_tf_ab(R_ac, t_ac, R_bc, t_bc):
    R_ab = R_ac.dot(R_bc.T)
    t_ab = R_ac.dot(-R_bc.T.dot(t_bc)) + t_ac
    return R_ab, t_ab


def main():
    K_av_left = np.load(os.path.join(CALIB_MATS_PATH_AV_LEFT, cst.K_MAT_FNAME))
    D_av_left = np.load(os.path.join(CALIB_MATS_PATH_AV_LEFT, cst.D_MAT_FNAME))
    K_av_right = np.load(os.path.join(CALIB_MATS_PATH_AV_RIGHT, cst.K_MAT_FNAME))
    D_av_right = np.load(os.path.join(CALIB_MATS_PATH_AV_RIGHT, cst.D_MAT_FNAME))
    K_zivid = np.load(os.path.join(CALIB_MATS_PATH_ZIVID, cst.K_MAT_FNAME))
    D_zivid = np.load(os.path.join(CALIB_MATS_PATH_ZIVID, cst.D_MAT_FNAME))
    P_av_left = np.load(os.path.join(CALIB_MATS_PATH_AV_LEFT, cst.P_MAT_FNAME))
    P_av_right = np.load(os.path.join(CALIB_MATS_PATH_AV_RIGHT, cst.P_MAT_FNAME))
    D_undistorted = np.zeros((4,), np.float64)

    img_left_av_all = []
    img_right_av_all = []
    img_zivid_all = []

    if CAPTURE_IMGS:
        os.makedirs(IMG_SAVE_PATH, exist_ok=True)
        av = Camera(cst.ALLIED_VISION, zivid_cam_choice="inclined", zivid_capture_type="2d", rectify_img=False)
        zivid = Camera(cst.ZIVID)
        counter = 0
        while True:
            is_capture, img_left_av, img_right_av, img_zivid = capture_camera_imgs(av, zivid)

            if is_capture:
                img_left_av_all.append(img_left_av)
                img_right_av_all.append(img_right_av)
                img_zivid_all.append(img_zivid)
                cv2.imwrite(
                    os.path.join(IMG_SAVE_PATH, f"{AV_LEFT_IMG_FNAME}{counter:02d}{cst.IMG_EXTENSION}"), img_left_av
                )
                cv2.imwrite(
                    os.path.join(IMG_SAVE_PATH, f"{AV_RIGHT_IMG_FNAME}{counter:02d}{cst.IMG_EXTENSION}"), img_right_av
                )
                cv2.imwrite(
                    os.path.join(IMG_SAVE_PATH, f"{ZIVID_IMG_FNAME}{counter:02d}{cst.IMG_EXTENSION}"), img_zivid
                )
                counter += 1
            else:
                av.stop()
                break
    else:  # load images
        for counter in range(N_IMGAGES):
            img_left_av = cv2.imread(
                os.path.join(IMG_SAVE_PATH, f"{AV_LEFT_IMG_FNAME}{counter:02d}{cst.IMG_EXTENSION}")
            )
            img_right_av = cv2.imread(
                os.path.join(IMG_SAVE_PATH, f"{AV_RIGHT_IMG_FNAME}{counter:02d}{cst.IMG_EXTENSION}")
            )
            img_zivid = cv2.imread(os.path.join(IMG_SAVE_PATH, f"{ZIVID_IMG_FNAME}{counter:02d}{cst.IMG_EXTENSION}"))
            img_left_av_all.append(img_left_av)
            img_right_av_all.append(img_right_av)
            img_zivid_all.append(img_zivid)

    n_imgs = len(img_left_av_all) if CAPTURE_IMGS else N_IMGAGES
    # aruco_dict_type = ARUCO_DICT["DICT_4X4_50"]
    aruco_dict_type = ARUCO_DICT["DICT_7X7_50"]

    for i in range(n_imgs):

        img_left_av = img_left_av_all[i]
        img_right_av = img_right_av_all[i]
        img_zivid = img_zivid_all[i]

        frame_left_av, rvecs_left_av, tvecs_left_av, ids_left_av, corners_left_av = get_tag2cam_pose(
            img_left_av, aruco_dict_type, P_av_left[:, :-1], D_undistorted
        )
        frame_right_av, rvecs_right_av, tvecs_right_av, ids_right_av, corners_right_av = get_tag2cam_pose(
            img_right_av, aruco_dict_type, P_av_right[:, :-1], D_undistorted
        )
        frame_zivid, rvecs_zivid, tvecs_zivid, ids_zivid, corners_zivid = get_tag2cam_pose(
            img_zivid, aruco_dict_type, K_zivid, D_zivid
        )

        #     # cv2.imshow("frame_left_av", frame_left_av)
        #     # cv2.imshow("frame_right_av", frame_right_av)
        #     # cv2.imshow("frame_zivid", frame_zivid)
        #     # cv2.waitKey(0)
        #     # cv2.destroyAllWindows()

        #     # corners_left_av, ids_left_av = detect_marker_corners(img_left_av, aruco_dict_type)
        #     # corners_right_av, ids_right_av = detect_marker_corners(img_right_av, aruco_dict_type)
        #     # corners_zivid, ids_zivid = detect_marker_corners(img_zivid, aruco_dict_type)

        # print(ids_zivid)
        ids_left_av_dict = {id[0]: index for index, id in enumerate(ids_left_av)}
        ids_right_av_dict = {id[0]: index for index, id in enumerate(ids_right_av)}
        ids_zivid_dict = {id[0]: index for index, id in enumerate(ids_zivid)}
        # print(ids_left_av)

        #     #     # ids_all_set = set.intersection(set(ids_left_av), set(ids_right_av), set(ids_zivid))
        #     #     # for id in ids_all_set:

        rvecs_av_left_av_right = []
        rvecs_av_left_zivid = []
        rvecs_av_right_zivid = []
        tvecs_av_left_av_right = []
        tvecs_av_left_zivid = []
        tvecs_av_right_zivid = []

        for id in ALL_TAG_IDS:
            if id in ids_left_av_dict.keys() and id in ids_right_av_dict.keys() and id in ids_zivid_dict.keys():
                print(f"id {id} in all images")
                rvec_av_left_tag = rvecs_left_av[ids_left_av_dict[id]]
                rvec_av_right_tag = rvecs_right_av[ids_right_av_dict[id]]
                rvec_zivid_tag = rvecs_zivid[ids_zivid_dict[id]]
                R_av_left_tag, _ = cv2.Rodrigues(rvec_av_left_tag)
                R_av_right_tag, _ = cv2.Rodrigues(rvec_av_right_tag)
                R_zivid_tag, _ = cv2.Rodrigues(rvec_zivid_tag)

                tvec_av_left_tag = tvecs_left_av[ids_left_av_dict[id]]
                tvec_av_right_tag = tvecs_right_av[ids_right_av_dict[id]]
                tvec_zivid_tag = tvecs_zivid[ids_zivid_dict[id]]

                R_av_left_av_right, tvec_av_left_av_right = get_tf_ab(
                    R_av_left_tag, tvec_av_left_tag, R_av_right_tag, tvec_av_right_tag
                )
                R_av_left_zivid, tvec_av_left_zivid = get_tf_ab(
                    R_av_left_tag, tvec_av_left_tag, R_zivid_tag, tvec_zivid_tag
                )
                R_av_right_zivid, tvec_av_right_zivid = get_tf_ab(
                    R_av_right_tag, tvec_av_right_tag, R_zivid_tag, tvec_zivid_tag
                )

                rvec_av_left_av_right, _ = cv2.Rodrigues(R_av_left_av_right)
                rvec_av_left_zivid, _ = cv2.Rodrigues(R_av_left_zivid)
                rvec_av_right_zivid, _ = cv2.Rodrigues(R_av_right_zivid)

                rvecs_av_left_av_right.append(rvec_av_left_av_right)
                rvecs_av_left_zivid.append(rvec_av_left_zivid)
                rvecs_av_right_zivid.append(rvec_av_right_zivid)
                tvecs_av_left_av_right.append(tvec_av_left_av_right)
                tvecs_av_left_zivid.append(tvec_av_left_zivid)
                tvecs_av_right_zivid.append(tvec_av_right_zivid)

            else:
                print(f"id {id} not in all images. Skipping")

    rvecs_av_left_av_right_mean = np.array(rvecs_av_left_av_right).mean(axis=0)
    rvecs_av_left_zivid_mean = np.array(rvecs_av_left_zivid).mean(axis=0)
    rvecs_av_right_zivid_mean = np.array(rvecs_av_right_zivid).mean(axis=0)
    tvecs_av_left_av_right_mean = np.array(tvecs_av_left_av_right).mean(axis=0)
    tvecs_av_left_zivid_mean = np.array(tvecs_av_left_zivid).mean(axis=0)
    tvecs_av_right_zivid_mean = np.array(tvecs_av_right_zivid).mean(axis=0)

    R_av_left_av_right_mean, _ = cv2.Rodrigues(rvecs_av_left_av_right_mean)
    R_av_left_zivid_mean, _ = cv2.Rodrigues(rvecs_av_left_zivid_mean)
    R_av_right_zivid_mean, _ = cv2.Rodrigues(rvecs_av_right_zivid_mean)

    tf_av_left_av_right = RigidTransform(
        R_av_left_av_right_mean, tvecs_av_left_av_right_mean, from_frame="av_right", to_frame="av_left"
    )
    tf_av_left_zivid = RigidTransform(
        R_av_left_zivid_mean, tvecs_av_left_zivid_mean, from_frame="zivid", to_frame="av_left"
    )
    tf_av_right_zivid = RigidTransform(
        R_av_right_zivid_mean, tvecs_av_right_zivid_mean, from_frame="zivid", to_frame="av_right"
    )

    # print(tf_av_left_av_right)
    # print(tf_av_left_av_right.inverse())
    print(tf_av_left_zivid)
    # print(tf_av_right_zivid)

    os.makedirs(TF_SAVE_PATH, exist_ok=True)
    tf_av_left_av_right.save(os.path.join(TF_SAVE_PATH, TF_AV_LEFT_AV_RIGHT_FNAME))
    tf_av_left_zivid.save(os.path.join(TF_SAVE_PATH, TF_AV_LEFT_ZIVID_FNAME))
    tf_av_right_zivid.save(os.path.join(TF_SAVE_PATH, TF_AV_RIGHT_ZIVID_FNAME))


if __name__ == "__main__":
    main()
