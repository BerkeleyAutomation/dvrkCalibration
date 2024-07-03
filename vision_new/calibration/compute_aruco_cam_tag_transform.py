"""
Sample Usage:-
python pose_estimation.py --K_Matrix calibration_matrix.npy --D_Coeff distortion_coefficients.npy --type DICT_5X5_100
"""
import os
from typing import Tuple
import numpy as np
import cv2
from datetime import datetime
from autolab_core import RigidTransform
import dvrk.vision.vision_constants as cst

IMG_FPATH = "/home/davinci/dvrk/data/aruco_imgs_zivid/img_left_000.png"
CALIB_MATS_DIR = "calibration_matrices_zivid"
CALIB_MATS_PATH = os.path.join(cst.DATA_PATH, CALIB_MATS_DIR)

now = datetime.now()
POSE_SAVE_FNAME = f"pose_tag2zivid_{now.strftime('%Y-%m-%d_%H-%M-%S')}.tf"
POSE_SAVE_DIR = os.path.join(cst.DATA_PATH, "aruco_calib_results")

TAG_SIZE = 0.0705  # edge size in m

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


def rvec_tvec_to_transform(rvec, tvec):
    """
    convert translation and rotation to pose
    """
    if rvec is None or tvec is None:
        return None

    R = cv2.Rodrigues(rvec)[0]
    t = tvec
    return RigidTransform(R, t, from_frame="tag", to_frame="camera")


def pose_estimation(frame, aruco_dict_type, matrix_coefficients, distortion_coefficients, tag_size):
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

    rvec, tvec = None, None
    object_points = np.zeros((4, 3), np.float32)
    object_points[0, :] = np.array([-TAG_SIZE / 2, TAG_SIZE / 2, 0])
    object_points[1, :] = np.array([TAG_SIZE / 2, TAG_SIZE / 2, 0])
    object_points[2, :] = np.array([TAG_SIZE / 2, -TAG_SIZE / 2, 0])
    object_points[3, :] = np.array([-TAG_SIZE / 2, -TAG_SIZE / 2, 0])

    if len(corners) > 0:
        for i in range(0, len(ids)):
            print(f"found {len(corners[i])} corners")
            # Estimate pose of each marker and return the values rvec and tvec---(different from those of camera coefficients)
            print(corners[i])
            ret, rvec, tvec = cv2.solvePnP(
                object_points, corners[i], matrix_coefficients, distortion_coefficients, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            # Draw a square around the markers
            cv2.aruco.drawDetectedMarkers(frame, corners)

            # Draw Axis
            cv2.drawFrameAxes(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)

    return frame, rvec, tvec


def get_tag_to_camera_tf(
    image: np.ndarray, camera_intrinsics: np.ndarray, distortion_coefficients: np.ndarray, tag_size: float
) -> Tuple[RigidTransform, np.ndarray]:
    aruco_dict_type = ARUCO_DICT["DICT_ARUCO_ORIGINAL"]
    output_img, rvec, tvec = pose_estimation(
        image, aruco_dict_type, camera_intrinsics, distortion_coefficients, tag_size
    )
    print(f"cam to aruco center dist is {np.linalg.norm(tvec)} m.")
    pose = rvec_tvec_to_transform(rvec, tvec)

    return pose, output_img


def main():
    img = cv2.imread(IMG_FPATH)
    K = np.load(os.path.join(CALIB_MATS_PATH, cst.K_MAT_FNAME))
    D = np.load(os.path.join(CALIB_MATS_PATH, cst.D_MAT_FNAME))
    pose, output_img = get_tag_to_camera_tf(img, K, D, TAG_SIZE)

    os.makedirs(POSE_SAVE_DIR, exist_ok=True)
    pose_save_fpath = os.path.join(POSE_SAVE_DIR, POSE_SAVE_FNAME)
    pose.save(pose_save_fpath)
    print(pose)

    cv2.imshow("Estimated Pose. Press q to exit", output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
