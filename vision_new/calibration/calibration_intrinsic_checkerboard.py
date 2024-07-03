import os
from tqdm import tqdm

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

import dvrk.vision.vision_constants as cst


N_IMAGES = 30
N_CHESSBOARD_COLS = 6
N_CHESSBOARD_ROWS = 4
GRID_SIZE = 12.7 * 0.001  # in mm
IMG_SHAPE = (1280, 960)  # (width, height) av(1280, 960), zivid(1920, 1200)

CALIB_IMGS_DIR = "left_calibration_2024_05_25"
CALIB_IMGS_PATH = os.path.join(cst.DATA_PATH, CALIB_IMGS_DIR)
CALIB_MATS_DIR = "left_calibration_matrices_2024_05_25"
CALIB_MATS_PATH = os.path.join(cst.DATA_PATH, CALIB_MATS_DIR)


def collect_checkerboard_corners(row, col, grid_size):
    # prepare object points, like ((0,0,0), (1,0,0), (2,0,0) ....,(row,col,0))*grid_size
    objp = np.zeros((row * col, 3), np.float32)
    objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2)
    objp = objp * grid_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    for i in tqdm(range(N_IMAGES)):
        img_fname = cst.IMG_PREFIX + f"{i:03d}" + cst.IMG_EXTENSION
        img_fpath = os.path.join(CALIB_IMGS_PATH, img_fname)
        img = cv2.imread(img_fpath)
        if img is None or len(img) == 0:
            pass
        else:
            assert img.shape[:-1][::-1] == IMG_SHAPE  # see Opencv calibration tutorial
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(
                gray, (col, row), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS
            )

            if ret:
                # If found, add object points, image points (after refining them)
                # termination criteria
                # cv2.drawChessboardCorners(img, (col, row), corners, ret)
                # cv2.imshow("img left", img)
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_ref = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints.append(corners_ref)
            else:
                print(f"img pair {i} was discarded.")
    return objpoints, imgpoints


def check_calib_visually(K, D):
    for i in tqdm(range(N_IMAGES)):
        img_fname = cst.IMG_PREFIX + f"{i:03d}" + cst.IMG_EXTENSION
        img_fpath = os.path.join(CALIB_IMGS_PATH, img_fname)
        img = cv2.imread(img_fpath)
        undistorted = cv2.undistort(img, K, D)

        ret1, corners1 = cv2.findChessboardCorners(
            undistorted,
            (N_CHESSBOARD_COLS, N_CHESSBOARD_ROWS),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS,
        )

        if ret1:
            point_a = tuple(corners1[0].astype(int).squeeze().tolist())
            point_b = tuple(corners1[6].astype(int).squeeze().tolist())
            point_c = tuple(corners1[28].astype(int).squeeze().tolist())
            point_d = tuple(corners1[34].astype(int).squeeze().tolist())

            # Draw a grid on the undistorted image
            cv2.line(undistorted, point_a, point_b, (0, 0, 255), thickness=3)
            cv2.line(undistorted, point_b, point_d, (0, 0, 255), thickness=3)
            cv2.line(undistorted, point_c, point_a, (0, 0, 255), thickness=3)
            cv2.line(undistorted, point_d, point_c, (0, 0, 255), thickness=3)

        cv2.imshow("undistorted", undistorted)
        cv2.waitKey(0)
    cv2.destroyAllWindows()


def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, D):
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    total_error = mean_error / len(objpoints)
    print(f"total error: {total_error} pixels")
    return total_error


def main():
    # collect checkerboard corners
    objpoints, imgpoints = collect_checkerboard_corners(
        row=N_CHESSBOARD_ROWS, col=N_CHESSBOARD_COLS, grid_size=GRID_SIZE
    )
    # get camera intrinsics
    D = np.zeros((5))
    K = np.zeros((3, 3))
    flag = 0
    flag |= cv2.CALIB_FIX_K4
    flag |= cv2.CALIB_FIX_K5
    ret, K, D, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, IMG_SHAPE, None, None, flags=flag)
    print("")
    print("RMS errors of calibration: ", ret)
    print("Camera intrinsics")
    print("K=", K)
    print("D=", D)
    # print("rvecs=", rvecs)
    # print("tvecs=", tvecs)
    compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, D)
    # check_calib_visually(K, D)

    #######################################
    ##   Save matrices
    #######################################
    # intrinsics
    os.makedirs(CALIB_MATS_PATH, exist_ok=True)

    np.save(os.path.join(CALIB_MATS_PATH, cst.K_MAT_FNAME), K)
    np.save(os.path.join(CALIB_MATS_PATH, cst.D_MAT_FNAME), D)
    # np.save(os.path.join(CALIB_MATS_PATH, cst.r_VEC_FNAME), rvecs)
    # np.save(os.path.join(CALIB_MATS_PATH, cst.t_VEC_FNAME), tvecs)
    # print("saved!")

    objpoints = np.array(objpoints).reshape(-1, 1, 3)
    imgpoints = np.array(imgpoints).reshape(-1, 1, 2)
    ret_left, rvec_left, tvec_left = cv2.solvePnP(objpoints, imgpoints, K, D)
    rotation_left, _ = cv2.Rodrigues(rvec_left)
    print(rotation_left)
    print(tvec_left)


if __name__ == "__main__":
    main()
