import os
from tqdm import tqdm

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

import dvrk.utils.CmnUtil as utils
import dvrk.vision.vision_constants as cst


N_IMAGES = 30
N_CHESSBOARD_COLS = 7
N_CHESSBOARD_ROWS = 5
GRID_SIZE = 13.59 * 0.001  # in mm
IMG_SHAPE = (1280, 960)  # (width, height) av, zivid(1200, 1920)

CALIB_IMGS_DIR_LEFT = "left_calibration_10_5_2023"
CALIB_IMGS_PATH_LEFT = os.path.join(cst.DATA_PATH, CALIB_IMGS_DIR_LEFT)
CALIB_IMGS_DIR_RIGHT = "right_calibration_10_5_2023"
CALIB_IMGS_PATH_RIGHT = os.path.join(cst.DATA_PATH, CALIB_IMGS_DIR_RIGHT)
CALIB_MATS_DIR_LEFT = "left_calibration_matrices_05_25_2024"
CALIB_MATS_PATH_LEFT = os.path.join(cst.DATA_PATH, CALIB_MATS_DIR_LEFT)
CALIB_MATS_DIR_RIGHT = "right_calibration_matrices_05_25_2024"
CALIB_MATS_PATH_RIGHT = os.path.join(cst.DATA_PATH, CALIB_MATS_DIR_RIGHT)

IMG_IDX_TO_SKIP = []


def collect_checkerboard_corners(row, col, nb_image, grid_size):
    # prepare object points, like ((0,0,0), (1,0,0), (2,0,0) ....,(row,col,0))*grid_size
    objp = np.zeros((row * col, 3), np.float32)
    objp[:, :2] = np.mgrid[0:col, 0:row].T.reshape(-1, 2)
    objp = objp * grid_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints1 = []  # 2d points in image plane.
    imgpoints2 = []
    for i in tqdm(range(nb_image)):
        if i in IMG_IDX_TO_SKIP:
            continue
        img_fname = cst.IMG_PREFIX + f"{i:03d}" + cst.IMG_EXTENSION
        img_left_fpath = os.path.join(CALIB_IMGS_PATH_LEFT, img_fname)
        img_right_fpath = os.path.join(CALIB_IMGS_PATH_RIGHT, img_fname)
        img_left = cv2.imread(img_left_fpath)
        img_right = cv2.imread(img_right_fpath)
        if img_left is None or img_right is None or len(img_left) == 0 or len(img_right) == 0:
            pass
        else:
            gray1 = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret1, corners1 = cv2.findChessboardCorners(
                gray1, (col, row), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS
            )
            ret2, corners2 = cv2.findChessboardCorners(
                gray2, (col, row), flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS
            )

            if ret1 and ret2:
                # If found, add object points, image points (after refining them)
                # cv2.drawChessboardCorners(img_left, (col, row), corners1, ret1)
                # cv2.drawChessboardCorners(img_right, (col, row), corners2, ret2)
                # img_stereo = np.hstack((img_left, img_right))
                # img_stereo = cv2.resize(img_stereo, None, fx=0.5, fy=0.5)
                # cv2.imshow("img stereo", img_stereo)
                # cv2.waitKey(0)
                # termination criteria
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_ref1 = cv2.cornerSubPix(gray1, corners1, (5, 5), (-1, -1), criteria)
                corners_ref2 = cv2.cornerSubPix(gray2, corners2, (5, 5), (-1, -1), criteria)
                objpoints.append(objp)
                imgpoints1.append(corners_ref1)
                imgpoints2.append(corners_ref2)
            else:
                print(f"img pair {i} was discarded.")
    # cv2.destroyAllWindows()
    return objpoints, imgpoints1, imgpoints2


def collect_trans_rot_vectors(objpoints, imgpoints1, imgpoints2, K, D):
    # obtain R, T matrices from left cam to right cam
    Rs = []
    Ts = []
    for objp, corners_ref1, corners_ref2 in zip(objpoints, imgpoints1, imgpoints2):
        # Transformation from model to the detected
        _, rvecs1, tvecs1, _ = cv2.solvePnPRansac(objp, corners_ref1, K[0], D[0])
        _, rvecs2, tvecs2, _ = cv2.solvePnPRansac(objp, corners_ref2, K[1], D[1])
        tc1m = tvecs1
        tc2m = tvecs2
        Rc1m = cv2.Rodrigues(rvecs1)[0]
        Rc2m = cv2.Rodrigues(rvecs2)[0]

        # Transformation matrix from cam 1 to cam 2
        Rc1c2 = Rc1m.dot(Rc2m.T)
        tc1c2 = Rc1m.dot(-Rc2m.T.dot(tc2m)) + tc1m
        Rs.append(Rc1c2)
        Ts.append(np.squeeze(tc1c2))
    Rs = np.array(Rs)
    Ts = np.array(Ts)
    return Rs, Ts


def plot_trans_rot_vectors(Rs, Ts):
    Eulers = np.array([utils.R_to_euler(R) for R in Rs])
    Ts = np.array(Ts)

    fig = plt.figure()
    ax = fig.add_subplot(211, projection="3d")  # translation
    ax.scatter(Ts[:, 0], Ts[:, 1], Ts[:, 2])
    ax.set_xlabel("X Label (m)")
    ax.set_ylabel("Y Label (m)")
    ax.set_zlabel("Z Label (m)")

    ax2 = fig.add_subplot(212, projection="3d")  # rotation
    Eulers_deg = np.rad2deg(Eulers)  # (deg)
    ax2.scatter(Eulers_deg[:, 0], Eulers_deg[:, 1], Eulers_deg[:, 2])
    ax2.set_xlabel("X Label (deg)")
    ax2.set_ylabel("Y Label (deg)")
    ax2.set_zlabel("Z Label (deg)")
    plt.show()


def remove_outliers(Rs, Ts):
    Eulers = np.array([utils.R_to_euler(R) for R in Rs])
    Ts = np.array(Ts)

    # for translation
    clustering = DBSCAN(eps=0.3, min_samples=2).fit(Ts)
    index = clustering.labels_

    # for rotation
    clustering2 = DBSCAN(eps=0.7, min_samples=2).fit(Eulers)
    index2 = clustering2.labels_

    # cross sectional index
    index_cross = index + index2
    Ts = Ts[index_cross == 0]
    T = np.average(Ts, axis=0)
    Euler = Eulers[index_cross == 0]
    Euler = np.average(Euler, axis=0)
    R = utils.euler_to_R(Euler)
    return R, T


def main():
    # collect checkerboard corners
    objpoints, imgpoints1, imgpoints2 = collect_checkerboard_corners(
        row=N_CHESSBOARD_ROWS, col=N_CHESSBOARD_COLS, nb_image=N_IMAGES, grid_size=GRID_SIZE
    )

    # get camera intrinsics
    K_left = np.load(os.path.join(CALIB_MATS_PATH_LEFT, "K_mat.npy"))
    K_right = np.load(os.path.join(CALIB_MATS_PATH_RIGHT, "K_mat.npy"))
    D_left = np.load(os.path.join(CALIB_MATS_PATH_LEFT, "D_mat.npy"))
    D_right = np.load(os.path.join(CALIB_MATS_PATH_RIGHT, "D_mat.npy"))

    #######################################
    ##  Remove outliers (Manually delete files)
    #######################################

    # collect trans & rot vectors between two cameras (from right to left)
    # Rs, Ts = collect_trans_rot_vectors(objpoints, imgpoints1, imgpoints2, K_left, D_left)

    # plot trans & rot vectors collected
    # plot_trans_rot_vectors(Rs, Ts)

    # remove outliers
    # R, T = remove_outliers(Rs, Ts)

    #######################################
    ##   Stereo calibration
    #######################################
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    # flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    # flags |= cv2.CALIB_FIX_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_ASPECT_RATIO
    # flags |= cv2.CALIB_ZERO_TANGENT_DIST
    # flags |= cv2.CALIB_RATIONAL_MODEL
    # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
    # flags |= cv2.CALIB_FIX_K3
    # flags |= cv2.CALIB_FIX_K4
    # flags |= cv2.CALIB_FIX_K5
    stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-5)
    # ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
    #     objpoints, imgpoints1, imgpoints2, K_left, D_left, K_right, D_right, IMG_SHAPE
    # )
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints1,
        imgpoints2,
        K_left,
        D_left,
        K_right,
        D_right,
        IMG_SHAPE,
        criteria=stereocalib_criteria,
        flags=flags,
    )

    print("")
    print("RMS error of extrinsic calibration: ", ret)
    # print("New camera intrinsics")
    # print("K_left new=", K_left_new)
    # print("K_left=", K_left)
    # print("D_left new=", D_left_new)
    # print("D_left=", D_left)

    #######################################
    ##   Rectification
    #######################################
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K_left, D_left, K_right, D_right, IMG_SHAPE, R, T, alpha=0)
    mapx1, mapy1 = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, IMG_SHAPE, cv2.CV_32F)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, IMG_SHAPE, cv2.CV_32F)

    # print results
    print("Stereo calibration results")
    print("R=", R)
    Euler = utils.R_to_euler(R)
    print("Euler=", np.rad2deg(Euler), "(deg)")
    print("T=", np.array(T), "(m)")
    print("E=", E)
    print("F=", F)
    print("")
    print("Rectify matrices")
    print("R1=", R1)
    print("R2=", R2)
    print("P1=", P1)
    print("P2=", P2)
    print("Q=", Q)
    print("mapx1=", mapx1)
    print("mapy1=", mapy1)
    print("mapx2=", mapx2)
    print("mapy2=", mapy2)

    #######################################
    ##   Save matrices
    #######################################
    os.makedirs(CALIB_MATS_PATH_LEFT, exist_ok=True)
    os.makedirs(CALIB_MATS_PATH_RIGHT, exist_ok=True)

    # intrinsics
    np.save(os.path.join(CALIB_MATS_PATH_LEFT, cst.K_MAT_FNAME), K_left)
    np.save(os.path.join(CALIB_MATS_PATH_LEFT, cst.D_MAT_FNAME), D_left)
    np.save(os.path.join(CALIB_MATS_PATH_RIGHT, cst.K_MAT_FNAME), K_right)
    np.save(os.path.join(CALIB_MATS_PATH_RIGHT, cst.D_MAT_FNAME), D_right)

    # extrinsics
    np.save(os.path.join(CALIB_MATS_PATH_LEFT, cst.R_STEREO_MAT_FNAME), R)
    np.save(os.path.join(CALIB_MATS_PATH_LEFT, cst.T_STEREO_MAT_FNAME), T)
    np.save(os.path.join(CALIB_MATS_PATH_LEFT, cst.E_STEREO_MAT_FNAME), E)
    np.save(os.path.join(CALIB_MATS_PATH_LEFT, cst.F_STEREO_MAT_FNAME), F)
    np.save(os.path.join(CALIB_MATS_PATH_LEFT, cst.Q_STEREO_MAT_FNAME), Q)

    # rectification
    np.save(os.path.join(CALIB_MATS_PATH_LEFT, cst.R_MAT_FNAME), R1)
    np.save(os.path.join(CALIB_MATS_PATH_RIGHT, cst.R_MAT_FNAME), R2)
    np.save(os.path.join(CALIB_MATS_PATH_LEFT, cst.P_MAT_FNAME), P1)
    np.save(os.path.join(CALIB_MATS_PATH_RIGHT, cst.P_MAT_FNAME), P2)
    np.save(os.path.join(CALIB_MATS_PATH_LEFT, cst.MAPX_MAT_FNAME), mapx1)
    np.save(os.path.join(CALIB_MATS_PATH_LEFT, cst.MAPY_MAT_FNAME), mapy1)
    np.save(os.path.join(CALIB_MATS_PATH_RIGHT, cst.MAPX_MAT_FNAME), mapx2)
    np.save(os.path.join(CALIB_MATS_PATH_RIGHT, cst.MAPY_MAT_FNAME), mapy2)
    print("saved!")


if __name__ == "__main__":
    main()
