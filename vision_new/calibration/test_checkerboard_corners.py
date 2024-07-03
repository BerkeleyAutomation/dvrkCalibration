# import os
# from typing import Tuple
# import time

# import numpy as np
# import cv2

# from dvrk.vision.cameras.Camera import Camera
# import dvrk.vision.vision_constants as cst

# IMGS_LEFT_DIR = "left_calibration_10_5_2023"  # use for mono cameras
# IMGS_RIGHT_DIR = "right_calibration_10_5_2023"
# DEPTH_ARRAY_DIR = "zivid_calibration_10_5_2023"
# IMGS_LEFT_PATH = os.path.join(cst.DATA_PATH, IMGS_LEFT_DIR)
# IMGS_RIGHT_PATH = os.path.join(cst.DATA_PATH, IMGS_RIGHT_DIR)
# DEPTH_ARRAY_PATH = os.path.join(cst.DATA_PATH, DEPTH_ARRAY_DIR)
# IMG_COUNT_START = 0

# CAMERA_TYPE = cst.ALLIED_VISION
# CAMERA_TYPE = cst.ZIVID

import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)
# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

img = cv.imread("img_017.png")
# img = cv.imread("test_img.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
ret, corners = cv.findChessboardCorners(gray, (7, 5), None)
# cv.imshow("img", gray)
# cv.waitKey(50000)
# print(ret)

# If found, add object points, image points (after refining them)
if ret == True:
    objpoints.append(objp)
    corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    imgpoints.append(corners2)
    # Draw and display the corners
    cv.drawChessboardCorners(img, (7, 5), corners2, ret)
    cv.imshow("img", img)
    cv.waitKey(50000)
cv.destroyAllWindows()


# def capture_camera_imgs(cam: Camera) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
#     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#     # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
#     objp = np.zeros((5 * 7, 3), np.float32)
#     objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)
#     # Arrays to store object points and image points from all the images.
#     objpoints = []  # 3d point in real world space
#     imgpoints = []  # 2d points in image plane.
#     while True:
#         if cam.is_stereo():
#             img_left, img_right = cam.capture()
#             pcl = None
#             if len(img_left) == 0 or len(img_right) == 0:
#                 time.sleep(0.1)
#                 continue
#             img_stereo = np.hstack((img_left, img_right))
#             img_stereo = cv2.resize(img_stereo, None, fx=0.5, fy=0.5)

#             gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
#             # Find the chess board corners
#             ret_left, corners_left = cv2.findChessboardCorners(gray_left, (7, 5), None)
#             # If found, add object points, image points (after refining them)
#             if ret_left == True:
#                 objpoints.append(objp)
#                 corners2_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
#                 imgpoints.append(corners2_left)
#                 # Draw and display the corners
#                 cv2.drawChessboardCorners(img_left, (7, 5), corners2_left, ret_left)
#             cv2.imshow("cam stereo img: press c to capture img, q to quit", img_left)

#         elif cam.has_depth():
#             img_left, pcl = cam.capture()
#             img_right = None
#             if len(img_left) == 0:
#                 time.sleep(0.1)
#                 continue
#             cv2.imshow("cam stereo img: press c to capture img, q to quit", img_left)
#         else:
#             img_left, img_right = cam.capture()
#             pcl = None
#             if len(img_left) == 0:
#                 time.sleep(0.1)
#                 continue
#             cv2.imshow("cam img: press c to capture img, q to quit", img_left)
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord("c"):
#             cv2.destroyAllWindows()
#             return True, img_left, img_right, pcl
#         elif key == ord("q"):
#             cv2.destroyAllWindows()
#             return False, None, None, None


# def main():
#     # define camera instance
#     cam = Camera(CAMERA_TYPE, zivid_cam_choice="inclined", zivid_capture_type="2d", rectify_img=False)

#     os.makedirs(IMGS_LEFT_PATH, exist_ok=True)
#     if cam.is_stereo():
#         os.makedirs(IMGS_RIGHT_PATH, exist_ok=True)
#     if cam.has_depth():
#         os.makedirs(DEPTH_ARRAY_PATH, exist_ok=True)

#     # capturing loop
#     cnt = IMG_COUNT_START
#     while True:
#         is_capture, img_left, img_right, pcl_array = capture_camera_imgs(cam)
#         if is_capture:
#             img_left_fname = cst.IMG_PREFIX + f"{cnt:03d}" + cst.IMG_EXTENSION
#             img_left_fpath = os.path.join(IMGS_LEFT_PATH, img_left_fname)
#             cv2.imwrite(img_left_fpath, img_left)

#             if cam.is_stereo():
#                 img_right_fname = cst.IMG_PREFIX + f"{cnt:03d}" + cst.IMG_EXTENSION
#                 img_right_fpath = os.path.join(IMGS_RIGHT_PATH, img_right_fname)
#                 cv2.imwrite(img_right_fpath, img_right)

#             if cam.has_depth():
#                 pcl_array_fname = cst.IMG_PREFIX + f"{cnt:03d}"
#                 pcl_array_fpath = os.path.join(DEPTH_ARRAY_PATH, pcl_array_fname)
#                 np.save(pcl_array_fpath, pcl_array)
#             cnt += 1
#         else:
#             cam.stop()
#             return


# if __name__ == "__main__":
#     main()
