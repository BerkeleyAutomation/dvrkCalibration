import os
from typing import Tuple
import time

import numpy as np
import cv2

from dvrk.vision.cameras.Camera import Camera
import dvrk.vision.vision_constants as cst
from dvrk.vision.cameras.AlliedVisionUtils import AlliedVisionUtils

IMGS_LEFT_DIR = "left_calibration_2024_05_25"  # use for mono cameras
IMGS_RIGHT_DIR = "right_calibration_2024_05_25"
DEPTH_ARRAY_DIR = "zivid_calibration_2024_05_25"
IMGS_LEFT_PATH = os.path.join(cst.DATA_PATH, IMGS_LEFT_DIR)
IMGS_RIGHT_PATH = os.path.join(cst.DATA_PATH, IMGS_RIGHT_DIR)
DEPTH_ARRAY_PATH = os.path.join(cst.DATA_PATH, DEPTH_ARRAY_DIR)
IMG_COUNT_START = 0

CAMERA_TYPE = cst.ALLIED_VISION
# CAMERA_TYPE = cst.ZIVID
av_util = AlliedVisionUtils()


def capture_camera_imgs(cam: Camera) -> Tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    while True:
        if cam.is_stereo():
            img_left, img_right = cam.capture()
            pcl = None
            if len(img_left) == 0 or len(img_right) == 0:
                time.sleep(0.1)
                continue

            img_left = av_util.rectify_single(img_left, is_left=True)
            img_right = av_util.rectify_single(img_right, is_left=False)
            img_stereo = np.hstack((img_left, img_right))
            img_stereo = cv2.resize(img_stereo, None, fx=0.5, fy=0.5)
            cv2.imshow("cam stereo img: press c to capture img, q to quit", img_stereo)

        elif cam.has_depth():
            img_left, pcl = cam.capture()
            img_right = None
            if len(img_left) == 0:
                time.sleep(0.1)
                continue
            cv2.imshow("cam stereo img: press c to capture img, q to quit", img_left)
        else:
            img_left, img_right = cam.capture()
            pcl = None
            if len(img_left) == 0:
                time.sleep(0.1)
                continue
            cv2.imshow("cam img: press c to capture img, q to quit", img_left)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            cv2.destroyAllWindows()
            return True, img_left, img_right, pcl
        elif key == ord("q"):
            cv2.destroyAllWindows()
            return False, None, None, None


def main():

    # define camera instance
    cam = Camera(CAMERA_TYPE, zivid_cam_choice="inclined", zivid_capture_type="2d", rectify_img=False)

    os.makedirs(IMGS_LEFT_PATH, exist_ok=True)
    if cam.is_stereo():
        os.makedirs(IMGS_RIGHT_PATH, exist_ok=True)
    if cam.has_depth():
        os.makedirs(DEPTH_ARRAY_PATH, exist_ok=True)

    # capturing loop
    cnt = IMG_COUNT_START
    while True:
        is_capture, img_left, img_right, pcl_array = capture_camera_imgs(cam)
        if is_capture:
            img_left_fname = cst.IMG_PREFIX + f"{cnt:03d}" + cst.IMG_EXTENSION
            img_left_fpath = os.path.join(IMGS_LEFT_PATH, img_left_fname)
            cv2.imwrite(img_left_fpath, img_left)

            if cam.is_stereo():
                img_right_fname = cst.IMG_PREFIX + f"{cnt:03d}" + cst.IMG_EXTENSION
                img_right_fpath = os.path.join(IMGS_RIGHT_PATH, img_right_fname)
                cv2.imwrite(img_right_fpath, img_right)

            if cam.has_depth():
                pcl_array_fname = cst.IMG_PREFIX + f"{cnt:03d}"
                pcl_array_fpath = os.path.join(DEPTH_ARRAY_PATH, pcl_array_fname)
                np.save(pcl_array_fpath, pcl_array)
            cnt += 1
        else:
            cam.stop()
            return


if __name__ == "__main__":
    main()
