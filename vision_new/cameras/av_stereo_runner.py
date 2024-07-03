import time

import numpy as np
import cv2

from dvrk.vision.cameras.Camera import Camera
import dvrk.vision.vision_constants as cst


def main():
    # define camera instance
    cam = Camera(cst.ALLIED_VISION, zivid_cam_choice="inclined", zivid_capture_type="2d", rectify_img=False)
    time.sleep(2)
    print(" ")
    print("AV Stereo ready!")
    input("Press enter to quit")
    cam.stop()
    return


if __name__ == "__main__":
    main()
