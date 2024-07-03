import os
import time

from vimba import *
import cv2
import numpy as np

import dvrk.vision.vision_constants as cst

# CAM_ID_LEFT: DEV_000F310199C1
# CAM_ID_RIGHT: DEV_000F31021FD1

CALIBRATION_DIR_LEFT = "calibration_matrices_av_left"
CALIBRATION_DIR_RIGHT = "calibration_matrices_av_right"


class AlliedVisionCaptureSingle:
    def __init__(self):
        self.cameras = {}
        self.cam_id_left = "DEV_000F310199C1"
        self.cam_id_right = "DEV_000F31021FD1"

        self.mapx1 = np.load(os.path.join(cst.DATA_PATH, CALIBRATION_DIR_LEFT, cst.MAPX_MAT_FNAME))
        self.mapy1 = np.load(os.path.join(cst.DATA_PATH, CALIBRATION_DIR_LEFT, cst.MAPY_MAT_FNAME))
        self.mapx2 = np.load(os.path.join(cst.DATA_PATH, CALIBRATION_DIR_RIGHT, cst.MAPX_MAT_FNAME))
        self.mapy2 = np.load(os.path.join(cst.DATA_PATH, CALIBRATION_DIR_RIGHT, cst.MAPY_MAT_FNAME))

        self.vimba_instance = Vimba.get_instance()

        with self.vimba_instance as vimba:
            for cam in vimba.get_all_cameras():
                cam_id = cam.get_id()
                self.setup_camera(cam, cam_id)
                self.cameras[cam_id] = cam

    def setup_camera(self, cam: Camera, cam_id):
        with cam:
            # Try to adjust GeV packet size. This Feature is only available for GigE - Cameras.
            try:
                cam.GVSPAdjustPacketSize.run()
                settings_file = os.path.join(cst.AV_SETTINGS_PATH, f"{cam_id}_settings.xml")
                cam.load_settings(settings_file, PersistType.All)

                while not cam.GVSPAdjustPacketSize.is_done():
                    pass

            except (AttributeError, VimbaFeatureError):
                pass

    def capture(self, exposure_time: int = 0, rectify: bool = False):
        with self.vimba_instance:
            if exposure_time > 0:
                with self.cameras[self.cam_id_left] as cam:
                    if exposure_time > 0:
                        exposure_time_feature = cam.get_feature_by_name("ExposureTimeAbs")
                        exposure_time_feature.set(exposure_time)
                with self.cameras[self.cam_id_right] as cam:
                    if exposure_time > 0:
                        exposure_time_feature = cam.get_feature_by_name("ExposureTimeAbs")
                        exposure_time_feature.set(exposure_time)
                time.sleep(0.1)
            with self.cameras[self.cam_id_left] as cam:
                img_left = cam.get_frame().as_opencv_image()
            with self.cameras[self.cam_id_right] as cam:
                img_right = cam.get_frame().as_opencv_image()
        if rectify and len(img_left) > 0 and len(img_right) > 0:
            img_left, img_right = self.rectify_imgs(img_left, img_right)
        return img_left, img_right

    def rectify_imgs(self, img_left, img_right):
        rectified_left = cv2.remap(img_left, self.mapx1, self.mapy1, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_right, self.mapx2, self.mapy2, cv2.INTER_LINEAR)
        return rectified_left, rectified_right


if __name__ == "__main__":
    avs = AlliedVisionCaptureSingle()
    t1 = time.time()
    iml, imr = avs.capture(70000, rectify=True)
    t2 = time.time()
    print(f"time {t2 - t1}")
    cv2.imshow("left", iml)
    cv2.imshow("right", imr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
