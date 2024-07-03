import time
import datetime

import cv2
import numpy as np
root = '/home/davinci/dvrkCalibration'
import sys
sys.path.append(root)
from vision_new.cameras.AlliedVisionCapture import AlliedVisionCapture
from vision_new.cameras.AlliedVisionCaptureSingle import AlliedVisionCaptureSingle
from vision_new.cameras.ZividCapture import ZividCapture
from vision_new import vision_constants as cst


class Camera:
    def __init__(
        self,
        camera_type: str,
        resolution="720p",
        brightness=4,
        contrast=4,
        hue=0,
        saturation=2,
        sharpness=4,
        white_balance=3000,
        exposure=30,
        gain=15,
        zivid_cam_choice="inclined",
        use_ROS=True,
        visualize=False,
        zivid_capture_type="2d",
        rectify_img=False,
    ) -> None:
        self._camera_type = camera_type

        if self._camera_type == cst.ALLIED_VISION:
            self.cam = AlliedVisionCapture(use_ROS=use_ROS, visualize=visualize)
            self.cam.start()
            self._is_stereo = True
            self._has_depth = False
            self._av_capture_type = "rectified" if rectify_img else "original"
        elif self._camera_type == cst.ALLIED_VISION_SINGLE:
            self.cam = AlliedVisionCaptureSingle()
            self._is_stereo = True
            self._has_depth = False
            self._rectify_img = rectify_img
            self._av_exposure_time = 0
        # elif self._camera_type == cst.ZED:
        #     self.cam = ZedImageCapture(
        #         resolution=resolution,
        #         brightness=brightness,
        #         contrast=contrast,
        #         hue=hue,
        #         saturation=saturation,
        #         sharpness=sharpness,
        #         whitebalance_temp=white_balance,
        #         exposure=exposure,
        #         gain=gain,
        #     )
        #     self._is_stereo = True
        #     self._has_depth = False
        elif self._camera_type == cst.ZIVID:
            self.cam = ZividCapture(which_camera=zivid_cam_choice)
            self.cam.start()
            self._is_stereo = False
            self._has_depth = zivid_capture_type == "3d"
            self._zivid_capture_type = zivid_capture_type
        else:
            raise

    def capture(self):
        """Outputs image in BGR"""
        if self._camera_type == cst.ALLIED_VISION:
            img_left, img_right = self.cam.capture(which=self._av_capture_type)
            return img_left, img_right
        elif self._camera_type == cst.ALLIED_VISION_SINGLE:
            img_left, img_right = self.cam.capture(self._av_exposure_time, self._rectify_img)
            self._av_exposure_time = 0
            return img_left, img_right
        # elif self._camera_type == cst.ZED:
        #     img_left, img_right = self.cam.capture_image()
        #     return img_left, img_right
        elif self._camera_type == cst.ZIVID:
            if self._zivid_capture_type == "2d":
                img = self.cam.capture_2Dimage(color="RGB")
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                pcl = None
                return img
            elif self._zivid_capture_type == "3d":
                img, depth, pcl,intrinsics_matrix,distortion_coefficients = self.cam.capture_3Dimage(color="RGB")
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img, depth, pcl,intrinsics_matrix,distortion_coefficients

    def get_camera_type(self):
        return self._camera_type

    def is_stereo(self):
        return self._is_stereo

    def has_depth(self):
        return self._has_depth

    def set_exposure(self, exposure):
        # if self._camera_type == cst.ZED:
        #     self.cam.set_exposure(exposure)
        if self._camera_type == cst.ZIVID:
            self.cam.settings_2d.acquisitions[0].exposure_time = datetime.timedelta(microseconds=exposure)
        # elif self._camera_type == cst.ALLIED_VISION:
        #     self.cam.set_exposure(exposure)
        elif self._camera_type == cst.ALLIED_VISION_SINGLE:
            self._av_exposure_time = exposure
        else:
            raise NotImplementedError

    def set_gain(self, gain):
        if self._camera_type == cst.ZED:
            self.cam.set_gain(gain)
        else:
            raise NotImplementedError

    def set_brightness(self, brightness):
        if self._camera_type == cst.ZED:
            self.cam.set_brightness(brightness)
        elif self._camera_type == cst.ZIVID:
            self.cam.settings_2d.acquisitions[0].brightness = brightness
        else:
            raise NotImplementedError

    def set_aperture(self, aperture):
        if self._camera_type == cst.ZIVID:
            self.cam.settings_2d.acquisitions[0].aperture = aperture
        else:
            raise NotImplementedError

    def stop(self):
        if self._camera_type == cst.ALLIED_VISION:
            self.cam.stop()
