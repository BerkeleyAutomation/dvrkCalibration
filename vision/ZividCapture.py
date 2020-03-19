import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import datetime, time
import zivid
import cv2
import numpy as np

class ZividCapture():
    def __init__(self):
        self.image = []
        self.depth = []
        self.point = []
        app = zivid.Application()
        self.camera = app.connect_camera()
        self.t = 0.0
        self.t_prev = 0.0
        self.interval = 0.0
        self.fps = 0.0

        # 2D image setting
        self.settings_2d = zivid.Settings2D()
        self.settings_2d.iris = 26
        self.settings_2d.exposure_time = datetime.timedelta(microseconds=8333)

        # 3D capture setting
        with self.camera.update_settings() as updater:
            updater.settings.iris = 26
            updater.settings.exposure_time = datetime.timedelta(microseconds=8333)
            updater.settings.filters.reflection.enabled = True

        # img cropping
        self.ycr = 430
        self.hcr = 400
        self.xcr = 680
        self.wcr = 520

    def measure_fps(self):
        self.t_prev = self.t
        self.t = time.clock()  # sec
        self.interval = self.t - self.t_prev
        self.fps = 1/self.interval
        # print(self.interval, self.fps)

    def capture_2Dimage(self):      # measured as 20~90 fps
        with self.camera.capture_2d(self.settings_2d) as frame_2d:
            np_array = frame_2d.image().to_array()
            # print(np_array.dtype.names)
            self.image = np.asarray([np_array["r"], np_array["g"], np_array["b"]])
            self.image = np.moveaxis(self.image, [0, 1, 2], [2, 0, 1])
            self.image = self.image.astype(np.uint8)
            # self.measure_fps()
            return self.image

    def capture_3Dimage(self, img_crop=False):      # measured as 7~10 fps
        with self.camera.capture() as frame:
            np_array = frame.get_point_cloud().to_array()
            # print (np_array.dtype.names)
            # image data
            self.image = np.asarray([np_array["r"], np_array["g"], np_array["b"]])
            self.image = np.moveaxis(self.image, [0, 1, 2], [2, 0, 1])
            self.image = self.image.astype(np.uint8)
            # depth data
            self.depth = np.asarray([np_array["z"]])    # unit = (mm)
            self.depth = np.moveaxis(self.depth, [0, 1, 2], [2, 0, 1])
            # point clouds data
            self.point = np.asarray([np_array["x"], np_array["y"], np_array["z"]])  # unit = (mm)
            self.point = np.moveaxis(self.point, [0, 1, 2], [2, 0, 1])
            if img_crop == True:
                self.image, self.depth, self.point = self.img_crop(self.image, self.depth, self.point)
            # self.measure_fps()
            return self.image, self.depth, self.point

    def pixel2world(self, x, y, depth):
        Xc = (x - self.__cx + self.xcr) / self.__fx * depth
        Yc = (y - self.__cy + self.ycr) / self.__fy * depth
        Zc = depth
        return Xc, Yc, Zc

    def world2pixel(self, Xc, Yc, Zc, Rc=0):
        x = self.__fx * Xc / Zc + self.__cx - self.xcr
        y = self.__fy * Yc / Zc + self.__cy - self.ycr
        r = (self.__fx+self.__fy)/2 * Rc / Zc
        return int(x), int(y), int(r)

    def img_crop(self, img_color, img_depth, img_point):
        color_cropped = img_color[self.ycr:self.ycr + self.hcr, self.xcr:self.xcr + self.wcr]
        depth_cropped = img_depth[self.ycr:self.ycr + self.hcr, self.xcr:self.xcr + self.wcr]
        point_cropped = img_point[self.ycr:self.ycr + self.hcr, self.xcr:self.xcr + self.wcr]
        return color_cropped, depth_cropped, point_cropped

    def measure_intrinsics(self):
        self.capture_3Dimage()
        img, dep, point = self.img_crop(self.image, self.depth, self.point)
        x = []
        y = []
        Xc = []
        Yc = []
        Zc = []
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                depth = point[j,i,2]
                if np.isnan(depth) == False:
                    x.append(i)
                    y.append(j)
                    Xc.append(point[j,i,0])
                    Yc.append(point[j,i,1])
                    Zc.append(depth)
        Xc = np.array(Xc)
        Yc = np.array(Yc)
        Zc = np.array(Zc)
        A1 = np.array([Xc/Zc, np.ones(len(x))]).T
        b1 = np.array(x) + self.xcr
        A2 = np.array([Yc/Zc, np.ones(len(y))]).T
        b2 = np.array(y) + self.ycr
        # Solve by method of least squares
        c1 = np.linalg.lstsq(A1, b1, rcond=None)[0]
        c2 = np.linalg.lstsq(A2, b2, rcond=None)[0]
        fx = c1[0]
        fy = c2[0]
        cx = c1[1]
        cy = c2[1]
        Xc_est = Zc/fx*(x-cx+self.xcr)
        Yc_est = Zc/fy*(y-cy+self.ycr)
        RMSE_Xc = np.sqrt(np.sum((Xc-Xc_est)**2) / len(Xc))
        RMSE_Yc = np.sqrt(np.sum((Yc-Yc_est)**2) / len(Yc))
        print ("RMSE: ", [RMSE_Xc, RMSE_Yc], "(mm)")
        print ("fx: ", fx, "fy: ", fy)
        print ("cx: ", cx, "cy: ", cy)
        return fx, fy, cx, cy

if __name__ == "__main__":
    zc = ZividCapture()
    zc.measure_intrinsics()
    import time
    while True:
        image2D = zc.capture_2Dimage()
        # image, depth, point = zc.capture_3Dimage(img_crop=True)
        image = zc.capture_2Dimage()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # print (np.shape(point))
        # time.sleep(0.3)
        cv2.imshow("", image)
        cv2.waitKey(1)