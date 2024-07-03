import numpy as np
import matplotlib.pyplot as plt
import time

class ZividUtils:
    def __init__(self, which_camera):
        # Define camera intrinsic parameters
        if which_camera == 'inclined':
            # Zivid One+ S
            self.D = np.array([-0.2847377359867096, 0.40678367018699646, -9.63963902904652e-05, 0.00011100077244918793, -0.5090765953063965])
            self.K = np.array([[2765.367919921875, 0.0, 962.8666381835938], [0.0, 2765.956787109375, 566.1170043945312], [0.0, 0.0, 1.0]])
            self.R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            self.P = np.array([[2765.367919921875, 0.0, 962.8666381835938, 0.0], [0.0, 2765.956787109375, 566.1170043945312, 0.0],[0.0, 0.0, 1.0, 0.0]])
        elif which_camera == 'overhead':
            # Zivid One+ M
            self.D = np.array([-0.2826650142669678, 0.42553916573524475, -0.0005135679966770113, -0.000839113024994731, -0.5215581655502319])
            self.K = np.array([[2776.604248046875, 0.0, 952.436279296875], [0.0, 2776.226318359375, 597.9248046875], [0.0, 0.0, 1.0]])
            self.R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            self.P = np.array([[2776.604248046875, 0.0, 952.436279296875, 0.0], [0.0, 2776.226318359375, 597.9248046875, 0.0], [0.0, 0.0, 1.0, 0.0]])
        self.fx = self.K[0, 0]
        self.fy = self.K[1, 1]
        self.cx = self.K[0, 2]
        self.cy = self.K[1, 2]

        # crop parameters
        self.xcr = 0
        self.wcr = 0
        self.ycr = 0
        self.hcr = 0

    def measure_fps(self):
        self.t_prev = self.t
        self.t = time.clock()  # sec
        self.interval = self.t - self.t_prev
        self.fps = 1/self.interval

    def visualize(self):
        rgb, _, xyz = self.capture_3Dimage(img_crop=False)
        self.display_rgb(rgb)
        self.display_depthmap(xyz)
        self.display_pointcloud(rgb, xyz)

    def display_rgb(self, rgb, block=True):
        plt.figure()
        plt.imshow(rgb)
        plt.title("RGB image")
        plt.show(block=block)

    def display_depthmap(self, xyz):
        plt.figure()
        plt.imshow(
            xyz[:, :, 2],   # depth value
            vmin=np.nanmin(xyz[:, :, 2]),
            vmax=np.nanmax(xyz[:, :, 2]),
            cmap="jet",
        )
        plt.colorbar()
        plt.title("Depth map")
        plt.show(block=True)

    # def display_pointcloud(self, xyz, rgb=[]):
    #     pcl = xyz
    #     pcl[np.isnan(xyz[:, :, 2])] = 0
    #     viewer = pptk.viewer(pcl)
    #     if rgb != []:
    #         viewer.attributes(rgb.reshape(-1, 3) / 255.0)

    def pixel2world(self, x, y, depth):
        Xc = (x - self.cx + self.xcr) / self.fx * depth
        Yc = (y - self.cy + self.ycr) / self.fy * depth
        Zc = depth
        return Xc, Yc, Zc

    def world2pixel(self, Xc, Yc, Zc, Rc=0):
        x = self.fx * Xc / Zc + self.cx - self.xcr
        y = self.fy * Yc / Zc + self.cy - self.ycr
        r = (self.fx+self.fy)/2 * Rc / Zc
        return int(x), int(y), int(r)

    def img_crop(self, color, depth, point):
        color_cropped = color[self.ycr:self.ycr + self.hcr, self.xcr:self.xcr + self.wcr]
        depth_cropped = depth[self.ycr:self.ycr + self.hcr, self.xcr:self.xcr + self.wcr]
        point_cropped = point[self.ycr:self.ycr + self.hcr, self.xcr:self.xcr + self.wcr]
        return color_cropped, depth_cropped, point_cropped

    def measure_intrinsics(self, color, point, img_crop=False):
        x = []
        y = []
        Xc = []
        Yc = []
        Zc = []
        if img_crop:
            xcr = self.xcr
            ycr = self.ycr
        else:
            xcr = 0
            ycr = 0
        for i in range(color.shape[1]):
            for j in range(color.shape[0]):
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
        b1 = np.array(x) + xcr
        A2 = np.array([Yc/Zc, np.ones(len(y))]).T
        b2 = np.array(y) + ycr
        # Solve by method of least squares
        c1 = np.linalg.lstsq(A1, b1, rcond=None)[0]
        c2 = np.linalg.lstsq(A2, b2, rcond=None)[0]
        fx = c1[0]
        fy = c2[0]
        cx = c1[1]
        cy = c2[1]
        Xc_est = Zc/fx*(x-cx+xcr)
        Yc_est = Zc/fy*(y-cy+ycr)
        RMSE_Xc = np.sqrt(np.sum((Xc-Xc_est)**2) / len(Xc))
        RMSE_Yc = np.sqrt(np.sum((Yc-Yc_est)**2) / len(Yc))
        print ("RMSE: ", [RMSE_Xc, RMSE_Yc], "(mm)")
        print ("fx: ", fx, "fy: ", fy)
        print ("cx: ", cx, "cy: ", cy)
        return fx, fy, cx, cy