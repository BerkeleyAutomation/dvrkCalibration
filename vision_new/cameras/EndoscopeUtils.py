from VisualServoing.path import *
import numpy as np
import cv2


class EndoscopeUtils:
    def __init__(self):
        # camera matrices
        path = 'VisualServoing/calibration_files/endoscope/'
        self.K = np.load(root + path + 'K.npy')
        self.R = np.load(root + path + 'R.npy')
        # self.T = np.load(root + path + 'T.npy')
        self.Q = np.load(root + path + 'Q.npy')
        self.P1 = np.load(root + path + 'P1.npy')
        self.P2 = np.load(root + path + 'P2.npy')

        self.fx = self.K[:, 0, 0]
        self.fy = self.K[:, 1, 1]
        self.cx = self.K[:, 0, 2]
        self.cy = self.K[:, 1, 2]
        self.mapx1 = np.load(root+path+"mapx1.npy")
        self.mapy1 = np.load(root+path+"mapy1.npy")
        self.mapx2 = np.load(root+path+"mapx2.npy")
        self.mapy2 = np.load(root+path+"mapy2.npy")

    def rectify(self, img_left, img_right):
        rectified_left = cv2.remap(img_left, self.mapx1, self.mapy1, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_right, self.mapx2, self.mapy2, cv2.INTER_LINEAR)
        return rectified_left, rectified_right

    def triangulate(self, pnts_L, pnts_R, pnts_radius=0, which_camera='left'):  # for rectified image
        """
        pnts = [[x1,y1],...,[xn,yn]].T
        """
        pnts_L = np.array(pnts_L)
        pnts_R = np.array(pnts_R)
        disp = (pnts_L - pnts_R)[0].reshape(1,-1)
        import pdb; pdb.set_trace()

        # Homogeneous coordinate:  [[x1,y1,disp1,1], ..., [xn,yn,dispn,1]].T
        if which_camera=='left':
            pnts_homo = np.concatenate((pnts_L, disp, np.ones_like(disp)), axis=0)
        elif which_camera=='right':
            pnts_homo = np.concatenate((pnts_R, disp, np.ones_like(disp)), axis=0)
        else:
            raise ValueError

        # Triangulate
        P = self.Q.dot(pnts_homo)
        X = (P[0,:] / P[3,:]).reshape(1,-1)
        Y = (P[1,:] / P[3,:]).reshape(1,-1)
        Z = (P[2,:] / P[3,:]).reshape(1,-1)
        f = self.Q[2, 3]

        if pnts_radius == 0:
            pnts_3D = np.concatenate((X, Y, Z), axis=0)
            return pnts_3D  # [[X1,Y1,Z1], ..., [Xn,Yn,Zn].T
        else:
            pnts_radius = np.array(pnts_radius).reshape(1, -1)
            R = pnts_radius * Z / f
            pnts_3D = np.concatenate((X, Y, Z, R), axis=0)
            return pnts_3D  # [[X1,Y1,Z1,R1], ..., [Xn,Yn,Zn,Rn]].T

    # def pixel2world(self, pl, pr):  # for rectified image
    #     pixel_homo = np.array([pl[0], pl[1], pl[0] - pr[0], 1]).T  # [x, y, disparity, 1].T
    #     P = self.Q.dot(pixel_homo)
    #     X = P[0] / P[3]
    #     Y = P[1] / P[3]
    #     Z = P[2] / P[3]
    #     f = self.Q[2, 3]
    #     R = (pl[2] + pr[2]) / 2 * Z / f
    #     return X, Y, Z, R

    def world2pixel(self, P, R=0, which='left'):  # for rectified image
        """
        :param P: 3D point sets w.r.t camera frame, i.e. [[x1, y1, z1], [x2, y2, z2], [x3, y3, z3], ..., [xn, yn, zn]].T
        :param R: radius of sphere if camera detects spheres
        :param which: camera selection, left or right
        :return:
        """
        P = np.array(P).reshape(3,-1)
        R = np.array(R)
        ones = np.ones((1, np.shape(P)[1]))
        P = np.concatenate((P,ones), axis=0)        # P = [x, y, z, 1].T, dim(P) = (4, n)
        if which == 'left':
            p = self.P1.dot(P)
            fx = self.P1[0, 0]
            fy = self.P1[1, 1]
            x = (p[0, :] / p[2, :]).reshape(1, -1)
            y = (p[1, :] / p[2, :]).reshape(1, -1)
        elif which == 'right':
            p = self.P2.dot(P)
            fx = self.P2[0, 0]
            fy = self.P2[1, 1]
            x = (p[0, :] / p[2, :]).reshape(1, -1)
            y = (p[1, :] / p[2, :]).reshape(1, -1)    # rectification error (y-offset)
        else:
            raise ValueError
        p = np.concatenate((x,y), axis=0)
        if R==0:
            r = 0
        else:
            Z = P[2,:]
            r = (fx + fy) / 2 * R / Z
        return p, r     # p = [x,y]

if __name__ == "__main__":
    from VisualServoing.vision.Endoscope import Endoscope
    from VisualServoing.utils.ImgUtils import ImgUtils
    endo = Endoscope()
    endo_util = EndoscopeUtils()
    # img_left = cv2.imread(root+"endoscope_calibration_images/img_left46.png")
    # img_right = cv2.imread(root+"endoscope_calibration_images/img_right46.png")
    while True:
        img_left = endo.img_left
        img_right = endo.img_right
        if len(img_left) == 0 or len(img_right) == 0:
            pass
        else:
            img_left_rect, img_right_rect = endo_util.rectify(img_left, img_right)
            stacked = ImgUtils.compare_rectified_img(img_left_rect, img_right_rect, scale=0.47)
            cv2.imshow("", stacked)
            cv2.waitKey(1)
