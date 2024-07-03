import os

import numpy as np
import cv2

import dvrk.vision.vision_constants as cst

CALIBRATION_DIR_LEFT = "calibration_matrices_av_left"
CALIBRATION_DIR_RIGHT = "calibration_matrices_av_right"


class AlliedVisionUtils:
    def __init__(self):
        # camera matrices
        self.K_left = np.load(os.path.join(cst.DATA_PATH, CALIBRATION_DIR_LEFT, cst.K_MAT_FNAME))
        self.K_right = np.load(os.path.join(cst.DATA_PATH, CALIBRATION_DIR_RIGHT, cst.K_MAT_FNAME))
        self.R = np.load(os.path.join(cst.DATA_PATH, CALIBRATION_DIR_LEFT, cst.R_STEREO_MAT_FNAME))
        self.T = np.load(os.path.join(cst.DATA_PATH, CALIBRATION_DIR_LEFT, cst.T_STEREO_MAT_FNAME))
        self.Q = np.load(os.path.join(cst.DATA_PATH, CALIBRATION_DIR_LEFT, cst.Q_STEREO_MAT_FNAME))
        self.P1 = np.load(os.path.join(cst.DATA_PATH, CALIBRATION_DIR_LEFT, cst.P_MAT_FNAME))
        self.P2 = np.load(os.path.join(cst.DATA_PATH, CALIBRATION_DIR_RIGHT, cst.P_MAT_FNAME))

        self.fx = self.K_left[0, 0]
        self.fy = self.K_left[1, 1]
        self.cx = self.K_left[0, 2]
        self.cy = self.K_left[1, 2]
        self.mapx1 = np.load(os.path.join(cst.DATA_PATH, CALIBRATION_DIR_LEFT, cst.MAPX_MAT_FNAME))
        self.mapy1 = np.load(os.path.join(cst.DATA_PATH, CALIBRATION_DIR_LEFT, cst.MAPY_MAT_FNAME))
        self.mapx2 = np.load(os.path.join(cst.DATA_PATH, CALIBRATION_DIR_RIGHT, cst.MAPX_MAT_FNAME))
        self.mapy2 = np.load(os.path.join(cst.DATA_PATH, CALIBRATION_DIR_RIGHT, cst.MAPY_MAT_FNAME))

    def rectify(self, img_left, img_right):
        rectified_left = cv2.remap(img_left, self.mapx1, self.mapy1, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(img_right, self.mapx2, self.mapy2, cv2.INTER_LINEAR)
        return rectified_left, rectified_right

    def rectify_single(self, img, is_left):
        if is_left:
            rect_img = cv2.remap(img, self.mapx1, self.mapy1, cv2.INTER_LINEAR)
        else:
            rect_img = cv2.remap(img, self.mapx2, self.mapy2, cv2.INTER_LINEAR)
        return rect_img

    def triangulate(self, pnts_L, pnts_R, pnts_radius=0, which_camera="left"):  # for rectified image
        """
        pnts = [[x1,y1],...,[xn,yn]]
        """
        pnts_L = np.array(pnts_L).T
        pnts_R = np.array(pnts_R).T
        disp = (pnts_L - pnts_R)[0].reshape(1, -1)

        # Homogeneous coordinate:  [[x1,y1,disp1,1], ..., [xn,yn,dispn,1]].T
        if which_camera == "left":
            pnts_homo = np.concatenate((pnts_L, disp, np.ones_like(disp)), axis=0)
        elif which_camera == "right":
            pnts_homo = np.concatenate((pnts_R, disp, np.ones_like(disp)), axis=0)
        else:
            raise ValueError

        # Triangulate
        P = self.Q.dot(pnts_homo)
        X = (P[0, :] / P[3, :]).reshape(1, -1)
        Y = (P[1, :] / P[3, :]).reshape(1, -1)
        Z = (P[2, :] / P[3, :]).reshape(1, -1)
        f = self.Q[2, 3]

        if pnts_radius == 0:
            pnts_3D = np.concatenate((X, Y, Z), axis=0).T
            return pnts_3D  # [[X1,Y1,Z1], ..., [Xn,Yn,Zn]
        else:
            pnts_radius = np.array(pnts_radius).reshape(1, -1)
            R = pnts_radius * Z / f
            pnts_3D = np.concatenate((X, Y, Z, R), axis=0).T
            return pnts_3D  # [[X1,Y1,Z1,R1], ..., [Xn,Yn,Zn,Rn]]

    # def pixel2world(self, pl, pr):  # for rectified image
    #     pixel_homo = np.array([pl[0], pl[1], pl[0] - pr[0], 1]).T  # [x, y, disparity, 1].T
    #     P = self.Q.dot(pixel_homo)
    #     X = P[0] / P[3]
    #     Y = P[1] / P[3]
    #     Z = P[2] / P[3]
    #     f = self.Q[2, 3]
    #     R = (pl[2] + pr[2]) / 2 * Z / f
    #     return X, Y, Z, R

    def world2pixel(self, P, R=0, which_camera="left"):  # for rectified image
        """
        :param P: 3D point sets w.r.t camera frame, i.e. [[X1, Y1, Z1], [X2, Y2, Z2], [X3, Y3, Z3], ..., [Xn, Yn, Zn]].T
        :param R: radius of sphere if camera detects spheres
        :param which: camera selection, left or right
        :return:
        """
        P = np.array(P).reshape(3, -1)
        R = np.array(R)
        ones = np.ones((1, np.shape(P)[1]))
        P = np.concatenate((P, ones), axis=0)  # P = [x, y, z, 1].T, dim(P) = (4, n)
        if which_camera == "left":
            p = self.P1.dot(P)
            fx = self.P1[0, 0]
            fy = self.P1[1, 1]
            x = (p[0, :] / p[2, :]).reshape(1, -1)
            y = (p[1, :] / p[2, :]).reshape(1, -1)
        elif which_camera == "right":
            p = self.P2.dot(P)
            fx = self.P2[0, 0]
            fy = self.P2[1, 1]
            x = (p[0, :] / p[2, :]).reshape(1, -1)
            y = (p[1, :] / p[2, :]).reshape(1, -1) + 3  # rectification error (y-offset)
        else:
            raise ValueError
        p = np.concatenate((x, y), axis=0)
        if R == 0:
            r = 0
        else:
            Z = P[2, :]
            r = (fx + fy) / 2 * R / Z
        return p, r  # p = [[x1,y1], ..., [xn,yn]].T


if __name__ == "__main__":
    av_util = AlliedVisionUtils()
