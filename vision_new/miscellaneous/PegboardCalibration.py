from dvrk.vision.BlockDetection3D import BlockDetection3D
from dvrk.vision.PCLRegistration import PCLRegistration
from FLSpegtransfer.path import *
import cv2
import numpy as np
import open3d as o3d


class PegboardCalibration:
    def __init__(self):
        self.lower_red = np.array([0 - 20, 60, 40])  # color range for masking
        self.upper_red = np.array([0 + 20, 255, 255])

    # define boundary
    def define_boundary(self, img_color):
        # color masking
        img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        img_color_masked = cv2.inRange(img_hsv, self.lower_red, self.upper_red)

        # get points inside contour
        cnts, _ = cv2.findContours(img_color_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        max = np.amax(cnts[0].reshape(-1,2), axis=0)
        min = np.amin(cnts[0].reshape(-1,2), axis=0)
        xcr = min[0]
        ycr = min[1]
        wcr = max[0]-min[0]
        hcr = max[1]-min[1]
        return ycr, hcr, xcr, wcr

    # calculate pegboard to camera transformation matrix
    def registration_pegboard(self, img_color, img_point):
        # color masking
        img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        img_color_masked = cv2.inRange(img_hsv, self.lower_red, self.upper_red)

        # get points inside contour
        cnts, _ = cv2.findContours(img_color_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        peg_board = np.zeros_like(img_color)
        cv2.drawContours(peg_board, cnts, 0, (255, 255, 255), -1)
        pnt_board = img_point[np.all(peg_board == [255, 255, 255], axis=-1)]
        pnt_board = BlockDetection3D.remove_nan(pnt_board)*0.001    # (m)
        pcl_board = o3d.geometry.PointCloud()
        pcl_board.points = o3d.utility.Vector3dVector(pnt_board)

        # load target (pegboard model)
        pcl_model = o3d.io.read_point_cloud(root + 'img/peg_board.pcd')
        pnt_model = PCLRegistration.convert(pcl_model)[1] * 0.001   # to (m)
        pcl_model = PCLRegistration.convert(pnt_model)[0]
        print (pnt_board)
        print (pnt_model)

        # registration
        Tpc = PCLRegistration.registration(source=pcl_board, target=pcl_model)
        PCLRegistration.display_registration(pcl_board, pcl_model, Tpc)
        print ('Tpc= ', Tpc)
        np.save('Tpc.npy', Tpc)
        print ('Tpc saved')

