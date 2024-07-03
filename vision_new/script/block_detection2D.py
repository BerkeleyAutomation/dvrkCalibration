import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.vision.BlockDetection2D import BlockDetection2D
from FLSpegtransfer.vision.VisualizeDetection import VisualizeDetection
from FLSpegtransfer.vision.GraspingPose import GraspingPose

bd = BlockDetection2D()
zivid = ZividCapture()
vd = VisualizeDetection(bd)
gp = GraspingPose(bd)

img_color = []
img_depth = []
while True:
    image = np.load('../../record/image.npy')
    depth = np.load('../../record/depth.npy')
    point = np.load('../../record/point.npy')
    img_color, img_depth, img_point = zivid.img_crop(image, depth, point)
    # zivid.capture_3Dimage()
    # img_color, img_depth, img_point = BD.img_crop(zivid.color, zivid.depth, zivid.point)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
    if img_color == [] or img_depth == [] or img_point == []:
        pass
    else:
        # find block
        pose_blks, img_blks, img_pegs, peg_pnts = bd.find_blocks(img_color, img_depth)

        # find grasping pose
        grasping_pose = gp.find_grasping_pose(pose_blks, peg_pnts, 'right_arm')

        # visualization
        img_ovl = vd.overlay(img_blks, img_pegs, pose_blks, peg_pnts)
        img_ovl = vd.overlay_grasping_pose(img_ovl, grasping_pose, (0, 0, 255))
        cv2.imshow("", img_ovl)
        cv2.waitKey(1)

        # Xc, Yc, Zc = zivid.pixel2world(x_pixel, y_pixel, d)  # 3D position in respect to the camera (m)
        # pose_blks = self.sort_moving_order(pose_blks, order=order)  # [block number, angle, tx, ty, depth, seen]