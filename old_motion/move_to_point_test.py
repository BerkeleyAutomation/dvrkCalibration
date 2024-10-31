import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from FLSpegtransfer.vision.BlockDetection import BlockDetection
from FLSpegtransfer.motion.dvrkMotionBridgeP import dvrkMotionBridgeP
from FLSpegtransfer.vision.ZividCapture import ZividCapture
BD = BlockDetection()
zivid = ZividCapture()
dvrk = dvrkMotionBridgeP()
Trc = np.load('../calibration_files/Trc.npy')
zivid.capture_3Dimage()
img_color, img_depth, img_point = BD.img_crop(zivid.image, zivid.depth, zivid.point)
img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
# image = np.load('../record/image.npy')
# depth = np.load('../record/depth.npy')
# point = np.load('../record/point.npy')
# img_color, img_depth, img_point = BD.img_crop(image, depth, point)

while True:
    if img_color == [] or img_depth == [] or img_point == []:
        pass
    else:
        # peg_points_3d = BD.peg_detection(img_depth, img_point)
        # for pp in peg_points_3d[6:]:
        #     n, Xc, Yc, Zc = pp
        #     Rrc = Trc[:3, :3]  # transform
        #     trc = Trc[:3, -1]
        #     x, y, z = Rrc.dot([Xc, Yc, Zc]) + trc  # position in terms of the robot's coordinate
        #     dvrk.set_pose(jaw1=[0.0])
        #     dvrk.set_arm_position([0.110, 0.0, -0.105])
        #     dvrk.set_arm_position([x,y,z+0.002])

        grasping_pose, img_pegs_ovl, img_blks_ovl = BD.FLSPerception(img_depth, img_point, 'l2r')
        if not grasping_pose == [[]] * 2:
            Rrc = Trc[:3, :3]  # transform
            trc = Trc[:3, -1]
            for gp in grasping_pose:
                n, ang, Xc, Yc, Zc, seen = gp
                if seen == True:
                    x, y, z = Rrc.dot([Xc, Yc, Zc]) + trc  # position in terms of the robot's coordinate
                    gp1 = [x, y, z, ang]
                    dvrk.set_pose(jaw1=[0.0])
                    dvrk.set_arm_position(pos1=[0.110, 0.0, -0.105])
                    dvrk.set_arm_position(pos1=[x,y,z+0.00])

        cv2.imshow("color", img_pegs_ovl)
        cv2.imshow("depth", img_blks_ovl)
        cv2.waitKey(1)