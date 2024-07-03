import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
from FLSpegtransfer.vision.ZividCapture import ZividCapture
from FLSpegtransfer.vision.BlockDetection3D import BlockDetection3D
from FLSpegtransfer.vision.PegboardCalibration import PegboardCalibration
from FLSpegtransfer.vision.VisualizeDetection import VisualizeDetection
from FLSpegtransfer.vision.GraspingPose3D import GraspingPose3D


zivid = ZividCapture()
zivid.start()
peg = PegboardCalibration()
bd = BlockDetection3D()
vd = VisualizeDetection()
gp = GraspingPose3D()

# define region of interest
color = np.load('../../record/color_new.npy')
depth = np.load('../../record/depth_new.npy')
point = np.load('../../record/point_new.npy')
# color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
color, depth, point = zivid.capture_3Dimage(color='BGR')
ycr, hcr, xcr, wcr = peg.define_boundary(color)
dx = 200
dy = 200
zivid.ycr = ycr-dy
zivid.hcr = hcr+2*dy
zivid.xcr = xcr-dx
zivid.wcr = wcr+2*dx
bd.find_pegs(img_color=color, img_point=point)

# pegboard registration if necessary
# peg.registration_pegboard(color, point)

import open3d as o3d

while True:
    # color = np.load('../../record/color_new.npy')
    # depth = np.load('../../record/depth_new.npy')
    # point = np.load('../../record/point_new.npy')
    # color, depth, point = zivid.img_crop(color, depth, point)
    # color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
    color, depth, point = zivid.capture_3Dimage(img_crop=True, color='BGR')
    if color == [] or depth == [] or point == []:
        pass
    else:
        # # find block pose & grasping pose for each
        # for i in range(12):
        #     nb_blk = i
        #     pose_blk, pnt_blk, pnt_mask = bd.find_block(block_number=nb_blk, img_color=color, img_point=point)
        #     pose_grasping = gp.find_grasping_pose(pose_blk=pose_blk, peg_point=bd.pnt_pegs[nb_blk])
        #
        #     # visualization
        #     pnt_grasping = [np.array(pose_grasping)[1:]]
        #     vd.plot3d(pnt_blk, pnt_mask, bd.pnt_pegs, pnt_grasping)


        # find all block pose & grasping pose
        bd.find_block_all(img_color=color, img_point=point)
        gp.find_grasping_pose_all(bd.pose_blks, bd.pnt_pegs)

        # visualization
        pnt_blocks = [x for x in bd.pnt_blks if x != []]
        pnt_blocks = np.concatenate(pnt_blocks, axis=0)
        pnt_masks = [x for x in bd.pnt_masks if x != []]
        pnt_masks = np.concatenate(pnt_masks, axis=0)
        pnt_grasping = [np.array(p)[2:5] for p in gp.pose_grasping if p[5]==True]
        vd.plot3d(pnt_blocks, pnt_masks, bd.pnt_pegs, pnt_grasping)