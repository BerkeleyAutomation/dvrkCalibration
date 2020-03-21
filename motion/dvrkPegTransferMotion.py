import FLSpegtransfer.utils.CmnUtil as U
from FLSpegtransfer.motion.dvrkCalibratedMotion import dvrkCalibratedMotion
import numpy as np
import time

class dvrkPegTransferMotion():
    """
    Motion library for peg transfer
    """
    def __init__(self):
        # motion library
        self.dvrk_CM = dvrkCalibratedMotion(model='neural net')

        # Motion variables
        self.pos_org1 = [0.055, 0.0, -0.1]
        self.rot_org1 = [0.0, 0.0, 0.0, 1.0]
        self.jaw_org1 = [0.0]
        self.pos_org2 = [0.0, 0.0, -0.1]
        self.rot_org2 = [0.0, 0.0, 0.0, 1.0]
        self.jaw_org2 = [0.0]

        self.height_grasp_offset_above = +0.002
        self.height_grasp_offset_below = -0.006
        self.height_ready = -0.115
        self.height_drop = -0.137
        self.jaw_opening = [np.deg2rad(75)]
        self.jaw_opening_drop = [np.deg2rad(75)]
        self.jaw_closing = [np.deg2rad(10)]

    def move_random(self):
        # Load trajectory
        root = '/home/hwangmh/pycharmprojects/FLSpegtransfer/'
        file_dir = root + 'experiment/0_trajectory_extraction/training_traj_insertion_1600.npy'
        joint_traj = np.load(file_dir)
        for joint in joint_traj[:10]:
            joint[0] = 0.55     # hold q1, q2, q3
            joint[1] = 0.0
            joint[2] = 0.1152
            self.dvrk_CM.set_joint(joint, jaw=[0.0])

    def move_origin(self):
        self.dvrk_CM.set_pose(self.pos_org1, self.rot_org1, self.jaw_org1)

    def pickup_block(self, grasping_pose1=[], grasping_pose2=[]):
        gp1 = grasping_pose1
        gp2 = grasping_pose2
        if gp1!=[]:
            # go above block to pickup
            pos = [gp1[0], gp1[1], self.height_ready]
            rot = np.deg2rad([gp1[3], 0.0, 0.0])
            rot = U.euler_to_quaternion(rot)
            jaw = self.jaw_closing
            self.dvrk_CM.set_pose(pos, rot, jaw)

            # approach block & open jaw
            pos = [gp1[0], gp1[1], gp1[2]+self.height_grasp_offset_above]
            rot = np.deg2rad([gp1[3], 0.0, 0.0])
            rot = U.euler_to_quaternion(rot)
            jaw = self.jaw_opening
            self.dvrk_CM.set_pose(pos, rot, jaw)

            # go down toward block & close jaw
            pos = [gp1[0], gp1[1], gp1[2]+self.height_grasp_offset_below]
            rot = np.deg2rad([gp1[3], 0.0, 0.0])
            rot = U.euler_to_quaternion(rot)
            jaw = self.jaw_closing
            self.dvrk_CM.set_pose(pos, rot, jaw)

            # go up with block
            pos = [gp1[0], gp1[1], self.height_ready]
            rot = np.deg2rad([gp1[3], 0.0, 0.0])
            rot = U.euler_to_quaternion(rot)
            jaw = self.jaw_closing
            self.dvrk_CM.set_pose(pos, rot, jaw)

        if gp2!=[]:
            raise NotImplementedError


    def place_block(self, placing_pose1=[], placing_pose2=[]):
        pp1 = placing_pose1
        pp2 = placing_pose2
        if pp1!=[]:
            # be ready to place & close jaw
            pos = [pp1[0], pp1[1], self.height_ready]
            rot = np.deg2rad([pp1[3], 0.0, 0.0])
            rot = U.euler_to_quaternion(rot)
            jaw = self.jaw_closing
            self.dvrk_CM.set_pose(pos, rot, jaw)

            # go down toward peg & open jaw
            pos = [pp1[0], pp1[1], self.height_drop]
            rot = np.deg2rad([pp1[3], 0.0, 0.0])
            rot = U.euler_to_quaternion(rot)
            jaw = self.jaw_opening_drop
            self.dvrk_CM.set_pose(pos, rot, jaw)

            # go up
            pos = [pp1[0], pp1[1], self.height_ready]
            rot = np.deg2rad([pp1[3], 0.0, 0.0])
            rot = U.euler_to_quaternion(rot)
            jaw = self.jaw_opening_drop
            self.dvrk_CM.set_pose(pos, rot, jaw)
        if pp2 != []:
            raise NotImplementedError


if __name__ == "__main__":
    motion = dvrkPegTransferMotion()
    motion.move_origin()