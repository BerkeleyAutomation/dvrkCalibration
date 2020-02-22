from FLSpegtransfer.motion.dvrkDualArm import dvrkDualArm
import FLSpegtransfer.utils.CmnUtil as U
import numpy as np

class dvrkBlockTransfer():
    """
    Motion library for dvrk
    """
    def __init__(self):
        # motion library
        self.__dvrk = dvrkDualArm()

        # Motion variables
        self.__pos1 = []
        self.__rot1 = []
        self.__pos2 = []
        self.__rot2 = []
        self.__pos_org1 = [0.055, 0.0, -0.1]  # xyz position in (m)
        self.__rot_org1 = [0.0, 0.0, 0.0]    # (deg)
        self.__jaw_org1 = [0.0]              # jaw angle in (deg)
        self.__pos_org2 = [0.0, 0.0, -0.1]  # xyz position in (m)
        self.__rot_org2 = [0.0, 0.0, 0.0]    # (deg)
        self.__jaw_org2 = [0.0]              # jaw angle in (deg)

        self.__height_ready = -0.115
        self.__height_drop = -0.128
        self.__height_adjusted = -0.011  # height difference between checkerboard & blocks
        self.__rot_offset1 = [0, 0, 0]   # rot offset of the arm base (deg)
        self.__rot_offset2 = [0, 0, 0]  # rot offset of the arm base (deg)
        self.__jaw_opening = [40]   # (deg)
        self.__jaw_opening_drop = [50]  # (deg)
        self.__jaw_closing = [-5]   # (deg)

        # threading
        self.nStart = 0.0
        self.nEnd = 0.0
        self.stop_flag = False

    """
    Motion for peg transfer task
    """

    def move_origin(self):
        rot_temp1 = (np.array(self.__rot_org1) + np.array(self.__rot_offset1))*np.pi/180.
        rot_temp2 = (np.array(self.__rot_org2) + np.array(self.__rot_offset2))*np.pi/180.
        rot_org1 = U.euler_to_quaternion(rot_temp1)
        rot_org2 = U.euler_to_quaternion(rot_temp2)
        jaw_org1 = np.array(self.__jaw_org1)*np.pi/180.
        jaw_org2 = np.array(self.__jaw_org2)*np.pi/180.
        self.__dvrk.set_pose(self.__pos_org1, rot_org1, self.__pos_org2, rot_org2)
        self.__dvrk.set_jaw(jaw_org1, jaw_org2)

    def pickup(self, pos_pick1, rot_pick1, pos_pick2, rot_pick2, which_arm='Both'):
        if (which_arm=='PSM1' or which_arm=='Both') and pos_pick1 != [] and rot_pick1 != []:
            pos_ready1 = [pos_pick1[0], pos_pick1[1], self.__height_ready]
            pos_pick1[2] += self.__height_adjusted
            rot_temp1 = (np.array(rot_pick1) + np.array(self.__rot_offset1))*np.pi/180.
            q_pick1 = U.euler_to_quaternion(rot_temp1)
            jaw_opening1 = np.array(self.__jaw_opening)*np.pi/180.
            jaw_closing1 = np.array(self.__jaw_closing)*np.pi/180.
        else:
            pos_pick1 = []
            q_pick1 = []
            pos_ready1 = []
            jaw_opening1 = []
            jaw_closing1 = []

        if (which_arm=='PSM2' or which_arm=='Both') and pos_pick2 != [] and rot_pick2 != []:
            pos_ready2 = [pos_pick2[0], pos_pick2[1], self.__height_ready]
            pos_pick2[2] += self.__height_adjusted
            rot_temp2 = (np.array(rot_pick2) + np.array(self.__rot_offset2)) * np.pi / 180.
            q_pick2 = U.euler_to_quaternion(rot_temp2)
            jaw_opening2 = np.array(self.__jaw_opening) * np.pi / 180.
            jaw_closing2 = np.array(self.__jaw_closing) * np.pi / 180.
        else:
            pos_pick2 = []
            q_pick2 = []
            pos_ready2 = []
            jaw_opening2 = []
            jaw_closing2 = []

        if pos_pick1 == []:     # go origin
            rot_temp1 = (np.array(self.__rot_org1) + np.array(self.__rot_offset1)) * np.pi / 180.
            rot_org1 = U.euler_to_quaternion(rot_temp1)
            self.__dvrk.set_pose(self.__pos_org1, rot_org1, pos_ready2, q_pick2)

        if pos_pick2 == []:     # go origin
            rot_temp2 = (np.array(self.__rot_org2) + np.array(self.__rot_offset2)) * np.pi / 180.
            rot_org2 = U.euler_to_quaternion(rot_temp2)
            self.__dvrk.set_pose(pos_ready1, q_pick1, self.__pos_org2, rot_org2)

        # move upon the pick-up spot and open the jaw
        self.__dvrk.set_pose(pos_ready1, q_pick1, pos_ready2, q_pick2)
        self.__dvrk.set_jaw(jaw_opening1, jaw_opening2)

        # move down toward the block
        self.__dvrk.set_pose(pos_pick1, q_pick1, pos_pick2, q_pick2)
        self.__dvrk.set_jaw(jaw_opening1, jaw_opening2)

        # close the jaw
        self.__dvrk.set_pose(pos_pick1, q_pick1, pos_pick2, q_pick2)
        self.__dvrk.set_jaw(jaw_closing1, jaw_closing2)

        # move upon the pick-up spot hopefully with block
        self.__dvrk.set_pose(pos_ready1, q_pick1, pos_ready2, q_pick2)
        self.__dvrk.set_jaw(jaw_closing1, jaw_closing2)

    def place(self, pos_place1, rot_place1, pos_place2, rot_place2, which_arm='Both'):
        if (which_arm=='PSM1' or which_arm=='Both') and pos_place1 != [] and rot_place1 != []:
            pos_ready1 = [pos_place1[0], pos_place1[1], self.__height_ready]
            pos_place1 = [pos_place1[0], pos_place1[1], self.__height_drop]
            pos_place1[2] += self.__height_adjusted
            rot_temp1 = (np.array(rot_place1) + np.array(self.__rot_offset1)) * np.pi / 180.
            q_place1 = U.euler_to_quaternion(rot_temp1)
            jaw_opening1 = np.array(self.__jaw_opening) * np.pi / 180.
            jaw_opening_drop1 = np.array(self.__jaw_opening_drop) * np.pi / 180.
            jaw_closing1 = np.array(self.__jaw_closing) * np.pi / 180.
        else:
            pos_place1 = []
            q_place1 = []
            pos_ready1 = []
            jaw_opening1 = []
            jaw_opening_drop1 = []
            jaw_closing1 = []

        if (which_arm=='PSM2' or which_arm=='Both') and pos_place2 != [] and rot_place2 != []:
            pos_ready2 = [pos_place2[0], pos_place2[1], self.__height_ready]
            pos_place2 = [pos_place2[0], pos_place2[1], self.__height_drop]
            pos_place2[2] += self.__height_adjusted
            rot_temp2 = (np.array(rot_place2) + np.array(self.__rot_offset2)) * np.pi / 180.
            q_place2 = U.euler_to_quaternion(rot_temp2)
            jaw_opening2 = np.array(self.__jaw_opening) * np.pi / 180.
            jaw_opening_drop2 = np.array(self.__jaw_opening_drop) * np.pi / 180.
            jaw_closing2 = np.array(self.__jaw_closing) * np.pi / 180.
        else:
            pos_place2 = []
            q_place2 = []
            pos_ready2 = []
            jaw_opening2 = []
            jaw_opening_drop2 = []
            jaw_closing2 = []

        # move upon the placing spot
        self.__dvrk.set_pose(pos_ready1, q_place1, pos_ready2, q_place2)
        self.__dvrk.set_jaw(jaw_closing1, jaw_closing2)

        # move downward to the block
        self.__dvrk.set_pose(pos_place1, q_place1, pos_place2, q_place2)
        self.__dvrk.set_jaw(jaw_closing1, jaw_closing2)

        # Open the jaw
        self.__dvrk.set_pose(pos_place1, q_place1, pos_place2, q_place2)
        self.__dvrk.set_jaw(jaw_opening_drop1, jaw_opening_drop2)

        # move upon the pick-up spot hopefully without block
        self.__dvrk.set_pose(pos_ready1, q_place1, pos_ready2, q_place2)
        self.__dvrk.set_jaw(jaw_opening_drop1, jaw_opening_drop2)

if __name__ == "__main__":
    motion = dvrkBlockTransfer()
    motion.move_origin()
    while True:
        pos_pick1 = [0.150, -0.070, -0.149]
        rot_pick1 = np.array([30, 0, 0])*np.pi/180.
        q_pick1 = U.euler_to_quaternion(rot_pick1)
        pos_pick2 = [-0.066, -0.084, -0.148]
        rot_pick2 = np.array([30, 0, 0])*np.pi/180.
        q_pick2 = U.euler_to_quaternion(rot_pick2)
        motion.pickup(pos_pick1, q_pick1, pos_pick2, q_pick2, which_arm='Both')

        pos_place1 = [0.079, -0.048, -0.150]
        rot_place1 = np.array([90, 0, 0])*np.pi/180.
        q_place1 = U.euler_to_quaternion(rot_place1)
        pos_place2 = [-0.124, -0.035, -0.147]
        rot_place2 = np.array([90, 0, 0])*np.pi/180.
        q_place2 = U.euler_to_quaternion(rot_place2)
        motion.place(pos_place1, q_place1, pos_place2, q_place2, which_arm='Both')