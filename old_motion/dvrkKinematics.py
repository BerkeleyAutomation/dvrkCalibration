from robopy.base.serial_link import SerialLink
from robopy.base.serial_link import Revolute
from robopy.base.serial_link import Prismatic
from math import pi
import numpy as np
from robopy import transforms as tr
from robopy import graphics
import FLSpegtransfer.utils.CmnUtil as U

class dvrkKinematics(SerialLink):
    def __init__(self):

        # self.qn = np.matrix([[0, pi / 4, pi, 0, pi / 4, 0]])
        # self.qr = np.matrix([[0, pi / 2, -pi / 2, 0, 0, 0]])
        # self.qz = np.matrix([[0, 0, 0, 0, 0, 0]])
        # self.qs = np.matrix([[0, 0, -pi / 2, 0, 0, 0]])
        # self.scale = 1
        # param = {
        #     "cube_axes_x_bounds": np.matrix([[-1.5, 1.5]]),
        #     "cube_axes_y_bounds": np.matrix([[-0.7, 1.5]]),
        #     "cube_axes_z_bounds": np.matrix([[-1.5, 1.5]]),
        #     "floor_position": np.matrix([[0, -0.7, 0]])
        # }
        self.L1 = 0.4318  # Rcc (m)
        self.L2 = 0.4162  # tool
        self.L3 = 0.0091  # pitch ~ yaw (m)
        self.L4 = 0.0102  # yaw ~ tip (m)
        links = [
            Revolute(j=0, theta=0, d=0, a=0, alpha=-pi/2, offset=pi/2, qlim=(-90 * np.pi / 180, 90 * np.pi / 180)),
            Revolute(j=0, theta=0, d=0, a=0, alpha=-pi/2, offset=pi/2, qlim=(-60 * np.pi / 180, 60 * np.pi / 180)),
            Prismatic(j=0, theta=0, d=0, a=0, alpha=0, offset=-self.L1+self.L2, qlim=(0, 0.5)),
            Revolute(j=0, theta=0, d=0, a=0, alpha=pi/2, offset=0, qlim=(-90 * np.pi / 180, 90 * np.pi / 180)),
            Revolute(j=0, theta=0, d=0, a=self.L3, alpha=-pi/2, offset=pi/2, qlim=(-90 * np.pi / 180, 90 * np.pi / 180)),
            Revolute(j=0, theta=0, d=0, a=self.L4, alpha=0, offset=0, qlim=(-90 * np.pi / 180, 90 * np.pi / 180))]

        base = tr.trotx(90, unit='deg')
        tool = tr.trotz(90, unit='deg')*tr.trotx(-90, unit='deg')
        # tool = tr.trotz(-90, unit='deg') * tr.trotx(-90, unit='deg')

        # def __init__(self, links, name=None, base=None, tool=None, stl_files=None, q=None, colors=None, param=None):

        file_names = SerialLink._setup_file_names(7)
        colors = graphics.vtk_named_colors(["Red", "DarkGreen", "Blue", "Cyan", "Magenta", "Yellow", "White"])

        super().__init__(links=links, name='dvrk', base=base, tool=tool, colors=colors)

    def pose_to_transform(self, pos, rot):
        """

        :param pos: position (m)
        :param rot: quaternion (qx, qy, qz, qw)
        :return:
        """
        T = np.zeros((4, 4))
        R = U.quaternion_to_R(rot[0], rot[1], rot[2], rot[3])
        T[:3,:3] = R
        T[:3,-1] = np.transpose(pos)
        T[-1,-1] = 1
        return T

    def pose_to_joint(self, pos, rot):
        if pos==[] or rot==[]:
            joint = []
        else:
            T = np.matrix(self.pose_to_transform(pos, rot))    # current transformation
            q0 = np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            ik_sol = self.ikine(T, q0)
            joint = [ik_sol[0,0], ik_sol[0,1], ik_sol[0,2], ik_sol[0,3], ik_sol[0,4], ik_sol[0,5]]
        return joint