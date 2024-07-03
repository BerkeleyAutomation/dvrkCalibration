from dvrk.utils.ImgUtils import ImgUtils
import numpy as np


class GraspingPose3D:
    def __init__(self, dist_gps, dist_pps, which_arm='PSM1'):
        # sample grasping poses
        self.sample_gps = self.sample_grasping_pose(gp_number=2, dist_center=dist_gps, dist_gp=2.0, which_arm=which_arm)
        self.sample_pps = self.sample_grasping_pose(gp_number=2, dist_center=dist_pps, dist_gp=2.0, which_arm=which_arm)
        self.which_arm = which_arm
        self._pose_grasping = []
        self._pose_placing = []
        self._pose_grasping_arg = []

    @property
    def pose_grasping(self):
        return self._pose_grasping

    @property
    def pose_grasping_arg(self):
        return self._pose_grasping_arg

    @pose_grasping.setter
    def pose_grasping(self, value):
        self._pose_grasping = value
        # print ("(" + self.which_arm + ") " + "grasping pose updated: ", value)

    @pose_grasping_arg.setter
    def pose_grasping_arg(self, value):
        self._pose_grasping_arg = value
        # print ('')
        # print ("(" + self.which_arm + ") " + "grasping pose arg updated: ", value)

    @property
    def pose_placing(self):
        return self._pose_placing

    @pose_placing.setter
    def pose_placing(self, value):
        self._pose_placing = value
        # print("(" + self.which_arm + ") " + "placing pose updated: ", value)

    def sample_grasping_pose(self, gp_number, dist_center, dist_gp, which_arm):
        group1 = []
        for i in range(gp_number):
            x = -(gp_number-1)*dist_gp/2
            y = dist_center
            group1.append([x+dist_gp*i, y])
        group0 = ImgUtils.transform_pnts(group1, [.0, .0], -120.0, .0, .0)
        group2 = ImgUtils.transform_pnts(group1, [.0, .0], 120.0, .0, .0)
        sample_gp = np.reshape([group0, group1, group2], (gp_number*3, -1))
        if which_arm == 'PSM2':
            angle = [-30]*gp_number + [90]*gp_number + [-150]*gp_number
        elif which_arm == 'PSM1':
            angle = [150]*gp_number + [-90]*gp_number + [30]*gp_number
        else:
            raise NameError
        angle = np.array(angle).reshape(3*gp_number, -1)
        sample_gp = np.concatenate((angle, sample_gp), axis=1)
        sample_gp = np.pad(sample_gp, ((0,0),(0,1)), mode='constant', constant_values=0.0)  # z = 0.0
        return sample_gp    # [angle, x, y, z]

    def transform_grasping_pose(self, sample_grasping_pose, block_angle, transform):
        gps = sample_grasping_pose
        T = transform
        R = T[:3, :3]
        t = T[:3, -1]
        transformed = []
        for gp in gps:  # [ang, x, y, z, 1]
            new_ang = gp[0] - block_angle   # block angle has a different z-direction
            coord =  R.dot(gp[1:]) + t
            transformed.append([new_ang] + list(coord))
        return transformed   # [angle, x, y, z]

    # grasping angle is positive in counter-clockwise direction
    def find_grasping_pose(self, pose_blk, peg_point):
        if pose_blk == []:
            pose_grasping = []
            arg = []
        else:
            blk_ang, T = pose_blk
            gps = self.sample_gps
            gps_transformed = np.array(self.transform_grasping_pose(gps, blk_ang, T))

            # 1. Select grasping points in roll angle of -70 ~ 70 (deg)
            # 2. Choose the farthest point from the peg among the three points on the same side
            con1 = (gps_transformed[:, 0] >= -70) & (gps_transformed[:, 0] <= 70)
            arg1 = np.argwhere(con1)
            group = gps_transformed[con1]

            dist = np.linalg.norm(group[:,1:3] - peg_point[:2], axis=1)
            arg2 = np.argmax(dist)

            arg = arg1[arg2]
            pose_grasping = gps_transformed[arg]
        self.pose_grasping_arg = arg[0]
        self.pose_grasping = pose_grasping[0]     # [grasping angle, x, y, z]

    # grasping angle is positive in counter-clockwise direction
    def find_grasping_pose_handover(self, pose_blk, which_arm): # which arm to pick-up
        if pose_blk == []:
            pose_grasping = []
            arg = []
        else:
            blk_ang, T = pose_blk
            gps = self.sample_gps
            gps_transformed = np.array(self.transform_grasping_pose(gps, blk_ang, T))

            # 1. Select grasping points in roll angle of -70 ~ 70 (deg)
            # 2. Choose the point on the far right
            if which_arm == "PSM1":
                con1 = (gps_transformed[:, 0] >= -70) & (gps_transformed[:, 0] <= 70)
            elif which_arm == "PSM2":
                con1 = (gps_transformed[:, 0] >= -70) & (gps_transformed[:, 0] <= 70)
            else:
                raise ValueError
            arg1 = np.argwhere(con1)
            group = gps_transformed[con1]

            if which_arm == "PSM1":
                arg2 = np.argmax(group[:, 1])    # select a point having the largest x
            elif which_arm == 'PSM2':
                arg2 = np.argmin(group[:, 1])
            else:
                raise ValueError
            arg = arg1[arg2]
            pose_grasping = gps_transformed[arg]
        self.pose_grasping_arg = arg[0]
        self.pose_grasping = pose_grasping[0]     # [grasping angle, x, y, z]

    # grasping angle is positive in counter-clockwise direction
    def find_grasping_pose_all(self, pose_blks, peg_points):
        grasping_pose = []
        for res in pose_blks:   # [n, blk_angle, transform]
            if res == []:
                grasping_pose.append([])
            else:
                n, blk_ang, T = res
                if T == []: # drop
                    grasping_pose.append([n, 0.0, 0.0, 0.0, 0.0, False])
                else:    # pick up
                    gps = self.sample_gps
                    gps_transformed = np.array(self.transform_grasping_pose(gps, blk_ang, T))

                    # 1. Select the gp group depending on block angle
                    # 2. Choose the farthest point from the peg among the three points on the same side
                    group = gps_transformed[(gps_transformed[:,0]>=-90) & (gps_transformed[:,0]<=90)]
                    dist = np.linalg.norm(group[:,1:3] - peg_points[n][:2], axis=1)
                    argmax = np.argmax(dist)
                    grasping_pose.append([n] + list(group[argmax]) + [True])
        self.pose_grasping = grasping_pose    # [peg number, grasping angle, x, y, z, seen]

    def find_placing_pose(self, peg_point):
        ang = self.sample_pps[self._pose_grasping_arg][0]
        pp = self.sample_pps[self._pose_grasping_arg][1:3]
        pp_rotated = ImgUtils.transform_pnts([pp[:2]], [.0, .0], -ang, .0, .0)
        pp_pad = np.pad(pp_rotated, ((0, 0), (0, 1)), mode='constant', constant_values=0.0)[0]
        pp_transformed = peg_point + pp_pad  # assuming the same z-coord of pp as peg_points
        placing_pose = [0.0] + list(pp_transformed)
        self.pose_placing = placing_pose   # [placing angle, x, y, z]

    # # keep orientation of the block
    # def find_placing_pose(self, peg_point):
    #     ang = self.sample_pps[self._pose_grasping_arg][0]
    #     pp = self.sample_pps[self._pose_grasping_arg][1:3]
    #     pp_rotated = ImgUtils.transform_pnts([pp[:2]], [.0, .0], -ang, .0, .0)
    #     pp_rotated = ImgUtils.transform_pnts(pp_rotated, [.0, .0], self._pose_grasping[0], .0, .0)
    #     pp_pad = np.pad(pp_rotated, ((0, 0), (0, 1)), mode='constant', constant_values=0.0)[0]
    #     pp_transformed = peg_point + pp_pad  # assuming the same z-coord of pp as peg_points
    #     placing_pose = [self._pose_grasping[0]] + list(pp_transformed)
    #     self.pose_placing = placing_pose   # [placing angle, x, y, z]

    @classmethod
    def find_placing_pose_all(cls, num_pickup, num_place, pose_blks, grasping_pose, peg_points):
        # inverse-transform grasping point
        T = pose_blks[num_pickup][2]
        R = T[:3, :3]
        t = T[:3, -1]
        R_inv = R.T
        t_inv = -R.T.dot(t)
        grasping_point = grasping_pose[num_pickup][2:5]
        gp_inv = R_inv.dot(grasping_point) + t_inv

        # transform placing point
        pp = ImgUtils.transform_pnts([gp_inv[:2]], [.0, .0], 30.0, .0, .0)
        pp = np.pad(pp, ((0, 0), (0, 1)), constant_values=0.0)[0]
        pp_transformed = peg_points[num_place] + pp    # assuming the same z-coord of pp as peg_points
        placing_pose = [num_place, 0.0] + list(pp_transformed) + [False]
        return placing_pose