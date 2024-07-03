from dvrk.utils.ImgUtils import ImgUtils
import numpy as np


class GraspingPose():
    def __init__(self, BlockDetection):
        self.mask_dx = BlockDetection.mask_dx
        self.mask_dy = BlockDetection.mask_dy
        self.sample_gps = self.load_grasping_point(gp_number=3, dist_center=15, dist_gp=2)  # sample grasping points
        self.sample_pps = self.load_grasping_point(gp_number=3, dist_center=15, dist_gp=2)

    def load_grasping_point(self, gp_number, dist_center, dist_gp):
        group1 = []
        for i in range(gp_number):
            x = -(gp_number-1)*dist_gp//2
            y = dist_center
            group1.append([x+dist_gp*i+self.mask_dx//2, y+self.mask_dy//2])
        group0 = ImgUtils.transform_pnts(group1, [self.mask_dx//2, self.mask_dy//2], -120, 0, 0)
        group2 = ImgUtils.transform_pnts(group1, [self.mask_dx//2, self.mask_dy//2], 120, 0, 0)
        sample_grasping_point = np.reshape([group0, group1, group2], (gp_number*3,2))
        return sample_grasping_point

    def get_sample_grasping_pose(self, angle_rotated, x, y, which_sample='pick'):
        if which_sample=='pick':
            sample_points = self.sample_gps
        elif which_sample=='place':
            sample_points = self.sample_pps
        theta = angle_rotated
        gp_rotated = np.array(ImgUtils.transform_pnts(sample_points, (self.mask_dx//2, self.mask_dy//2), theta, 0, 0))
        if theta > 0:
            grasping_angle_rotated = [-30 + theta, -30 + theta, -30 + theta,
                                      -90 + theta, -90 + theta, -90 + theta,
                                      30 + theta, 30 + theta, 30 + theta]
        else:
            grasping_angle_rotated = [-30 + theta, -30 + theta, -30 + theta,
                                      90 + theta, 90 + theta, 90 + theta,
                                      30 + theta, 30 + theta, 30 + theta]
            # [gp number, gp_angle, tx, ty]
        return [[i, ga, gp[0]+x, gp[1]+y] for i,(ga,gp) in enumerate(zip(grasping_angle_rotated, gp_rotated))]

    def find_all_grasping_pose(self, pose_blks):
        all_grasping_pose = []
        for res in pose_blks:
            n, theta, x, y, seen = res
            gp_rotated = np.array(self.get_sample_grasping_pose(theta, x, y, 'pick'))
            for gp in gp_rotated:
                all_grasping_pose.append([n, gp[0], gp[1], gp[2], gp[3], seen])
        return all_grasping_pose  # [peg number, gp number, theta, x, y, seen]

    # theta is positive in counter-clockwise direction
    def find_grasping_pose(self, pose_blks, peg_points, which_side):
        grasping_pose = []
        pixel_coord = []
        for res in pose_blks:
            if res == []:
                grasping_pose.append([])
                pixel_coord.append([])
            else:
                n, ang, x, y, depth, seen = res
                if seen == True:    # pick up
                    # [gp number, gp_angle, tx, ty]
                    gp_rotated = np.array(self.get_sample_grasping_pose(ang, x, y, 'pick')).reshape(3,3,4)
                    gp_rotated_middle = gp_rotated[:,1,:]

                    # Choose the nearest point from the end effector
                    # a = 1.5  # slope of the line
                    if which_side == 'right_arm':
                        x_gp = gp_rotated_middle[:, 2]
                        arg = np.argsort(x_gp)[-2:]  # find maximum x
                        # k = gp_rotated_middle[:,3] - a * gp_rotated_middle[:,2]  # y = a*x + k
                        # arg = np.argmin(k)
                    elif which_side == 'left_arm':
                        x_gp = gp_rotated_middle[:, 2]
                        arg = np.argsort(x_gp)[:2]  # find minimum x
                        # k = gp_rotated_middle[:, 3] + a * gp_rotated_middle[:, 2]  # y = -a*x + k
                        # arg = np.argmin(k)

                    # Choose the farthest point from the peg among the three points on the same side
                    dist1 = np.linalg.norm(gp_rotated[arg[0]][:, 2:] - peg_points[n], axis=1)
                    dist2 = np.linalg.norm(gp_rotated[arg[1]][:, 2:] - peg_points[n], axis=1)
                    if max(dist1) > max(dist2):
                        arg = arg[0]
                        argmax = np.argmax(dist1)
                    else:
                        arg = arg[1]
                        argmax = np.argmax(dist2)
                else:   # drop
                    # [gp number, gp_angle, tx, ty]
                    gp_rotated = np.array(self.get_sample_grasping_pose(ang, x, y, 'place')).reshape(3, 3, 4)
                    # gp_rotated_middle = gp_rotated[:, 1, :]

                theta = gp_rotated[arg][argmax][1]      # grasping angle
                x = gp_rotated[arg][argmax][2]    # grasping position x
                y = gp_rotated[arg][argmax][3]    # grasping position y
                grasping_pose.append([n, theta, x, y, depth, seen])
        return grasping_pose    # [peg number, theta, x, y, depth, seen]