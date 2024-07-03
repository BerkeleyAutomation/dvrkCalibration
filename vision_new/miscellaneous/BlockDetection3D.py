import cv2
from dvrk.vision.BlockDetection2D import BlockDetection2D
from dvrk.vision.PCLRegistration import PCLRegistration
import dvrk.utils.CmnUtil as U
from FLSpegtransfer.path import *
import numpy as np
import open3d as o3d
import time


class BlockDetection3D:
    def __init__(self, Tpc):
        # parameters
        self.depth_peg = [-23, -18]     # depth for peg masking
        self.depth_block = [-17.5, -10]  # depth for block masking
        self.depth_block_servo = [-60, -30]
        self.lower_red = np.array([0 - 20, 60, 40])  # color range for masking
        self.upper_red = np.array([0 + 20, 255, 255])
        self.d_cr_max = 20  # crop diameter around peg point
        self.d_cr_min = 5
        self.Tpc = Tpc

        # detection result
        self._pnt_pegs = []
        self.pnt_blks = []
        self.pnt_masks = []     # mask transformed (for visualization)
        self._pose_blks = []    # pose of all blocks

        # load models
        self.pcl_model = o3d.io.read_point_cloud(root+'img/block_top.pcd')
        self.pcl_model_servo = o3d.io.read_point_cloud(root + 'img/block_cr.pcd')
        self.pnt_model = np.asarray(self.pcl_model.points)
        self.pnt_model_servo = np.asarray(self.pcl_model_servo.points)

    @property
    def pnt_pegs(self):
        return self._pnt_pegs

    @pnt_pegs.setter
    def pnt_pegs(self, value):
        self._pnt_pegs = value
        print("pnt_pegs updated")

    @property
    def pose_blks(self):
        return self._pose_blks

    @pose_blks.setter
    def pose_blks(self, value):
        self._pose_blks = value
        print ("pose_blks updated")

    @classmethod
    def remove_nan(cls, img_point):
        img_point = np.reshape(img_point, (-1, 3))
        img_point = img_point[~np.isnan(img_point).any(axis=1)]
        return img_point

    @classmethod
    def mask_color(cls, img_color, img_point, hsv_range):
        img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
        masked = cv2.inRange(img_hsv, hsv_range[0], hsv_range[1])
        return img_point[masked==255]

    @classmethod
    def mask_depth(cls, points, depth_range):
        # masked = points[(points[:, :, 2] > depth_range[0]) & (points[:, :, 2] < depth_range[1])]
        masked = points[(points[:, 2] > depth_range[0]) & (points[:, 2] < depth_range[1])]
        return masked

    @classmethod
    def mask_around_peg(cls, points, pnt_peg):
        masked = points[(points[:, 0] > pnt_peg[0]-10) & (points[:, 0] < pnt_peg[0]+10) & (points[:, 1] > pnt_peg[1]-10) & (points[:, 1] < pnt_peg[1]+10)]
        return masked

    def cluster_pegs(self, pnt_pegs, density, min_size, visualize=False):
        # numpy to open3d.PointCloud
        pcl_pegs = o3d.geometry.PointCloud()
        pcl_pegs.points = o3d.utility.Vector3dVector(pnt_pegs)
        if visualize:
            o3d.visualization.draw_geometries([pcl_pegs])

        # clustering
        labels = np.array(pcl_pegs.cluster_dbscan(eps=density, min_points=min_size, print_progress=False))
        pnt_pegs = np.asarray(pcl_pegs.points)
        clustered = [list(np.average(pnt_pegs[labels==i], axis=0)) for i in range(labels.max()+1)]
        return clustered

    def cluster_block(self, pnt_blk, pnt_peg, d_cr_min, d_cr_max):
        x = pnt_blk[:,0] - pnt_peg[0]
        y = pnt_blk[:,1] - pnt_peg[1]
        arg = (x*x + y*y > d_cr_min/2*d_cr_min/2) & (x*x + y*y < d_cr_max/2*d_cr_max/2)
        if np.count_nonzero(arg) > 300:
            clustered = pnt_blk[arg]
        else:
            clustered = []
        return clustered

    def cluster_blocks(self, pnt_blks, pnt_pegs, d_cr_min, d_cr_max):
        clustered = []
        for p in pnt_pegs:
            x = pnt_blks[:,0] - p[0]
            y = pnt_blks[:,1] - p[1]
            arg = (x*x + y*y > d_cr_min/2*d_cr_min/2) & (x*x + y*y < d_cr_max/2*d_cr_max/2)
            if np.count_nonzero(arg) > 2000:
                clustered.append(pnt_blks[arg])
            else:
                clustered.append([])
        return clustered

    # remove outliers & prune by using segmentation & clustering
    def prune_block(self, pnt_blk, visualize=False):
        if pnt_blk == []:
            pruned = []
        else:
            plane, outlier, coefficient = PCLRegistration.segment_surface(pnt_blk, dist_threshold=0.5, visualize=visualize)
            labels = np.array(plane.cluster_dbscan(eps=1, min_points=30, print_progress=False))
            clustered = np.asarray(plane.points)
            arg = np.histogram(labels, [-1,0,1,2,3,4,5])[0].argmax()
            pruned = clustered[labels == arg-1]
        return pruned

    # remove outliers & prune by using segmentation & clustering
    def prune_block_servo(self, pnt_blk, visualize=False):
        pruned = o3d.geometry.PointCloud()
        for i in range(2):
            plane, outlier = PCLRegistration.segment_surface(pnt_blk, dist_threshold=0.5, visualize=visualize)
            if len(plane.points) > 500:
                labels = np.array(plane.cluster_dbscan(eps=1, min_points=40, print_progress=False))
                clustered = np.asarray(plane.points)
                arg = np.histogram(labels, [-1,0,1,2,3,4,5])[0].argmax()
                pruned = pruned + clustered[labels == arg-1]
                pnt_blk = outlier
        return pruned

    # remove outliers & prune by using segmentation & clustering
    def prune_blocks(self, pnt_blks, visualize=False):
        pruned = []
        for p in pnt_blks:
            if p == []:
                pruned.append([])
            else:
                plane, outlier, coefficient = PCLRegistration.segment_surface(p, dist_threshold=0.5, visualize=visualize)
                labels = np.array(plane.cluster_dbscan(eps=1, min_points=30, print_progress=False))
                clustered = np.asarray(plane.points)
                arg = np.histogram(labels, [-1,0,1,2,3,4,5])[0].argmax()
                clustered = clustered[labels == arg-1]
                pruned.append(clustered)
        return pruned

    def get_pose(self, transform):
        if transform == []:
            pose = []
        else:
            R = transform[:3, :3]
            euler = np.rad2deg(U.R_to_euler(R))
            pose = euler[0]
        return pose     # block angle

    def get_poses(self, transform):
        pose = []
        for n, T in enumerate(transform):
            if T == []:
                pose.append([n, 0.0, []])
            else:
                R = T[:3,:3]
                euler = np.rad2deg(U.R_to_euler(R))
                pose.append([n, euler[2], T])
        return pose # [nb_block, euler angle, T]

    # find coordinates of the clustered pegs
    def find_pegs(self, img_color, img_point, visualize=False):
        # color masking & transform points to pegboard coordinate
        pnt_masked = self.mask_color(img_color, img_point, [self.lower_red, self.upper_red])
        pnt_masked = self.remove_nan(pnt_masked)
        pnt_transformed = U.transform(pnt_masked*0.001, self.Tpc)*1000


        # depth masking
        pnt_pegs = self.mask_depth(pnt_transformed, self.depth_peg)
        clustered = self.cluster_pegs(pnt_pegs, density=3, min_size=30, visualize=visualize)
        self.pnt_pegs = BlockDetection2D.sort_pegs(clustered)  # [[x0, y0, z0], ..., [xn, yn, zn]]

    # find block, given block number
    def find_block(self, block_number, img_color, img_point, which_arm=None):
        # color masking & transform points to pegboard coordinate
        pnt_masked = self.mask_color(img_color, img_point, [self.lower_red, self.upper_red])
        pnt_masked = self.remove_nan(pnt_masked)
        pnt_transformed = U.transform(pnt_masked*0.001, self.Tpc)*1000

        # block image masking by depth
        pnt_blks = self.mask_depth(pnt_transformed, self.depth_block)

        # cluster blocks by peg position
        pnt_blk = self.cluster_block(pnt_blks, self.pnt_pegs[block_number], self.d_cr_min, self.d_cr_max)

        # prune block points by segmenting two planes
        pnt_blk = self.prune_block(pnt_blk)

        # registration
        if pnt_blk == []:
            pose_blk = []
            pnt_mask = []
        else:
            pcl_blk = o3d.geometry.PointCloud()
            pcl_blk.points = o3d.utility.Vector3dVector(pnt_blk)
            st = time.time()
            T = PCLRegistration.registration(pcl_blk, self.pcl_model, downsample=2, use_svr=False, save_image=False, visualize=False)
            print(time.time() - st)
            T = np.linalg.inv(T)    # transform from model to block
            blk_ang = self.get_pose(T)
            pose_blk = [blk_ang, T]
            pnt_mask = T[:3, :3].dot(self.pnt_model.T).T + T[:3, -1].T  # transformed mask points
        return pose_blk, pnt_blk, pnt_mask

    def find_block_servo(self, img_color, img_point, pnt_peg, which_arm=None):
        # color masking & transform points to pegboard coordinate
        pnt_masked = self.mask_color(img_color, img_point, [self.lower_red, self.upper_red])
        pnt_masked = self.remove_nan(pnt_masked)
        pnt_transformed = U.transform(pnt_masked*0.001, self.Tpc)*1000
        # block image masking by depth
        pnt_depth_masked = self.mask_depth(pnt_transformed, self.depth_block_servo)
        pnt_blk = self.mask_around_peg(pnt_depth_masked, pnt_peg)

        # remove outliers
        # pnt_blk, _ = PCLRegistration.remove_outliers(pnt_blk, nb_points=20, radius=2, visualize=False)
        pcl_blk, pnt_blk = PCLRegistration.convert(pnt_blk)
        # registration
        if len(pnt_blk) < 100:
            pose_blk = []
            pnt_mask = []
        else:
            T = PCLRegistration.registration(pcl_blk, self.pcl_model_servo, downsample=2, use_svr=False, save_image=False, visualize=False)
            T = np.linalg.inv(T)    # transform from model to block
            blk_ang = self.get_pose(T)
            pose_blk = [blk_ang, T]
            pnt_mask = T[:3, :3].dot(self.pnt_model_servo.T).T + T[:3, -1].T  # transformed mask points
        return pose_blk, pnt_blk, pnt_mask

    def find_block_servo_handover(self, img_color, img_point):
        # color masking & transform points to pegboard coordinate
        pnt_masked = self.mask_color(img_color, img_point, [self.lower_red, self.upper_red])
        pnt_masked = self.remove_nan(pnt_masked)
        pnt_transformed = U.transform(pnt_masked*0.001, self.Tpc)*1000
        # block image masking by depth
        pnt_blk = self.mask_depth(pnt_transformed, self.depth_block_servo)

        # remove outliers
        # pnt_blk, _ = PCLRegistration.remove_outliers(pnt_blk, nb_points=20, radius=2, visualize=False)
        pcl_blk, pnt_blk = PCLRegistration.convert(pnt_blk)
        # registration
        if len(pnt_blk) < 100:
            pose_blk = []
            pnt_mask = []
        else:
            T = PCLRegistration.registration(pcl_blk, self.pcl_model_servo, downsample=2, use_svr=False, save_image=False, visualize=False)
            T = np.linalg.inv(T)    # transform from model to block
            blk_ang = self.get_pose(T)
            pose_blk = [blk_ang, T]
            pnt_mask = T[:3, :3].dot(self.pnt_model_servo.T).T + T[:3, -1].T  # transformed mask points
        return pose_blk, pnt_blk, pnt_mask

    def find_block_all(self, img_color, img_point):
        # color masking & transform points to pegboard coordinate
        pnt_masked = self.mask_color(img_color, img_point, [self.lower_red, self.upper_red])
        pnt_masked = self.remove_nan(pnt_masked)
        pnt_transformed = U.transform(pnt_masked*0.001, self.Tpc)*1000

        # block image masking by depth
        pnt_blks = self.mask_depth(pnt_transformed, self.depth_block)

        # cluster blocks by peg position
        pnt_blks = self.cluster_blocks(pnt_blks, self.pnt_pegs, self.d_cr_min, self.d_cr_max)

        # prune block points by segmenting two planes
        self.pnt_blks = self.prune_blocks(pnt_blks)

        # registration
        Ts = []
        for b in self.pnt_blks:
            if b == []:
                Ts.append([])
            else:
                pcl_blks = o3d.geometry.PointCloud()
                pcl_blks.points = o3d.utility.Vector3dVector(b)
                T = PCLRegistration.registration(pcl_blks, self.pcl_model, use_svr=False, save_image=False, visualize=False)
                Ts.append(np.linalg.inv(T))

        self.pose_blks = self.get_poses(Ts)   # [n, angle, T]
        self.pnt_masks = [T[:3, :3].dot(self.pnt_model.T).T + T[:3,-1].T if T!=[] else [] for T in Ts]