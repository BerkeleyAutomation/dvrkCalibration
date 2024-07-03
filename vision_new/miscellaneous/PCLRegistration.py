# from probreg import features
# from probreg import callbacks
# from probreg import l2dist_regs
import open3d as o3d
import open3d.cpu.pybind.pipelines.registration as o3d_reg

import numpy as np
import copy

class PCLRegistration:
    # cv = lambda x: np.asarray(x.points if isinstance(x, o3d.geometry.PointCloud) else x)
    @classmethod
    def convert(cls, array):
        if isinstance(array, o3d.geometry.PointCloud):
            pcl = copy.deepcopy(array)
            pnt = np.asarray(array.points)
        else:
            pcl = o3d.geometry.PointCloud()
            pcl.points = o3d.utility.Vector3dVector(array)
            pnt = copy.deepcopy(array)
        return pcl, pnt

    @classmethod
    def remove_outliers(cls, array, nb_points, radius, visualize=False):
        pcl, pnt = PCLRegistration.convert(array)

        # Downsample the point cloud with a voxel of 0.02
        # voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)

        # Every 5th points are selected
        # uni_down_pcd = pcd.uniform_down_sample(every_k_points=5)

        # Statistical oulier removal
        # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=30, std_ratio=0.2)
        # display_inlier_outlier(pcd, ind)

        # Radius oulier removal
        cloud, ind = pcl.remove_radius_outlier(nb_points, radius)
        if visualize:
            PCLRegistration.display_inlier_outlier(pcl, ind)
        return pcl.select_by_index(ind), ind

    @classmethod
    def segment_surface(cls, array, dist_threshold, ransac_n=10, num_iterations=500, visualize=False):
        pcl, pnt = PCLRegistration.convert(array)

        coefficients, inliers = pcl.segment_plane(
            distance_threshold=dist_threshold, ransac_n=ransac_n, num_iterations=num_iterations)
        # coefficients = [a,b,c,d], where ax+by+cz+d = 0 (plane eqn.)

        inlier = pcl.select_by_index(inliers)
        outlier = pcl.select_by_index(inliers, invert=True)
        if visualize:
            inlier.paint_uniform_color([0.8, 0.8, 0.8])  # inlier in gray color
            outlier.paint_uniform_color([1.0, 0.0, 0.0])  # outlier in red color
            o3d.visualization.draw_geometries([inlier, outlier])
        return inlier, outlier, coefficients

    @classmethod
    def registration(cls, source, target, downsample=1, use_svr=False, save_image=False, visualize=False):
        source_down = source.uniform_down_sample(every_k_points=downsample)
        target_down = target.uniform_down_sample(every_k_points=downsample)

        source, _ = PCLRegistration.convert(source_down)
        target, _ = PCLRegistration.convert(target_down)

        src = copy.deepcopy(source)
        tgt = copy.deepcopy(target)
        T = np.identity(4)

        # coarse registration by mean matching
        pnt_src = np.asarray(src.points)
        pnt_tgt = np.asarray(tgt.points)
        mean_source = pnt_src.mean(axis=0)
        mean_target = pnt_tgt.mean(axis=0)
        t = mean_target - mean_source
        T_temp = np.identity(4)
        T_temp[:3, -1] = t
        src.transform(T_temp)
        T = copy.deepcopy(T_temp).dot(T)

        if use_svr:
            # svr registration
            tf_param = l2dist_regs.registration_svr(src, tgt)
            T_temp[:3, :3] = tf_param.rot
            T_temp[:3, -1] = tf_param.t
            src.transform(T_temp)
            T = copy.deepcopy(T_temp).dot(T)

        if visualize:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            src.paint_uniform_color([1, 0, 0])
            tgt.paint_uniform_color([0.8, 0.8, 0.8])
            vis.add_geometry(src)
            vis.add_geometry(tgt)
            vis.get_view_control().rotate(1000.0, 150)
            for _ in range(10):
                vis.get_view_control().scale(0.8)
            icp_iteration = 100
            threshold = [20, 10, 5, 2, 1, 0.5, 0.2]
            for i in range(len(threshold)):
                for j in range(icp_iteration):
                    reg_p2p = o3d_reg.registration_icp(src, tgt, threshold[i], np.identity(4),
                                                                o3d_reg.TransformationEstimationPointToPoint(),
                                                                o3d_reg.ICPConvergenceCriteria(max_iteration=1))
                    src.transform(reg_p2p.transformation)
                    T = copy.deepcopy(reg_p2p.transformation).dot(T)
                    vis.update_geometry(src)
                    vis.update_geometry(tgt)
                    vis.poll_events()
                    vis.update_renderer()
                    if save_image:
                        vis.capture_screen_image("image_%04d.jpg" % j)
            vis.run()
        else:
            # Point-to-point ICP
            threshold = [20, 10, 5, 2, 1, 0.5, 0.2]
            for i in range(len(threshold)):
                reg_p2p = o3d_reg.registration_icp(src, tgt, threshold[i], np.identity(4), o3d_reg.TransformationEstimationPointToPoint())
                # reg_p2p = o3d.registration.registration_icp(src, tgt, threshold[i], np.identity(4), o3d.registration.TransformationEstimationPointToPoint())
                src.transform(reg_p2p.transformation)
                T = copy.deepcopy(reg_p2p.transformation).dot(T)
        return T

    @classmethod
    def display_inlier_outlier(cls, array, ind):
        pcl, pnt = PCLRegistration.convert(array)

        inlier_cloud = pcl.select_by_index(ind)
        outlier_cloud = pcl.select_by_index(ind, invert=True)

        print("Showing outliers (red) and inliers (gray): ")
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
        o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

    @classmethod
    def display_registration(cls, source, target, transformation):
        source, _ = PCLRegistration.convert(source)
        target, _ = PCLRegistration.convert(target)
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        o3d.visualization.draw_geometries([source_temp, target_temp])