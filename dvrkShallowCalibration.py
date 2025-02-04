import sys

for p in sys.path:
    if p == "/opt/ros/kinetic/lib/python2.7/dist-packages":
        sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from vision_new.cameras.Camera import Camera
from vision_new import vision_constants as cst
from vision.ZividCaptureNew import ZividCapture
from vision.BallDetection import BallDetection
from dvrk.motion.dvrkArm import dvrkArm
from dvrk.motion.dvrkTypes import dvrkTypes
from dvrk.motion.dvrkKinematics import dvrkKinematics
import utils.CmnUtil as U
import os
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import torch

root = "/home/davinci/dvrkCalibration"
JAW_OPEN_ANGLE = [np.pi / 2]
JAW_CLOSE_ANGLE = [-0.3]  # Angle for closing jaw to grasp


class dvrkCalibration:
    def __init__(self, know_transform):
        self.psm_number = input("Which PSM are you calibrating. Pick 1 or 2 ")
        psm_string = ""
        if self.psm_number == "1":
            psm_string = "/PSM1"
        elif self.psm_number == "2":
            psm_string = "/PSM2"
        else:
            print("Please select 1 or 2")
            exit()
        self.dvrk_type_input = input(
            "Which PSM driver type are you calibrating. Large SutureCut Needle Driver (1) or Large Needle Driver (2)"
        )
        self.dvrk_type = None

        if self.dvrk_type_input == "1":
            self.dvrk_type = dvrkTypes.LRG_SUTURECUT_NEEDLE_DRIVER
        elif self.dvrk_type_input == "2":
            self.dvrk_type = dvrkTypes.LARGE_NEEDLE_DRIVER
        else:
            print("Please select 1 or 2")
            exit()
        root_path = os.path.dirname(os.path.abspath(__file__))
        # self.calibration_output_path = os.path.join(
        #     root_path, "experiment/0_trajectory_extraction/shallow_and_deep_calibration_outputs"
        # )

        self.calibration_output_path = os.path.join(
            root_path, "experiment/0_trajectory_extraction/shallow_and_deep_calibration_outputs"
        )

        self.robot_to_cam_ = np.eye(4)
        if know_transform:
            if self.psm_number == "1":
                psm_string = "/PSM1"
                self.robot_to_cam_ = np.load(
                    "/home/davinci/automated_suturing/surgical_suturing_catkin_ws/src/calibration_config/cam_to_robot/Trc_inclined_PSM1.npy"
                )
            elif self.psm_number == "2":
                psm_string = "/PSM2"
                self.robot_to_cam_ = np.load("/home/davinci/dvrkCalibration/data/zivid_to_psm2.npy")
            else:
                print("Please select 1 or 2")
                exit()

        # objects
        self.dvrk = dvrkArm(psm_string, dvrk_type=self.dvrk_type, use_rnn=False)
        self.zivid = Camera(cst.ZIVID, zivid_capture_type="3d")
        self.BD = BallDetection(self.robot_to_cam_)

        # Load trajectory
        # filename = root + 'experiment/0_trajectory_extraction/verification_traj_random_sampling_10000.npy'
        filename = os.path.join(
            self.calibration_output_path, "prime_psm" + self.psm_number + "_shallow_random_sampled.npy"
        )
        self.joint_traj = self.load_trajectory(filename)

    def load_trajectory(self, filename):
        joint = np.load(filename)
        pos = np.array(
            [self.BD.fk_position(q[0], q[1], q[2], 0, 0, 0, L1=self.BD.L1, L2=self.BD.L2, L3=0, L4=0) for q in joint]
        )
        q1 = joint[:, 0]
        q2 = joint[:, 1]
        q3 = joint[:, 2]
        q4 = joint[:, 3]
        q5 = joint[:, 4]
        q6 = joint[:, 5]

        # Create 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        plt.plot(pos[:, 0], pos[:, 1], pos[:, 2], "b.-")
        ax.set_xlabel("X Label")
        ax.set_ylabel("Y Label")
        ax.set_zlabel("Z Label")
        print("data length: ", len(joint))
        plt.show()

        # Create 2D plot for joint angles
        plt.subplot(611)
        plt.plot(q1 * 180.0 / np.pi, "b-")
        plt.ylabel("q1 ($^\circ$)")
        plt.subplot(612)
        plt.plot(q2 * 180.0 / np.pi, "b-")
        plt.ylabel("q2 ($^\circ$)")
        plt.subplot(613)
        plt.plot(q3, "b-")
        plt.ylabel("q3 (mm)")
        plt.subplot(614)
        plt.plot(q4 * 180.0 / np.pi, "b-")
        plt.ylabel("q4 ($^\circ$)")
        plt.subplot(615)
        plt.plot(q5 * 180.0 / np.pi, "b-")
        plt.ylabel("q5 ($^\circ$)")
        plt.subplot(616)
        plt.plot(q6 * 180.0 / np.pi, "b-")
        plt.ylabel("q6 ($^\circ$)")
        plt.xlabel("(step)")
        plt.show()
        return joint

    def exp0_get_transform(self):
        jaw1 = [5.0 * np.pi / 180.0]
        self.dvrk.set_jaw(jaw=jaw1)
        j1 = self.joint_traj[:, 0]
        j2 = self.joint_traj[:, 1]
        j3 = self.joint_traj[:, 2]
        j4 = np.zeros_like(j1)
        j5 = np.zeros_like(j1)
        j6 = np.zeros_like(j1)
        self.collect_data_joint(j1, j2, j3, j4, j5, j6, transform="unknown")

    def exp1_move_all_joints(self):
        jaw1 = [5.0 * np.pi / 180.0]
        self.dvrk.set_jaw(jaw=jaw1)
        j1 = self.joint_traj[:, 0]
        j2 = self.joint_traj[:, 1]
        j3 = self.joint_traj[:, 2]
        j4 = self.joint_traj[:, 3]
        j5 = self.joint_traj[:, 4]
        j6 = self.joint_traj[:, 5]
        self.collect_data_joint(j1, j2, j3, j4, j5, j6, transform="known")

    def exp2_move_q4_q5_q6(self):
        jaw1 = [5.0 * np.pi / 180.0]
        self.dvrk.set_pose(jaw1=jaw1)
        j4 = self.joint_traj[:, 3]
        j5 = self.joint_traj[:, 4]
        j6 = self.joint_traj[:, 5]
        j1 = np.ones_like(j4) * self.dvrk.act_joint1[0]
        j2 = np.ones_like(j4) * self.dvrk.act_joint1[1]
        j3 = np.ones_like(j4) * self.dvrk.act_joint1[2]
        self.collect_data_joint(j1, j2, j3, j4, j5, j6, transform="known")

    def exp3_move_q4_only(self):
        jaw1 = [5.0 * np.pi / 180.0]
        self.dvrk.set_pose(jaw1=jaw1)
        j4 = self.joint_traj[:, 3]
        j5 = np.zeros_like(j4)
        j6 = np.zeros_like(j4)
        j1 = np.ones_like(j4) * self.dvrk.act_joint1[0]
        j2 = np.ones_like(j4) * self.dvrk.act_joint1[1]
        j3 = np.ones_like(j4) * self.dvrk.act_joint1[2]
        self.collect_data_joint(j1, j2, j3, j4, j5, j6, transform="known")

    def exp4_move_q5_only(self):
        jaw1 = [5.0 * np.pi / 180.0]
        self.dvrk.set_pose(jaw1=jaw1)
        j5 = self.joint_traj[:, 4]
        j4 = np.zeros_like(j5)
        j6 = np.zeros_like(j5)
        j1 = np.ones_like(j5) * self.dvrk.act_joint1[0]
        j2 = np.ones_like(j5) * self.dvrk.act_joint1[1]
        j3 = np.ones_like(j5) * self.dvrk.act_joint1[2]
        self.collect_data_joint(j1, j2, j3, j4, j5, j6, transform="known")

    def exp5_move_q6_only(self):
        jaw1 = [5.0 * np.pi / 180.0]
        self.dvrk.set_pose(jaw1=jaw1)
        j6 = self.joint_traj[:, 5]
        j4 = np.zeros_like(j6)
        j5 = np.zeros_like(j6)
        j1 = np.ones_like(j6) * self.dvrk.act_joint1[0]
        j2 = np.ones_like(j6) * self.dvrk.act_joint1[1]
        j3 = np.ones_like(j6) * self.dvrk.act_joint1[2]
        self.collect_data_joint(j1, j2, j3, j4, j5, j6, transform="known")

    def click_event(self, event, u, v, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pixel_value = param["img"][v, u]
            print(f"Pixel coordinates: (u={u}, v={v}) - Pixel value: {pixel_value}")

            # Display the coordinates and pixel value on the image
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                param["img"],
                f"({u},{v})",
                (u, v),
                font,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                param["img"],
                str(pixel_value),
                (u, v + 20),
                font,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            time.sleep(0.5)
            param["u"] = u
            param["v"] = v

    def pixel_to_3d_line(self, u, v, camera_matrix, dist_coeffs):
        """
        Given pixel coordinates (u, v), the camera intrinsic matrix, and distortion coefficients,
        compute the 3D line that passes through the camera center and the pixel.
        """
        # Convert pixel coordinates to normalized undistorted coordinates
        pixel = np.array([[u, v]], dtype=np.float32)  # Shape: (1, 1, 2)
        undistorted = cv2.undistortPoints(pixel, camera_matrix, dist_coeffs)

        # Extract x, y from undistorted coordinates (z = 1.0 assumed for direction)
        x, y = undistorted[0][0]
        z = 1.0  # Assume unit depth

        # Ray direction in camera space (origin: [0,0,0])
        ray_dir = np.array([x, y, z])
        ray_dir /= np.linalg.norm(ray_dir)  # Normalize the direction

        return np.array([0, 0, 0]), ray_dir  # Camera origin and ray direction

    def closest_point_between_lines(self, origin1, dir1, origin2, dir2):
        # Normalize direction vectors
        dir1 = dir1 / np.linalg.norm(dir1)
        dir2 = dir2 / np.linalg.norm(dir2)

        # Solve for t and s
        A = np.column_stack((dir1, -dir2))  # 3x2 matrix
        b = origin2 - origin1  # 3x1 vector

        # Solve least-squares (minimizing error in case of skew lines)
        try:
            ts, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)
            t, s = ts  # Extract parameters
        except np.linalg.LinAlgError:
            return None  # If matrix is singular, no intersection

        # Compute the intersection points on each line
        point1 = origin1 + t * dir1
        point2 = origin2 + s * dir2
        distance = np.linalg.norm(point1 - point2)
        return point1, point2, distance

    def initial_setup(self, q3_height):
        self.dvrk.set_joint(np.array([0, 0, q3_height, 0, 0, 0]))
        time.sleep(1.0)
        self.dvrk.set_joint(np.array([0, 0, q3_height, 0, 0, 0]))
        time.sleep(0.1)
        curr_joints = self.dvrk.get_current_joint()
        q3_joint = curr_joints[2]
        measured_q3_joint = input("Manually measure the Q3 offset and input it in meters: ") 
        print("Debug:" + str(q3_joint))
        q3_offset = float(measured_q3_joint) - q3_joint
        got_red_fiducial_measurement = False
        while not got_red_fiducial_measurement:
            print("Press enter when you have moved q1,q2 such that the tip is visible to the PSM")
            while True:
                zivid_image, zivid_depth, zivid_pcl, intrinsics_matrix, distortion_coefficients = self.zivid.capture()
                cv2.imshow("Press enter when you have moved q1,q2 such that the tip is visible to the PSM", zivid_image)
                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # ASCII code for Enter key
                    break
            cv2.destroyAllWindows()

            img_color, img_depth, img_point = (
                zivid_image,
                zivid_depth,
                zivid_pcl,
            )  # self.BD.img_crop(zivid_image,zivid_depth,zivid_pcl)
            # img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
            img_color_org = np.copy(img_color)
            # Find balls
            pbs = self.BD.find_shallow_balls(img_color_org, img_depth, img_point)

            img_color = self.BD.overlay_balls(img_color, pbs, intrinsics_matrix, distortion_coefficients)
            cv2.imshow("Press y if the fiducial overlay is correct. Otherwise press any other key", img_color)
            key = cv2.waitKey(0) & 0xFF
            if key == ord("y"):  # ASCII code for Enter key
                got_red_fiducial_measurement = True
            else:
                got_red_fiducial_measurement = False
            cv2.destroyAllWindows()
        got_pixel_tip = False
        while not got_pixel_tip:
            params = {"img": img_color.copy(), "u": None, "v": None}
            cv2.imshow("Click the end effector tip", img_color)
            cv2.setMouseCallback("Click the end effector tip", self.click_event, param=params)

            # Wait until a key is pressed and the coordinates are set
            while params["u"] is None or params["v"] is None:
                if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
                    break

            # Retrieve the u, v pixel coordinates from the callback
            u, v = params["u"], params["v"]

            print(f"Selected coordinates: (u={u}, v={v})")
            cv2.destroyAllWindows()

            pixel_origin, pixel_ray_dir = self.pixel_to_3d_line(
                u=u, v=v, camera_matrix=intrinsics_matrix, dist_coeffs=distortion_coefficients
            )
            big_ball_center = np.array([pbs[0][0], pbs[0][1], pbs[0][2]]) / 1000
            big_ball_radius = pbs[0][3] / 1000

            small_ball_center = np.array([pbs[1][0], pbs[1][1], pbs[1][2]]) / 1000
            small_ball_radius = pbs[1][3] / 1000
            direction = small_ball_center - big_ball_center
            normalized_direction = direction / np.linalg.norm(direction)
            small_ball_origin, small_ball_ray_dir = small_ball_center, normalized_direction
            point1, point2, distance = self.closest_point_between_lines(
                pixel_origin, pixel_ray_dir, small_ball_origin, small_ball_ray_dir
            )
            correct_pixel_input = input(
                "Distance should be 0. It is actually " + str(distance * 1000) + " mm. Is that acceptable (y)"
            )
            if correct_pixel_input:
                got_pixel_tip = True
        pixel_tip_point = point2  # We pick the point actually on the line
        small_ball_to_tip_offset = np.linalg.norm(pixel_tip_point - small_ball_origin)
        return q3_offset, small_ball_to_tip_offset

    def optimize_transform(self, psm1_to_zivid_initial, pos_act, pos_des):
        initial_rotation_matrix = psm1_to_zivid_initial[:3, :3]
        initial_pos = psm1_to_zivid_initial[:3, 3]
        initial_quat = R.from_matrix(initial_rotation_matrix).as_quat()
        x0 = np.concatenate((initial_pos, initial_quat))

        def calibration_loss(x):
            pos = x[0:3]
            quat = x[3:]
            rotation_matrix = R.from_quat(quat).as_matrix()
            psm1_to_zivid = np.eye(4)
            psm1_to_zivid[:3, :3] = rotation_matrix
            psm1_to_zivid[:3, 3] = pos
            errors = []
            preds = []
            targets = []
            for psm_pos, cam_pos in zip(pos_des, pos_act):
                cam_pos_homogenous = np.append(cam_pos, 1).reshape(-1, 1)
                estimated_psm_pos_homogenous = psm1_to_zivid @ cam_pos_homogenous
                estimated_psm_pos = estimated_psm_pos_homogenous[:-1] / estimated_psm_pos_homogenous[-1]
                estimated_psm_pos = estimated_psm_pos.reshape(
                    -1,
                )
                preds.append(estimated_psm_pos)
                targets.append(psm_pos)
                error = np.linalg.norm(psm_pos - estimated_psm_pos)
                errors.append(error)
            np_errors = np.array(errors)
            torch_preds = torch.tensor(np.array(preds))
            torch_targets = torch.tensor(np.array(targets))
            torch_loss = torch.nn.MSELoss()
            loss_value = torch_loss(torch_preds, torch_targets).item()
            return loss_value

        res = minimize(
            calibration_loss,
            x0,
        )
        pos = res.x[0:3]
        quat = res.x[3:]
        rotation_matrix = R.from_quat(quat).as_matrix()
        psm1_to_zivid = np.eye(4)
        psm1_to_zivid[:3, :3] = rotation_matrix
        psm1_to_zivid[:3, 3] = pos
        return psm1_to_zivid

    def collect_data_joint(self, j1, j2, j3, j4, j5, j6, transform="known"):  # j1, ..., j6: joint trajectory
        q3_height = j3[0]
        q3_offset, small_ball_to_tip_offset = self.initial_setup(q3_height)
        input("Press enter when the other PSM is cleared out of the way")
        # try:
        time_st = time.time()  # (sec)
        time_stamp = []
        q_des = []
        q_act = []
        pos_des = []
        pos_act = []
        assert len(j1) == len(j2) == len(j3) == len(j4) == len(j5) == len(j6)
        i = 1
        for qd1, qd2, qd3, qd4, qd5, qd6 in zip(j1, j2, j3, j4, j5, j6):
            joint1 = [qd1, qd2, qd3, qd4, qd5, qd6]
            self.dvrk.set_jaw(JAW_CLOSE_ANGLE)
            self.dvrk.set_joint(joint=joint1)
            self.dvrk.set_jaw(JAW_CLOSE_ANGLE)
            time.sleep(1)
            # Capture image from Zivid
            zivid_image, zivid_depth, zivid_pcl, intrinsics_matrix, distortion_coefficients = self.zivid.capture()
            img_color, img_depth, img_point = (
                zivid_image,
                zivid_depth,
                zivid_pcl,
            )  # self.BD.img_crop(zivid_image,zivid_depth,zivid_pcl)
            # img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
            img_color_org = np.copy(img_color)
            # Find balls
            pbs = self.BD.find_shallow_balls(img_color_org, img_depth, img_point)

            img_color = self.BD.overlay_balls(img_color, pbs, intrinsics_matrix, distortion_coefficients)
            if len(pbs) >= 2:
                big_ball_center = np.array([pbs[0][0], pbs[0][1], pbs[0][2]]) / 1000
                big_ball_radius = pbs[0][3] / 1000

                small_ball_center = np.array([pbs[1][0], pbs[1][1], pbs[1][2]]) / 1000
                small_ball_radius = pbs[1][3] / 1000
                direction = small_ball_center - big_ball_center
                normalized_direction = direction / np.linalg.norm(direction)

                # shallow_offset: from tip to the lower boundary of the small ball
                ee_point = small_ball_center + (small_ball_to_tip_offset * normalized_direction)
                ee_point = ee_point.reshape(-1, 1)
                pixel, _ = cv2.projectPoints(
                    objectPoints=ee_point,
                    rvec=np.array([[0.0, 0.0, 0.0]]),
                    tvec=np.array([[0.0, 0.0, 0.0]]),
                    cameraMatrix=intrinsics_matrix,
                    distCoeffs=distortion_coefficients,
                )
                pixel = np.round(pixel.squeeze()).astype(int)
                img_color = cv2.circle(img_color, (pixel[0], pixel[1]), 5, (0, 255, 0), -1)
                pt = ee_point
                curr_joints = self.dvrk.get_current_joint()
                curr_joints[2] += q3_offset
                pos_des_temp, _ = dvrkKinematics.joint_to_pose(
                    curr_joints,
                    L1=self.dvrk.l_rcc_,
                    L2=self.dvrk.l_tool_,
                    L3=self.dvrk.l_pitch_2_yaw_,
                    L4=self.dvrk.l_yaw_2_ctrl_pnt_,
                )
                pos_des.append(pos_des_temp)
                pos_act.append(
                    pt.reshape(
                        -1,
                    )
                )
                print("index: ", len(pos_des), "/", len(j1))
                print("pos_des: ", pos_des_temp)
                print("pos_act: ", pt)
                print(" ")
                cv2.imshow("images", img_color)
                cv2.waitKey(1) & 0xFF
                i += 1
            else:
                print("Bad reading")
            # if i == 30:
            #     break

        np.save(os.path.join(self.calibration_output_path, "psm" + str(self.psm_number) + "_pos_des"), pos_des)
        np.save(os.path.join(self.calibration_output_path, "psm" + str(self.psm_number) + "_pos_act"), pos_act)
        psm1_to_zivid_initial = U.get_rigid_transform(np.array(pos_act), np.array(pos_des))
        T = self.optimize_transform(psm1_to_zivid_initial, pos_act, pos_des)

        np.save(os.path.join(self.calibration_output_path, "psm" + str(self.psm_number) + "_robot_to_zivid"), T)
        np.save(os.path.join(self.calibration_output_path, "psm" + str(self.psm_number) + "_t_stamp_raw"), time_stamp)
        print("Data is successfully saved")
        # finally:
        #     import pdb

        #     pdb.set_trace()
        #     # Save data to a file
        #     if transform == "known":
        #         np.save(os.path.join(self.calibration_output_path, "psm" + str(self.psm_number) + "_q_des_raw"), q_des)
        #         np.save(os.path.join(self.calibration_output_path, "psm" + str(self.psm_number) + "_q_act_raw"), q_act)
        #     elif transform == "unknown":
        #         # Get transform from robot to camera
        #         np.save(os.path.join(self.calibration_output_path, "psm" + str(self.psm_number) + "_pos_des"), pos_des)
        #         np.save(os.path.join(self.calibration_output_path, "psm" + str(self.psm_number) + "_pos_act"), pos_act)
        #         T = U.get_rigid_transform(np.array(pos_act), np.array(pos_des))
        #         np.save(os.path.join(self.calibration_output_path, "psm" + str(self.psm_number) + "_robot_to_zivid"), T)
        #     np.save(
        #         os.path.join(self.calibration_output_path, "psm" + str(self.psm_number) + "_t_stamp_raw"), time_stamp
        #     )
        #     print("Data is successfully saved")


if __name__ == "__main__":
    cal = dvrkCalibration(know_transform=False)
    cal.exp0_get_transform()
