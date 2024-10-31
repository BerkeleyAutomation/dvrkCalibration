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
        elif self.driver_offset_input == "2":
            self.dvrk_type = dvrkTypes.LARGE_NEEDLE_DRIVER
        else:
            print("Please select 1 or 2")
            exit()
        root_path = os.path.dirname(os.path.abspath(__file__))
        self.calibration_output_path = os.path.join(root_path, "experiment/0_trajectory_extraction/calibration_outputs")
        self.robot_to_cam_ = np.eye(4)
        if know_transform:
            import pdb

            pdb.set_trace()
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
        filename = os.path.join(self.calibration_output_path, "prime_psm" + self.psm_number + "_random_sampled.npy")
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

    def initial_setup(self):
        initial_pos = np.array([0, 0, -0.13])
        initial_euler = np.array([0, 0, 0])
        initial_quat = np.array([0, 0, 0, 1])
        initial_pose = initial_pos, initial_quat
        self.dvrk.set_pose(*initial_pose)
        self.dvrk.set_jaw(JAW_OPEN_ANGLE)
        time.sleep(0.1)
        input("Press enter to start")
        self.dvrk.set_jaw(JAW_CLOSE_ANGLE)
        time.sleep(1)

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
        self.initial_setup()

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
                pitch_to_yaw = 0.0091
                yaw_to_control_point = 0.0102

                # TODO: Check which needle driver you are using, if you are using suturecut driver or needle offset driver
                offset = self.dvrk.shallow_calibration_offset + (small_ball_radius / 2)
                ee_point = small_ball_center + (offset * normalized_direction)
                ee_point = ee_point.reshape(-1, 1)
                pixel, _ = cv2.projectPoints(
                    objectPoints=ee_point,
                    rvec=np.array([[0.0, 0.0, 0.0]]),
                    tvec=np.array([[0.0, 0.0, 0.0]]),
                    cameraMatrix=intrinsics_matrix,
                    distCoeffs=distortion_coefficients,
                )
                pixel = np.round(pixel.squeeze()).astype(int)
                img_color = cv2.circle(img_color, (pixel[0], pixel[1]), 3, (0, 255, 0), -1)
                pt = ee_point
                pos_des_temp, _ = self.dvrk.get_current_pose()
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
                cv2.waitKey(1000) & 0xFF
                i += 1
            else:
                print("Bad reading")

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
