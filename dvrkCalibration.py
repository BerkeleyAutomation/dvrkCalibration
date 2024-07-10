import sys
for p in sys.path:
    if p == '/opt/ros/kinetic/lib/python2.7/dist-packages':
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import pdb
pdb.set_trace()
from vision_new.cameras.Camera import Camera
from vision_new import vision_constants as cst
from vision.ZividCaptureNew import ZividCapture
from vision.BallDetection import BallDetection
from motion.dvrkMotionBridgeP import dvrkMotionBridgeP
from motion.dvrkArmNew import dvrkArm
import utils.CmnUtil as U
root = '/home/davinci/dvrkCalibration'
JAW_OPEN_ANGLE = [np.pi / 2]
JAW_CLOSE_ANGLE = [-0.3]  # Angle for closing jaw to grasp
class dvrkCalibration():
    def __init__(self):
        psm_number = input("Which PSM are you calibrating. Pick 1 or 2 ")
        psm_string = ''
        data_path = '/home/davinci/dvrkCalibration/data/'
        self.robot_to_cam_ = None
        if(psm_number == '1'):
            psm_string='/PSM1'
            self.robot_to_cam_ = np.load('/home/davinci/automated_suturing/surgical_suturing_catkin_ws/src/calibration_config/cam_to_robot/Trc_inclined_PSM1.npy')
        elif(psm_number == '2'):
            psm_string = '/PSM2'
            self.robot_to_cam_ = np.load('/home/davinci/dvrkCalibration/data/zivid_to_psm2.npy')
        else:
            print("Please select 1 or 2")
            exit()

        # objects
        self.dvrk = dvrkArm(psm_string)
        self.zivid = Camera(cst.ZIVID,zivid_capture_type='3d')
        self.BD = BallDetection(self.robot_to_cam_)

        # Load trajectory
        # filename = root + 'experiment/0_trajectory_extraction/verification_traj_random_sampling_10000.npy'
        filename = data_path + 'prime_psm' + psm_number + '_random_sampled.npy'
        self.joint_traj = self.load_trajectory(filename)

    def load_trajectory(self, filename):
        joint = np.load(filename)
        pos = np.array([self.BD.fk_position(q[0], q[1], q[2], 0, 0, 0, L1=self.BD.L1, L2=self.BD.L2, L3=0, L4=0) for q in joint])
        q1 = joint[:, 0]
        q2 = joint[:, 1]
        q3 = joint[:, 2]
        q4 = joint[:, 3]
        q5 = joint[:, 4]
        q6 = joint[:, 5]

        # Create 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b.-')
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        print('data length: ', len(joint))
        plt.show()

        # Create 2D plot for joint angles
        plt.subplot(611)
        plt.plot(q1 * 180. / np.pi, 'b-')
        plt.ylabel('q1 ($^\circ$)')
        plt.subplot(612)
        plt.plot(q2 * 180. / np.pi, 'b-')
        plt.ylabel('q2 ($^\circ$)')
        plt.subplot(613)
        plt.plot(q3, 'b-')
        plt.ylabel('q3 (mm)')
        plt.subplot(614)
        plt.plot(q4 * 180. / np.pi, 'b-')
        plt.ylabel('q4 ($^\circ$)')
        plt.subplot(615)
        plt.plot(q5 * 180. / np.pi, 'b-')
        plt.ylabel('q5 ($^\circ$)')
        plt.subplot(616)
        plt.plot(q6 * 180. / np.pi, 'b-')
        plt.ylabel('q6 ($^\circ$)')
        plt.xlabel('(step)')
        plt.show()
        return joint

    def exp0_get_transform(self):
        jaw1 = [5. * np.pi / 180.]
        self.dvrk.set_pose(jaw1=jaw1)
        j1 = self.joint_traj_tranform[:, 0]
        j2 = self.joint_traj_tranform[:, 1]
        j3 = self.joint_traj_tranform[:, 2]
        j4 = np.zeros_like(j1)
        j5 = np.zeros_like(j1)
        j6 = np.zeros_like(j1)
        self.collect_data_joint(j1,j2,j3,j4,j5,j6, transform='unknown')

    def exp1_move_all_joints(self):
        jaw1 = [5. * np.pi / 180.]
        self.dvrk.set_jaw(jaw=jaw1)
        j1 = self.joint_traj[:, 0]
        j2 = self.joint_traj[:, 1]
        j3 = self.joint_traj[:, 2]
        j4 = self.joint_traj[:, 3]
        j5 = self.joint_traj[:, 4]
        j6 = self.joint_traj[:, 5]
        self.collect_data_joint(j1,j2,j3,j4,j5,j6, transform='known')

    def exp2_move_q4_q5_q6(self):
        jaw1 = [5. * np.pi / 180.]
        self.dvrk.set_pose(jaw1=jaw1)
        j4 = self.joint_traj[:, 3]
        j5 = self.joint_traj[:, 4]
        j6 = self.joint_traj[:, 5]
        j1 = np.ones_like(j4)*self.dvrk.act_joint1[0]
        j2 = np.ones_like(j4)*self.dvrk.act_joint1[1]
        j3 = np.ones_like(j4)*self.dvrk.act_joint1[2]
        self.collect_data_joint(j1,j2,j3,j4,j5,j6, transform='known')

    def exp3_move_q4_only(self):
        jaw1 = [5. * np.pi / 180.]
        self.dvrk.set_pose(jaw1=jaw1)
        j4 = self.joint_traj[:, 3]
        j5 = np.zeros_like(j4)
        j6 = np.zeros_like(j4)
        j1 = np.ones_like(j4) * self.dvrk.act_joint1[0]
        j2 = np.ones_like(j4) * self.dvrk.act_joint1[1]
        j3 = np.ones_like(j4) * self.dvrk.act_joint1[2]
        self.collect_data_joint(j1, j2, j3, j4, j5, j6, transform='known')

    def exp4_move_q5_only(self):
        jaw1 = [5. * np.pi / 180.]
        self.dvrk.set_pose(jaw1=jaw1)
        j5 = self.joint_traj[:, 4]
        j4 = np.zeros_like(j5)
        j6 = np.zeros_like(j5)
        j1 = np.ones_like(j5) * self.dvrk.act_joint1[0]
        j2 = np.ones_like(j5) * self.dvrk.act_joint1[1]
        j3 = np.ones_like(j5) * self.dvrk.act_joint1[2]
        self.collect_data_joint(j1, j2, j3, j4, j5, j6, transform='known')

    def exp5_move_q6_only(self):
        jaw1 = [5. * np.pi / 180.]
        self.dvrk.set_pose(jaw1=jaw1)
        j6 = self.joint_traj[:, 5]
        j4 = np.zeros_like(j6)
        j5 = np.zeros_like(j6)
        j1 = np.ones_like(j6) * self.dvrk.act_joint1[0]
        j2 = np.ones_like(j6) * self.dvrk.act_joint1[1]
        j3 = np.ones_like(j6) * self.dvrk.act_joint1[2]
        self.collect_data_joint(j1, j2, j3, j4, j5, j6, transform='known')

    def collect_data_joint(self, j1, j2, j3, j4, j5, j6, transform='known'):    # j1, ..., j6: joint trajectory
        
        try:
            time_st = time.time()   # (sec)
            time_stamp = []
            q_des = []
            q_act = []
            pos_des = []
            pos_act = []
            assert len(j1)==len(j2)==len(j3)==len(j4)==len(j5)==len(j6)
            i = 1
            for qd1,qd2,qd3,qd4,qd5,qd6 in zip(j1,j2,j3,j4,j5,j6):
                joint1 = [qd1, qd2, qd3, qd4, qd5, qd6]
                self.dvrk.set_jaw(JAW_CLOSE_ANGLE)
                self.dvrk.set_joint(joint=joint1)
                self.dvrk.set_jaw(JAW_CLOSE_ANGLE)
                
                # Capture image from Zivid
                zivid_image,zivid_depth,zivid_pcl,intrinsics_matrix,distortion_coefficients = self.zivid.capture()
                img_color, img_depth, img_point = zivid_image,zivid_depth,zivid_pcl # self.BD.img_crop(zivid_image,zivid_depth,zivid_pcl)
                #img_color = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
                img_color_org = np.copy(img_color)
                cv2.imwrite('/home/davinci/dvrkCalibration/lab_meeting/zivid_frame/img_color_' + str(i) + '.png',img_color_org)
                # Find balls
                pbs = self.BD.find_balls(img_color_org, img_depth, img_point)
                img_color = self.BD.overlay_balls(img_color, pbs,intrinsics_matrix,distortion_coefficients)
                
                # Find tool position, joint angles, and overlay
                if pbs[0] == [] or pbs[1] == []:
                    qa1=0.0; qa2=0.0; qa3=0.0; qa4=0.0; qa5=0.0; qa6=0.0
                else:
                    # Find tool position, joint angles, and overlay
                    pt = self.BD.find_tool_position(pbs[0], pbs[1])  # tool position of pitch axis
                    pt = np.array(pt) * 0.001  # (m)
                    
                    if transform == 'known':
                        pt = self.BD.Rrc.dot(pt) + self.BD.trc
                        qa1, qa2, qa3 = self.BD.ik_position(pt)

                        # Find tool orientation, joint angles, and overlay
                        count_pbs = [pbs[2], pbs[3], pbs[4], pbs[5]]
                        if count_pbs.count([]) >= 2:
                            qa4=0.0; qa5=0.0; qa6=0.0
                        else:
                            Rm = self.BD.find_tool_orientation(pbs[2], pbs[3], pbs[4], pbs[5])  # orientation of the marker
                            qa4, qa5, qa6 = self.BD.ik_orientation(qa1, qa2, Rm)
                            img_color = self.BD.overlay_tool(img_color, [qa1, qa2, qa3, qa4, qa5, qa6], (0, 255, 0),intrinsics_matrix,distortion_coefficients)
                            cv2.imwrite('/home/davinci/dvrkCalibration/lab_meeting/tool_overlay/img_color_' + str(i) + '.png',img_color)
                # Append data pairs
                if transform == 'known':
                    # joint angles
                    q_des.append([qd1, qd2, qd3, qd4, qd5, qd6])
                    q_act.append([qa1, qa2, qa3, qa4, qa5, qa6])
                    time_stamp.append(time.time() - time_st)
                    print('index: ', len(q_des),'/',len(j1))
                    print('t_stamp: ', time.time() - time_st)
                    print('q_des: ', [qd1, qd2, qd3, qd4, qd5, qd6])
                    print('q_act: ', [qa1, qa2, qa3, qa4, qa5, qa6])
                    print(' ')
                elif transform == 'unknown':
                    # positions
                    pos_des_temp = self.BD.fk_position(q1=qd1, q2=qd2, q3=qd3, q4=0, q5=0, q6=0,
                                                                L1=self.BD.L1, L2=self.BD.L2, L3=0, L4=0)
                    pos_des.append(pos_des_temp)
                    pos_act.append(pt)
                    print('index: ', len(pos_des), '/', len(j1))
                    print('pos_des: ', pos_des_temp)
                    print('pos_act: ', pt)
                    print(' ')
                # import pdb
                # pdb.set_trace()
                # self.dvrk.set_joint(joint=[qd1, qd2, qd3, qd4, qd5, qd6])
                # self.dvrk.set_joint(joint=[qa1, qa2, qa3, qa4, qa5, qa6])
                # Visualize
                i += 1
                cv2.imshow("images", img_color)
                cv2.waitKey(1) & 0xFF
                # cv2.waitKey(0)
        finally:
            # Save data to a file
            if transform == 'known':
                np.save('q_des_raw', q_des)
                np.save('q_act_raw', q_act)
            elif transform == 'unknown':
                # Get transform from robot to camera
                np.save('pos_des', pos_des)
                np.save('pos_act', pos_act)
                T = U.get_rigid_transform(np.array(pos_act), np.array(pos_des))
                np.save('Trc', T)
            np.save('t_stamp_raw', time_stamp)
            print("Data is successfully saved")

if __name__ == "__main__":
    cal = dvrkCalibration()
    # cal.exp0_get_transform()
    cal.exp1_move_all_joints()
    # cal.exp2_move_q4_q5_q6()
    # cal.exp3_move_q4_only()
    # cal.exp4_move_q5_only()
    # cal.exp5_move_q6_only()
