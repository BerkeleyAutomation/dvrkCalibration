import numpy as np
np.set_printoptions(formatter={"float_kind": "{:.3f}".format})

from dvrk.vision.miscellaneous.BallDetectionRGBD import BallDetectionRGBD
# from dvrk.vision.RSBallDetectionRGBD import RSBallDetectionRGBD
from dvrk.vision.cameras.ZividCapture import ZdvrkKinematicsividCapture
from dvrk.motion.dvrkKinematics import dvrkKinematics
# from dvrk_shunt.servoing.realsense import RealSense
from dvrk.motion.dvrkArm import dvrkArm

import dvrk.motion.dvrkVariables as dvrkVar
import dvrk.utils.CmnUtil as UTILS
from path import *
import sys
import cv2
import time

from dotmap import DotMap
from dvrk.utils.plot import *
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QPushButton,
    QGridLayout,
    QLabel,
    QMessageBox,
    QRadioButton,
    QGroupBox,
    QLineEdit,
)

class GenTrajectoryGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.conf = DotMap()
        self.pos_ws = []
        self.joint_traj = []
        self.runUI()

    def runUI(self):
        grid = QGridLayout()
        grid.setColumnStretch(0, 4)
        grid.setColumnStretch(1, 4)
        # row = 0
        # label0 = QLabel("0. Attach spheres on surgical instrument")
        # label0.setStyleSheet("color: black;" "font-size: 15px;" "font: bold;")
        # grid.addWidget(label0, row, 0, 1, 3)

        # row = 1
        # label0 = QLabel("1. Set background black")
        # label0.setStyleSheet("color: black;" "font-size: 15px;" "font: bold;")
        # grid.addWidget(label0, row, 0, 1, 3)

        # row = 2
        # label1 = QLabel("2. Power ON devices:  PSM and Zivid camera")
        # label1.setStyleSheet("color: black;" "font-size: 15px;" "font: bold;")
        # grid.addWidget(label1, row, 0, 1, 3)

        # row = 3
        # label2 = QLabel("3. Set configurations:")
        # label2.setStyleSheet("color: black;" "font-size: 15px;" "font: bold;")
        # grid.addWidget(label2, row, 0, 1, 3)

        row = 4
        grid.addWidget(self.createZividSelectionGroup(), row, 0)
        grid.addWidget(self.createPSMSelectionGroup(), row, 1)
        self.btnConfigure = QPushButton("Configure", self)
        self.btnConfigure.clicked.connect(self.btnConfigureClicked)
        grid.addWidget(self.btnConfigure, row, 2)

        row = 5
        label4 = QLabel("4. Define workspace of PSM:")
        label4.setStyleSheet("color: black;" "font-size: 15px;" "font: bold;")
        grid.addWidget(label4, row, 0, 1, 3)


        ###===### SELECT POSITION ###===###
        row = 6
        grid.addWidget(self.createDefineWorkspaceLabelGroup(), row, 0, 1, 2)
        grid.addWidget(self.createDefineWorkspaceButtonGroup(), row, 2, 1, 2)


        row = 7
        label6 = QLabel("5. Define Number of Samples (Default=500):")
        label6.setStyleSheet("color: black;" "font-size: 15px;" "font: bold;")
        grid.addWidget(label6, row, 0, 1, 3)

        row = 8
        edit = QLineEdit()
        edit.setObjectName("edit_sample")
        edit.setText("500")
        grid.addWidget(edit, row, 0)
        edit.setObjectName("edit_sample")
        label8 = QLabel("# of Sample = 500")
        label8.setObjectName("label_sample")
        grid.addWidget(label8, row, 1)
        self.btnSetandPlot = QPushButton("Set, Plot, and Save 6-q_joint_traj.npy", self)
        self.btnSetandPlot.setEnabled(False)


        ###===### RANDOM SAMPLING ###===###
        self.btnSetandPlot.clicked.connect(self.btnSetandPlotClicked)
        grid.addWidget(self.btnSetandPlot, row, 3)


        row = 9
        label7 = QLabel("6. Start Calibration:")
        label7.setStyleSheet("color: black;" "font-size: 15px;" "font: bold;")
        grid.addWidget(label7, row, 0, 1, 3)

        row = 10
        self.btnStartCalibration = QPushButton("Start Calibration", self)
        self.btnStartCalibration.setEnabled(False)
        self.btnStartCalibration.clicked.connect(self.btnStartCalibrationClicked)
        grid.addWidget(self.btnStartCalibration, row, 0, 2, 2)

        self.setLayout(grid)
        # self.setWindowTitle("dVRK Cam2Rob Calibration (by Minho, Jun 7, 2021)")
        self.setGeometry(500, 150, 600, 700)
        self.show()

    def createZividSelectionGroup(self):
 
        groupbox = QGroupBox("Zivid Selection")
        rbtn1 = QRadioButton("inclined")
        rbtn2 = QRadioButton("overhead")
        rbtn3 = QRadioButton("realsense")
        rbtn1.setChecked(True)
        self.conf.which_camera = rbtn1.text()
        rbtn1.toggled.connect(self.rbtnZividToggled)
        rbtn2.toggled.connect(self.rbtnZividToggled)
        rbtn3.toggled.connect(self.rbtnZividToggled)
        vbox = QVBoxLayout()
        vbox.addWidget(rbtn1)
        vbox.addWidget(rbtn2)
        vbox.addWidget(rbtn3)
        groupbox.setLayout(vbox)
        return groupbox

    def createPSMSelectionGroup(self):
        groupbox = QGroupBox("PSM Selection")
        rbtn1 = QRadioButton("PSM1")
        rbtn2 = QRadioButton("PSM2")
        rbtn1.setChecked(True)
        self.conf.which_arm = rbtn1.text()
        rbtn1.toggled.connect(self.rbtnPSMToggled)
        rbtn2.toggled.connect(self.rbtnPSMToggled)
        vbox = QVBoxLayout()
        vbox.addWidget(rbtn1)
        vbox.addWidget(rbtn2)
        groupbox.setLayout(vbox)
        return groupbox

    def createDefineWorkspaceLabelGroup(self):
        groupbox = QGroupBox()
        label_pos1 = QLabel("Wrist Position1:  x=0.0,  y=0.0,  z=0.0", self)
        label_pos1.setObjectName("label_pos1")
        label_pos2 = QLabel("Wrist Position2:  x=0.0,  y=0.0,  z=0.0", self)
        label_pos2.setObjectName("label_pos2")
        vbox = QVBoxLayout()
        vbox.addWidget(label_pos1)
        vbox.addWidget(label_pos2)
        groupbox.setLayout(vbox)
        return groupbox


    def createDefineWorkspaceButtonGroup(self):
        groupbox = QGroupBox()
        self.btnCapture = QPushButton("Capture Image", self)
        self.btnCapture.setEnabled(False)
        self.btnCapture.clicked.connect(self.btnCaptureClicked)
        self.btnSelectPos = QPushButton("Select Pos", self)


        ###===### SELECT POSITION ###===###
        self.btnSelectPos.setEnabled(False)
        self.btnSelectPos.clicked.connect(self.btnSelectPosClicked)


        self.btnResetWS = QPushButton("Reset Pos", self)
        self.btnResetWS.setEnabled(False)
        self.btnResetWS.clicked.connect(self.btnResetWSClicked)
        vbox = QVBoxLayout()
        vbox.addWidget(self.btnCapture)
        vbox.addWidget(self.btnSelectPos)
        vbox.addWidget(self.btnResetWS)
        groupbox.setLayout(vbox)
        return groupbox


    def rbtnZividToggled(self):
        rbtn = self.sender()
        if rbtn.isChecked():
            self.conf.which_camera = rbtn.text()
            print("camera selection: ", self.conf.which_camera)

    def rbtnPSMToggled(self):
        rbtn = self.sender()
        if rbtn.isChecked():
            self.conf.which_arm = rbtn.text()
            print("PSM selection: ", self.conf.which_arm)

    def btnConfigureClicked(self):
        self.dvrk = dvrkArm("/" + self.conf.which_arm)
        if self.conf.which_camera == "realsense":
            self.zivid = RealSense()
        else:
            self.zivid = ZividCapture(which_camera=self.conf.which_camera)
        self.zivid.start()
        QMessageBox.about(self, "Message", "Successfully Configured!")
        self.btnConfigure.setEnabled(False)
        self.btnCapture.setEnabled(True)
        self.btnSelectPos.setEnabled(True)
        self.btnResetWS.setEnabled(True)

    def btnCaptureClicked(self):
        image = self.zivid.capture_2Dimage(color="BGR")
        scale = 0.9
        w = int(image.shape[1] * scale)
        h = int(image.shape[0] * scale)
        dim = (w, h)
        image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("", image)
        cv2.waitKey(0)


    ###===### SELECT POSITION ###===###

    def btnSelectPosClicked(self):
        pos, _ = self.dvrk.get_current_pose(wait_callback=True)
        if len(self.pos_ws) == 0:
            label_pos1 = self.findChild(QLabel, "label_pos1")
            label_pos1.setText("Wrist Position1:  x=%1.3f,  y=%1.3f,  z=%1.3f" % (pos[0], pos[1], pos[2]))
            label_pos1.setStyleSheet("color: black;" "font: bold;")
            self.pos_ws.append(pos)
        elif len(self.pos_ws) == 1:
            label_pos2 = self.findChild(QLabel, "label_pos2")
            label_pos2.setText("Wrist Position2:  x=%1.3f,  y=%1.3f,  z=%1.3f" % (pos[0], pos[1], pos[2]))
            label_pos2.setStyleSheet("color: black;" "font: bold;")
            self.pos_ws.append(pos)
            self.btnSelectPos.setEnabled(False)
            self.btnSetandPlot.setEnabled(True)

    ###===###


    def btnResetWSClicked(self):
        self.pos_ws = []
        label_pos1 = self.findChild(QLabel, "label_pos1")
        label_pos1.setText("Wrist Position1:  x=0.0,  y=0.0,  z=0.0")
        label_pos1.setStyleSheet("color: black;")
        label_pos2 = self.findChild(QLabel, "label_pos2")
        label_pos2.setText("Wrist Position2:  x=0.0,  y=0.0,  z=0.0")
        label_pos2.setStyleSheet("color: black;")
        self.btnSelectPos.setEnabled(True)
        self.btnSetandPlot.setEnabled(False)
        self.btnStartCalibration.setEnabled(False)


    ###===### RANDOM SAMPLING ###===###

    def btnSetandPlotClicked(self):
        edit_sample = self.findChild(QLineEdit, "edit_sample")
        n_sample = int(edit_sample.text())
        label_sample = self.findChild(QLabel, "label_sample")
        label_sample.setText("# of Sample = " + str(n_sample))

        # Default values for PSM1
        # self.pos_ws[0] = [0.108, 0.094, -0.119]
        # self.pos_ws[1] = [0.001, 0.054, -0.099]
        
        joint_traj = self.random_sampling(n_sample, self.pos_ws[0], self.pos_ws[1])
        np.save("6-q_joint_traj", np.array(joint_traj))

        pos_traj = []
        for q in joint_traj:
            # fk_pos = dvrkKinematics.fk([q[0], q[1], q[2], 0, 0, 0], L1=dvrkVar.L1, L2=dvrkVar.L2, L3=0, L4=0)
            # fk_pos = np.array(fk_pos)[:3, -1]

            # NOTE: Is this correct?
            fk_pos = dvrkKinematics.fk(q, L1=dvrkVar.L1, L2=dvrkVar.L2, L3=0, L4=0) # Now using all 6 vars in q
            fk_pos = np.array(fk_pos)     # Now using all of fk_pos
            pos_traj.append(fk_pos)

        plot_trajectory(joint_traj, pos_traj)
        self.joint_traj = joint_traj
        self.btnStartCalibration.setEnabled(True)

    ###===###


    def btnStartCalibrationClicked(self):
        self.btnCapture.setEnabled(False)
        self.btnSelectPos.setEnabled(False)
        self.btnResetWS.setEnabled(False)
        self.btnSetandPlot.setEnabled(False)
        self.btnStartCalibration.setEnabled(False)

        print("Breakpoint")
        return

        self.calibration = GenTrajectory(self.dvrk, self.zivid)
        pos_rob, pos_cam, Trc, time_stamp = self.calibration.start(self.conf, self.joint_traj)

        pos_cam_tr = Trc[:3, :3].dot(pos_cam.T).T + Trc[:3, -1]
        plot_position(pos_rob, pos_cam_tr)

        # Save data to a file
        np.save("pos_rob", pos_rob)
        np.save("pos_cam", pos_cam)
        np.save("Trc_" + self.conf.which_camera + "_" + self.conf.which_arm, Trc)
        print("Trc=", Trc)
        print("Tcr=", np.linalg.inv(Trc))
        np.save("t_stamp", time_stamp)
        QMessageBox.about(self, "Message", "Data is successfully saved!")
        self.btnStartCalibration.setEnabled(False)


    ###===### RANDOM SAMPLING ###===###

    def random_sampling(self, n_samples, pos_min, pos_max):
        joint_traj = []

        for _ in range(n_samples):
            p, q = self.random_pose(pos_min, pos_max)

            # NOTE: method from /home/davinci/dvrk/dvrk/motion/dvrkKinematics.py
            joint = dvrkKinematics.pose_to_joint(p, q)[0] # NOTE: list of values

            joint_traj.append(joint)
        return joint_traj


    def random_pose(self, lower, upper, min_z=0.67):
        q = None
        PI = np.pi
        
        while q is None:
            get_q = lambda x, y: UTILS.euler_to_quaternion(np.random.uniform(x, y))

            # Enable first line of each pair for vertical positions and second line for horizontal ones
            if False:
                quaternion = get_q([PI/2 - 0.1, 0, PI/4], [PI/2 + 0.1, PI*2, 7*PI/4])
                axis = 2
            else:
                quaternion = get_q([-PI, -PI, -PI], [PI , PI , PI]) # NOTE: Kush take a look at these!!
                axis = 1

            R = UTILS.quaternion_to_R(quaternion)
            proj = np.array([[0], [0], [0]])
            proj[axis] = 1
            dir = np.dot(R, proj)

            # if dir[axis] > min_z:
            #     q = quaternion
            if dir[2] > 0.85:
                q = quaternion

        p = np.random.uniform(lower, upper)
        return p, q
    
    ###===###


    # NOTE: Method from /home/davinci/dvrkCalibration/motion/dvrkKinematics.py is missing IKINE().
    def pose_to_joint(self, pos, rot):
        if pos == [] or rot == []:
            joint = []
        else:
            T = np.zeros((4, 4))
            R = UTILS.quaternion_to_R(rot[0], rot[1], rot[2], rot[3])
            T[:3,:3] = R
            T[:3,-1] = np.transpose(pos)
            T[-1,-1] = 1
            T = np.matrix(T)
            q0 = np.matrix([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            ik_sol = ikine(T, q0)
            joint = [ik_sol[0,0], ik_sol[0,1], ik_sol[0,2], ik_sol[0,3], ik_sol[0,4], ik_sol[0,5]]
        return joint


if __name__ == "__main__":
    app = QApplication(sys.argv)
    cal = GenTrajectoryGUI()
    sys.exit(app.exec_())


###===### CALIBRATION LOGIC ###===###

exit()

class GenTrajectory():
    def __init__(self, dvrk, cam):
        self.dvrk = dvrk
        self.zivid = cam

    def pose_estimation(
        self, BallDetection, pbr, pbg, pbb, pby, use_Trc, which_arm
    ):  # Find tool position, joint angles
        bd = BallDetection
        pt = []
        q_phy = []
        if len(pbr) < 2:
            pass
        else:
            pt = bd.find_tool_pitch(pbr[0], pbr[1])  # tool position of pitch axis
            pt = np.array(pt) * 0.001  # (m)
            if use_Trc:
                pt = bd.Rrc.dot(pt) + bd.trc
                qp1, qp2, qp3 = dvrkKinematics.ik_position_straight(pt, L3=0, L4=0)  # position of pitch axis

                # Find tool orientation, joint angles, and overlay
                temp = [pbr[2], pbg, pbb, pby]
                if len(pbr) < 3:
                    qp4 = 0.0
                    qp5 = 0.0
                    qp6 = 0.0
                elif temp.count([]) >= 2:
                    qp4 = 0.0
                    qp5 = 0.0
                    qp6 = 0.0
                else:
                    Rm = bd.find_tool_orientation(pbr[2], pbg, pbb, pby, which_arm)  # orientation of the marker
                    qp4, qp5, qp6 = dvrkKinematics.ik_orientation(qp1, qp2, Rm)
                q_phy = [qp1, qp2, qp3, qp4, qp5, qp6]
            else:
                q_phy = []
        return pt, q_phy

    def start(self, conf, joint_traj):
        which_arm = conf.which_arm
        which_camera = conf.which_camera

        if which_camera == "realsense":
            bd = RSBallDetectionRGBD(Trc=[], intrinsics=self.zivid.intr)
        else:
            bd = BallDetectionRGBD(Trc=[], which_camera=which_camera)
        use_Trc = False

        # bd = BallDetectionRGBD(Trc=np.load("Trc_inclined_PSM1.npy"), which_camera=which_camera)
        # use_Trc = True

        # collect data
        time_st = time.time()  # (sec)
        time_stamp = []
        q_cmd_ = []
        q_phy_ = []
        pos_rob = []
        pos_cam = []
        for q_cmd in joint_traj:
            if q_cmd[2] < 0.103 or q_cmd[2] > 0.208:
                continue
            if use_Trc:
                pass
            else:
                q_cmd[3] = 0.0
                q_cmd[4] = 0.0
                q_cmd[5] = 0.0
            self.dvrk.set_jaw_interpolate(jaw=np.deg2rad([-np.pi / 4]))
            self.dvrk.set_joint_interpolate(joint=q_cmd)
            time.sleep(0.5)

            # Capture image from Zivid
            color, _, point = self.zivid.capture_3Dimage(color="BGR")

            # Find balls
            if use_Trc:
                pbr = bd.find_balls(color, point, "red", nb_sphere=3, visualize=False)
                pbg = bd.find_balls(color, point, "green", nb_sphere=1, visualize=False)
                pbb = bd.find_balls(color, point, "blue", nb_sphere=1, visualize=False)
                pby = bd.find_balls(color, point, "yellow", nb_sphere=1, visualize=False)
            else:
                pbr = bd.find_balls(color, point, "red", nb_sphere=2, visualize=False)
                pbg = []
                pbb = []
                pby = []

            # pose estimation
            pt, q_phy = self.pose_estimation(bd, pbr, pbg, pbb, pby, use_Trc, which_arm)

            # overlay
            color = bd.overlay_ball(color, pbr)
            color = bd.overlay_ball(color, [pbg])
            color = bd.overlay_ball(color, [pbb])
            color = bd.overlay_ball(color, [pby])
            color = bd.overlay_dot(color, pt, "pitch")
            if use_Trc:
                color = bd.overlay_tool(color, q_phy, (0, 0, 255))  # measured on red
                color = bd.overlay_tool(color, q_cmd, (0, 255, 0))  # commanded on green

            # Append data pairs
            if use_Trc:
                # joint angles
                q_cmd_.append(q_cmd)
                q_phy_.append(q_phy)
                time_stamp.append(time.time() - time_st)
                print("index: ", len(q_cmd_), "/", len(joint_traj))
                print("t_stamp: ", time.time() - time_st)
                print("q_cmd: ", np.array(q_cmd))
                print("q_phy: ", np.array(q_phy))
                print(" ")
            else:
                if len(pt) != 0:
                    # positions of pitch axis
                    pos_rob_temp = dvrkKinematics.fk(
                        [q_cmd[0], q_cmd[1], q_cmd[2], 0, 0, 0], L1=dvrkVar.L1, L2=dvrkVar.L2, L3=0, L4=0
                    )[:3, -1]
                    pos_rob.append(pos_rob_temp)
                    pos_cam.append(pt)
                    print("index: ", len(pos_rob), "/", len(joint_traj))
                    print("pos_rob: ", pos_rob_temp)
                    print("pos_cam: ", pt)
                    print(" ")

            # Visualize
            scale = 0.8
            w = int(color.shape[1] * scale)
            h = int(color.shape[0] * scale)
            dim = (w, h)
            color = cv2.resize(color, dim, interpolation=cv2.INTER_AREA)
            cv2.imshow("", color)
            cv2.waitKey(1) & 0xFF
            # cv2.waitKey(0)

        Trc = U.get_rigid_transform(np.array(pos_cam), np.array(pos_rob))
        # Tcr = np.linalg.inv(Trc)
        return np.array(pos_rob), np.array(pos_cam), Trc, np.array(time_stamp)
