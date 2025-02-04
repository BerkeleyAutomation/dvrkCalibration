import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from dvrk.motion.dvrkArm import dvrkArm
from dvrk.motion.dvrkTypes import dvrkTypes
from dvrk.motion.dvrkKinematics import dvrkKinematics
import termios
import select
import sys
import tty
import os
from scipy.spatial.transform import Rotation as R
import time
import os
import math

JAW_OPEN_ANGLE = [np.pi / 2]  # Angle for opening jaw to release
JAW_CLOSE_ANGLE = [-0.3]  # Angle for closing jaw to grasp
psm1 = dvrkArm("/PSM1", dvrk_type=dvrkTypes.LRG_SUTURECUT_NEEDLE_DRIVER, use_rnn=False)
psm2 = dvrkArm("/PSM2", dvrk_type=dvrkTypes.LARGE_NEEDLE_DRIVER, use_rnn=False)


def plot_position(pos_des):
    # plot trajectory of des & act position
    pos_des = np.array(pos_des)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plt.plot(pos_des[:, 0], pos_des[:, 1], pos_des[:, 2], "b.--")
    # plt.plot(pos_act[:, 0], pos_act[:, 1], pos_act[:, 2], 'r.-')
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    # plt.title('Trajectory of tool position')
    plt.legend(["desired", "actual"])
    plt.show()


def plot_joint(q_des):
    q_des = np.array(q_des)
    # Create plot
    # plt.title('joint angle')
    ax = plt.subplot(611)
    plt.plot(q_des[:, 0] * 180.0 / np.pi, "b-")
    # plt.plot(q_act[:, 0] * 180. / np.pi, 'r-')
    plt.legend(loc="upper center", bbox_to_anchor=(0.9, 2))
    ax.set_xticklabels([])
    plt.ylabel("q1 ($^\circ$)")

    ax = plt.subplot(612)
    plt.plot(q_des[:, 1] * 180.0 / np.pi, "b-")
    # plt.plot(q_act[:, 1] * 180. / np.pi, 'r-')
    ax.set_xticklabels([])
    plt.ylabel("q2 ($^\circ$)")

    ax = plt.subplot(613)
    plt.plot(q_des[:, 2], "b-")
    # plt.plot(q_act[:, 2], 'r-')
    ax.set_xticklabels([])
    plt.ylabel("q3 (m)")

    ax = plt.subplot(614)
    plt.plot(q_des[:, 3] * 180.0 / np.pi, "b-")
    # plt.plot(q_act[:, 3]*180./np.pi, 'r-')
    ax.set_xticklabels([])
    plt.ylabel("q4 ($^\circ$)")

    ax = plt.subplot(615)
    plt.plot(q_des[:, 4] * 180.0 / np.pi, "b-")
    # plt.plot(q_act[:, 4]*180./np.pi, 'r-')
    ax.set_xticklabels([])
    plt.ylabel("q5 ($^\circ$)")

    plt.subplot(616)
    plt.plot(q_des[:, 5] * 180.0 / np.pi, "b-")
    # plt.plot(q_act[:, 5]*180./np.pi, 'r-')
    plt.ylabel("q6 ($^\circ$)")
    plt.xlabel("(sample)")
    plt.show()


def ik_position(pos):
    x = pos[0]  # (m)
    y = pos[1]
    z = pos[2]
    L1 = 0.4318  # Rcc (m)
    L2 = 0.4162  # tool

    # Forward Kinematics
    # x = np.cos(q2)*np.sin(q1)*(L2-L1+q3)
    # y = -np.sin(q2)*(L2-L1+q3)
    # z = -np.cos(q1)*np.cos(q2)*(L2-L1+q3)

    # Inverse Kinematics
    q1 = np.arctan2(x, -z)  # (rad)
    q2 = np.arctan2(-y, np.sqrt(x**2 + z**2))  # (rad)
    q3 = np.sqrt(x**2 + y**2 + z**2) + L1 - L2  # (m)
    return q1, q2, q3


def random_sampling(sample_number, psm_number, q1_min, q1_max, q2_min, q2_max, q3):
    psm = None
    if psm_number == "1":
        psm = psm1
    elif psm_number == "2":
        psm = psm2
    else:
        print("Please select PSM1 or 2")
        exit()
    q_target = []
    pos_target = []
    pos_min = []
    pos_max = []

    q4_range = np.array([0, 0]) * np.pi / 180.0
    q5_range = np.array([0, 0]) * np.pi / 180.0
    q6_range = np.array([-90, 90]) * np.pi / 180.0

    for i in range(sample_number):
        q1 = np.random.uniform(q1_min, q1_max)
        q2 = np.random.uniform(q2_min, q2_max)
        q4, q5, q6 = 0.0, 0.0, 0.0
        curr_q = [q1, q2, q3, q4, q5, q6]
        q_target.append(curr_q)
        pose = dvrkKinematics.joint_to_pose(
            curr_q, L1=psm.l_rcc_, L2=psm.l_tool_, L3=psm.l_pitch_2_yaw_, L4=psm.l_yaw_2_ctrl_pnt_
        )
        pos_target.append(pose[0])

    return q_target, pos_target


def set_bounding_box_input(psm_number):
    psm = None
    if psm_number == "1":
        psm = psm1
    elif psm_number == "2":
        psm = psm2
    else:
        print("Please select PSM1 or 2")
        exit()

    initial_pos = np.array([0, 0, -0.13])
    initial_euler = np.array([0, 0, 0])
    initial_quat = np.array([0, 0, 0, 1])
    initial_pose = initial_pos, initial_quat
    psm.set_pose(*initial_pose)
    time.sleep(0.1)
    input("Rotate gripper 90 degrees in x direction such that the plus sign will be up/down (easier to hit the table)")
    time.sleep(0.1)
    psm.set_jaw(JAW_OPEN_ANGLE)
    time.sleep(0.1)
    input("Place plus sign fiducial with little yellow up")
    psm.set_jaw(JAW_CLOSE_ANGLE)
    print(
        "Now manually move the robot around and call psm.get_current_pose() to get your bounding box. For the z values, make sure you add 0.0193 to the z because we want position to be before the pitch joint, not the end effector tip"
    )
    import pdb

    pdb.set_trace()
    # x_min,y_min,z_min = 0.0,0.0,-0.111
    # x_max,y_max,z_max = 0.115,0.115, -0.107


def is_there_a_key_press():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])


def find_range(psm):
    print(
        'Finding range of motion for joint 0\nMove the arm manually (pressing the clutch) to find the maximum range of motion for the first joint (left to right motion).\n - press "d" when you are done\n - press "q" to abort\n'
    )

    q1_min = math.radians(180.0)
    q1_max = math.radians(-180.0)
    q2_min = math.radians(180.0)
    q2_max = math.radians(-180.0)
    done = False

    # termios settings
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while not done:
            # process key
            if is_there_a_key_press():
                c = sys.stdin.read(1)
                if c == "d":
                    done = True
                elif c == "q":
                    sys.exit("... calibration aborted by user")

            # get measured joint values
            p = psm.get_current_joint()
            if p[0] > q1_max:
                q1_max = p[0]
            elif p[0] < q1_min:
                q1_min = p[0]
            if p[1] > q2_max:
                q2_max = p[1]
            elif p[1] < q2_min:
                q2_min = p[1]

            # display current range on the same line
            sys.stdout.write(
                "\rQ1 Range[%02.2f, %02.2f] | Q2 Range[%02.2f, %02.2f]"
                % (math.degrees(q1_min), math.degrees(q1_max), math.degrees(q2_min), math.degrees(q2_max))
            )
            sys.stdout.flush()

            # sleep
            time.sleep(0.1)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        print("")
    return q1_min, q1_max, q2_min, q2_max


if __name__ == "__main__":
    psm_number = input("Which PSM are you calibrating for suturing workspace. Pick 1 or 2 ")
    print(psm_number)
    psm = None
    if psm_number == "1":
        psm = psm1
    elif psm_number == "2":
        psm = psm2
    else:
        print("Please select PSM1 or 2")
        exit()
    input("Press enter when camera is running so you can visually verify the bounding box positions")
    input("Press enter to set height of translation joint")
    curr_joints = psm.get_current_joint()
    q3 = curr_joints[2]
    start_joints = np.array([0, 0, q3, 0, 0, 0])
    psm.set_joint(start_joints)

    q1_min, q1_max, q2_min, q2_max = find_range(psm)

    q_target, pos_target = random_sampling(200, psm_number, q1_min, q1_max, q2_min, q2_max, q3)
    print(np.shape(q_target))
    print(np.shape(pos_target))
    plot_position(pos_target)
    plot_joint(q_target)

    # if not os.path.exists("shallow_and_deep_calibration_outputs"):
    #     os.makedirs("shallow_and_deep_calibration_outputs")
    # np.save("shallow_and_deep_calibration_outputs/prime_psm" + psm_number + "_random_sampled", q_target)

    if not os.path.exists("shallow_and_deep_calibration_outputs"):
        os.makedirs("shallow_and_deep_calibration_outputs")
    np.save("shallow_and_deep_calibration_outputs/prime_psm" + psm_number + "_shallow_random_sampled", q_target)

    # I'm not saving to true filepath because I don't want to overwrite anything while I'm still testing
    # But eventually save it to this folder: /home/davinci/dvrkCalibration/experiment/0_trajectory_extraction
