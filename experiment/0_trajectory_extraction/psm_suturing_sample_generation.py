import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from dvrk.motion.dvrkArm import dvrkArm
from dvrk.motion.dvrkTypes import dvrkTypes
from dvrk.motion.dvrkKinematics import dvrkKinematics
import os
from scipy.spatial.transform import Rotation as R
import time
import copy
import os

JAW_OPEN_ANGLE = [np.pi / 2]  # Angle for opening jaw to release
JAW_CLOSE_ANGLE = [-0.3]  # Angle for closing jaw to grasp
psm1 = dvrkArm("/PSM1", dvrk_type=dvrkTypes.LRG_SUTURECUT_NEEDLE_DRIVER, use_rnn=False)
psm2 = dvrkArm("/PSM2", dvrk_type=dvrkTypes.LARGE_NEEDLE_DRIVER, use_rnn=False)
WOUND_WIDTH = 0.009
WOUND_LENGTH = 0.04152823
NUM_SUTURES = 6


def run_prime_insertion_motion():
    radius = 0.01775
    # abby changed this (i needed it to be lower or sometimes it missed inserting and fell out faster)
    # #changed from [ 0.10532994,  0.09117879 ,-0.13457497])
    psm1_start_pos = np.array([0.10880398, 0.08446415, -0.1007803])
    smooth_constant = 8
    euler_list = np.linspace(np.array([np.pi / 2, np.pi / 2, 0]), np.array([np.pi / 2, 0, 0]), num=smooth_constant)
    psm1_start_quat = R.from_euler("xyz", np.array([np.pi / 2, np.pi / 2, 0])).as_quat()
    psm1_start_pose = psm1_start_pos, psm1_start_quat

    angle_list = np.linspace(np.pi / 2, np.pi, num=smooth_constant)
    x_coords, z_coords = [], []
    for angle in angle_list:
        x, z = radius * np.cos(angle), radius * np.sin(angle)
        x_coords.append(x)
        z_coords.append(z)
    x_coords = np.array(x_coords)
    z_coords = np.array(z_coords)

    # Step 1: Translate in x by half of needle radius + wound width
    # Right now wound width is hardcoded but we get it from the suture plan
    # changed this to be more severe; this should be enough to get through first wall of phantom
    translation_scale_factor = 2
    delta_translate = np.array([translation_scale_factor * ((radius + WOUND_WIDTH) / 2), 0, 0])

    psm1_pos, psm1_quat = psm1.get_current_pose()
    time.sleep(0.1)
    psm1_euler = R.from_quat(psm1_quat).as_euler("xyz")
    new_psm1_pos = psm1_pos + delta_translate
    new_psm1_pose = new_psm1_pos, psm1_start_quat
    print("Step 1")

    psm1.set_pose(*new_psm1_pose)
    time.sleep(0.1)

    # Step 2: Rotate the needle inside the wound by the value done here
    # TODO: Make this value cleaner instead of doing full interpolation and cutting it off after 3
    # AKA: We should be able to say rotate the needle by X degrees
    psm1_pos, psm1_quat = psm1.get_current_pose()
    psm1_euler = R.from_quat(psm1_quat).as_euler("xyz")
    delta_extraction_rotate_euler_angles = np.array([0, -np.pi / 2, 0]) / 5
    # for i in range(euler_list.shape[0]):
    for i in range(int(smooth_constant) - 1):

        psm1_euler = euler_list[i]
        new_psm1_euler = psm1_euler + delta_extraction_rotate_euler_angles
        new_psm1_quat = R.from_euler("xyz", new_psm1_euler).as_quat()
        new_psm1_pose = new_psm1_pos, new_psm1_quat
        print("Add pose: " + str(new_psm1_pose))
        psm1.set_pose(*new_psm1_pose)
        time.sleep(0.1)
        # abbys changed the two lines below this
        translation_scale_factor = 1.45
        down_right = np.array(
            [
                translation_scale_factor * (x_coords[i + 1] - x_coords[i]),
                0,
                translation_scale_factor * (z_coords[i + 1] - z_coords[i]),
            ]
        )
        # pdb.set_trace()
        new_psm1_pos = new_psm1_pos + down_right
        #    import pdb
        #    #print(f'down_right {down_right}')
        #    pdb.set_trace()
        psm1.set_pose(*new_psm1_pose)
        time.sleep(0.4)
        psm1_pos, psm1_quat = psm1.get_current_pose()
        psm1_euler = R.from_quat(psm1_quat).as_euler("xyz")
    # lucky number 7
    translation_scale_factor_2 = 0.07
    delta_translate_2 = np.array([translation_scale_factor_2 * ((radius + WOUND_WIDTH) / 2), 0, 0])
    psm1_pos, psm1_quat = psm1.get_current_pose()
    psm1_euler = R.from_quat(psm1_quat).as_euler("xyz")
    new_psm1_pos = psm1_pos + delta_translate_2
    new_psm1_pose = new_psm1_pos, psm1_quat
    psm1.set_pose(*new_psm1_pose)
    time.sleep(0.2)


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


def gen_sampling(pose_lists):
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
    for pos, quat in pose_lists:
        pos_target.append(pos)
        q_target.append(
            dvrkKinematics.pose_to_joint(pos, quat, psm.l_rcc_, psm.l_tool_, psm.l_pitch_2_yaw_, psm.l_yaw_2_ctrl_pnt_)[
                0
            ]
        )
    return q_target, pos_target


def set_suturing_pipeline_input(psm_number):
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
    curr_joints = psm.get_current_joint()

    curr_joints[3:] = [0, 0, 0]
    psm.set_joint(joint=curr_joints)

    time.sleep(0.1)
    psm.set_jaw(JAW_OPEN_ANGLE)
    time.sleep(0.1)
    input("Place plus sign fiducial with little green forward")
    psm.set_jaw(JAW_CLOSE_ANGLE)
    pose_lists = []
    # Add home pose
    pose_lists.append((np.array([0, 0, -0.13]), np.array([0, 0, 0, 1])))
    wound_start = None
    # Add warmup poses
    if psm_number == "1":
        pose_lists.append((np.array([0.01, 0, -0.12]), np.array([0.32743764, 0.79021532, 0.43636955, 0.27915223])))
        pose_lists.append((np.array([0.01, 0.05, -0.135]), np.array([0.1098817, 0.54847876, 0.7428689, 0.3677538])))
        pose_lists.append((np.array([0.05, 0, -0.11]), np.array([0.06069057, 0.49024802, 0.86942407, 0.00867864])))
        pose_lists.append((np.array([0.05, 0.05, -0.13]), np.array([0.63309931, -0.3134932, -0.31293999, 0.63480379])))
        # Add home pose
        pose_lists.append((np.array([0, 0, -0.13]), np.array([0, 0, 0, 1])))

        # Move to suture clear pose
        pose_lists.append(
            (
                np.array([0.11316637, 0.01303404, -0.04916395]),
                np.array([0.12616457, -0.60540458, -0.22937437, 0.75163502]),
            )
        )

        # Add home pose
        pose_lists.append((np.array([0, 0, -0.13]), np.array([0, 0, 0, 1])))

        # Go to pre-handover pose
        pose_lists.append(
            (
                np.array([0.08880125, 0.05858506, -0.10344337]),
                np.array([0.61735759, -0.17638474, 0.31953042, 0.69689192]),
            )
        )

        # Go to handover pose
        pose_lists.append(
            (
                np.array([0.08738169, 0.08474459, -0.10285583]),
                np.array([0.61735759, -0.17638474, 0.31953042, 0.69689192]),
            )
        )

        # Correction Step 1
        pose_lists.append(
            (
                np.array([0.08417361, 0.08523226, -0.10458831]),
                np.array([0.61557816, -0.19970439, 0.29569177, 0.70267209]),
            )
        )

        # Correction Step 2
        pose_lists.append(
            (
                np.array([0.08435525, 0.08498609, -0.10476889]),
                np.array([0.64382882, 0.35338942, -0.22771418, 0.63933295]),
            )
        )

        # Pre-insertion
        pose_lists.append(
            (np.array([0.10880398, 0.08446415, -0.0957803]), np.array([0.64635628, 0.3529954, -0.22424672, 0.63822505]))
        )
        wound_start = pose_lists[-1][0][1]
        # Insertion poses
        pose_lists.append(
            (
                np.array([0.1088187, 0.08370875, -0.09635209]),
                np.array([0.50746778, 0.49848271, -0.49660786, 0.49736514]),
            )
        )
        pose_lists.append(
            (np.array([0.13533885, 0.08416236, -0.09621308]), np.array([0.5720614, 0.41562694, -0.41562694, 0.5720614]))
        )
        pose_lists.append(
            (
                np.array([0.12961172, 0.08416236, -0.09685837]),
                np.array([0.61499985, 0.34896301, -0.34896301, 0.61499985]),
            )
        )
        pose_lists.append(
            (
                np.array([0.12417177, 0.08416236, -0.09876189]),
                np.array([0.65020432, 0.27791067, -0.27791067, 0.65020432]),
            )
        )
        pose_lists.append(
            (np.array([0.11929178, 0.08416236, -0.10182819]), np.array([0.6772321, 0.20336343, -0.20336343, 0.6772321]))
        )
        pose_lists.append(
            (
                np.array([0.11521647, 0.08416236, -0.10590351]),
                np.array([0.69574328, 0.12625879, -0.12625879, 0.69574328]),
            )
        )
        pose_lists.append(
            (np.array([0.11215017, 0.08416236, -0.1107835]), np.array([0.7055051, 0.04756637, -0.04756637, 0.7055051]))
        )
        pose_lists.append(
            (
                np.array([0.11024665, 0.08416236, -0.11622345]),
                np.array([0.70639477, -0.03172423, 0.03172423, 0.70639477]),
            )
        )
    elif psm_number == "2":
        # Add warmup poses
        pose_lists.append(
            (
                np.array([-0.01659861, 0.06651666, -0.09047811]),
                np.array([0.72054657, 0.28439698, 0.05348962, 0.63013479]),
            )
        )
        pose_lists.append(
            (np.array([0.00258789, 0.07366597, -0.10036293]), np.array([0.82623543, -0.1924434, 0.0961523, 0.52062971]))
        )
        pose_lists.append(
            (
                np.array([-0.02584431, 0.09027522, -0.08787239]),
                np.array([0.59117077, -0.50339768, -0.40847555, 0.47984958]),
            )
        )
        pose_lists.append(
            (
                np.array([-0.04668781, 0.04265233, -0.10490519]),
                np.array([0.20280202, 0.60934815, 0.58818048, 0.4915383]),
            )
        )
        # Add home pose
        pose_lists.append((np.array([-0.04668781, 0.04265233, -0.10490519]), np.array([0, 0, 0, 1])))
        pose_lists.append((np.array([0, 0, -0.13]), np.array([0, 0, 0, 1])))

        # Pre extraction grasp pose
        pose_lists.append(
            (
                np.array([-0.00327809, 0.06993814, -0.09375261]),
                np.array([7.07051926e-01, 5.99832506e-04, 1.70147858e-04, 7.07161357e-01]),
            )
        )

        # Move to grasp pose
        pose_lists.append(
            (
                np.array([-0.00331871, 0.07249852, -0.09361149]),
                np.array([7.07361560e-01, 1.03204488e-03, 3.02682179e-04, 7.06851092e-01]),
            )
        )
        pose_lists.append(
            (
                np.array([-0.00329718, 0.07776445, -0.09365492]),
                np.array([7.07164612e-01, 8.60931939e-04, 1.69561672e-04, 7.07048401e-01]),
            )
        )
        pose_lists.append(
            (
                np.array([-0.00329332, 0.08040503, -0.09367767]),
                np.array([7.07287031e-01, 9.15858349e-04, 2.75659770e-04, 7.06925839e-01]),
            )
        )
        pose_lists.append(
            (
                np.array([-0.0032874, 0.08302868, -0.09369996]),
                np.array([7.07203358e-01, 9.49169746e-04, 3.40021237e-04, 7.07009472e-01]),
            )
        )
        pose_lists.append(
            (
                np.array([-0.00326865, 0.0856655, -0.09373203]),
                np.array([7.07144990e-01, 7.97969507e-04, 5.85044373e-05, 7.07068118e-01]),
            )
        )
        pose_lists.append(
            (
                np.array([-0.00326997, 0.08831234, -0.09375984]),
                np.array([7.07034239e-01, 9.25223242e-04, 2.71736907e-04, 7.07178658e-01]),
            )
        )
        pose_lists.append(
            (
                np.array([-0.00326187, 0.09095559, -0.09376726]),
                np.array([7.07172469e-01, 9.13351335e-04, 2.37230268e-04, 7.07040458e-01]),
            )
        )
        pose_lists.append(
            (
                np.array([-0.00324995, 0.0936142, -0.0937894]),
                np.array([7.07151152e-01, 8.70938557e-04, 1.68840881e-04, 7.07061851e-01]),
            )
        )
        pose_lists.append(
            (
                np.array([-0.00324348, 0.09626458, -0.09379274]),
                np.array([7.07233951e-01, 8.48174702e-04, 1.36126872e-04, 7.06979066e-01]),
            )
        )

        # Extract needle out
        pose_lists.append(
            (
                np.array([0.01312951, 0.09889637, -0.09361976]),
                np.array([0.70562256, -0.00202415, -0.00583741, 0.70856096]),
            )
        )
        pose_lists.append(
            (
                np.array([0.01272332, 0.09898687, -0.09338105]),
                np.array([0.60499303, 0.03099817, 0.15263624, 0.78084872]),
            )
        )
        pose_lists.append(
            (
                np.array([0.01258553, 0.09910719, -0.09332384]),
                np.array([0.49590754, 0.01949667, 0.30733305, 0.81193719]),
            )
        )
        pose_lists.append(
            (np.array([0.01238249, 0.09911633, -0.09332727]), np.array([0.3984451, -0.0287268, 0.44561825, 0.80114958]))
        )
        pose_lists.append(
            (
                np.array([0.02895457, 0.09918231, -0.09324889]),
                np.array([0.39816653, -0.03562343, 0.44138289, 0.80335268]),
            )
        )

        # Lift needle up after extraction
        pose_lists.append(
            (
                np.array([0.02878806, 0.0992226, -0.09243341]),
                np.array([0.39894103, -0.03848371, 0.43890544, 0.80419343]),
            )
        )
        pose_lists.append(
            (
                np.array([0.02874347, 0.09923958, -0.09242115]),
                np.array([0.70465186, -0.00346142, -0.00595913, 0.70951974]),
            )
        )
        pose_lists.append(
            (
                np.array([0.02868079, 0.0993365, -0.09227627]),
                np.array([0.81629318, -0.00578801, -0.01058233, 0.57751187]),
            )
        )

        # Clear thread out for suture
        pose_lists.append(
            (
                np.array([0.13829023, 0.09931807, -0.08252595]),
                np.array([0.81630298, -0.00760414, -0.01556391, 0.57736418]),
            )
        )
        pose_lists.append(
            (
                np.array([-0.0232587, 0.0834717, -0.0774891]),
                np.array([0.81672468, -0.0008379, -0.00430813, 0.57701086]),
            )
        )

        # Sweep additional thread
        pose_lists.append(
            (
                np.array([0.01495802, 0.03011758, -0.10610778]),
                np.array([0.7064068, -0.00085404, -0.0037863, 0.70779543]),
            )
        )
        pose_lists.append(
            (
                np.array([0.01467714, 0.03013397, -0.10633704]),
                np.array([0.49731534, -0.49257601, 0.50018968, 0.50976132]),
            )
        )
        pose_lists.append(
            (
                np.array([0.02098723, 0.06112507, -0.09850433]),
                np.array([0.51330746, -0.48945411, 0.50420181, 0.49267703]),
            )
        )
        pose_lists.append(
            (
                np.array([-0.0062987, 0.09572576, -0.09258981]),
                np.array([0.5920103, -0.42891067, 0.42317844, 0.53523775]),
            )
        )

        pose_lists.append(
            (
                np.array([0.07067624, 0.09471856, -0.09430691]),
                np.array([0.56327813, -0.4771277, 0.46454174, 0.48915016]),
            )
        )
        pose_lists.append(
            (
                np.array([0.09067624, 0.09471856, -0.09430691]),
                np.array([0.56327813, -0.4771277, 0.46454174, 0.48915016]),
            )
        )
        pose_lists.append(
            (
                np.array([0.11067624, 0.09471856, -0.09430691]),
                np.array([0.56327813, -0.4771277, 0.46454174, 0.48915016]),
            )
        )
        pose_lists.append(
            (
                np.array([0.13067624, 0.09471856, -0.09430691]),
                np.array([0.56327813, -0.4771277, 0.46454174, 0.48915016]),
            )
        )

    else:
        print("Not implemented yet")

        import pdb

        pdb.set_trace()

    print("Now manually move the robot around. Call psm.get_current_pose() and append it to pose_lists")
    # for position, quaternion in pose_lists:
    #     psm.set_pose(position, quaternion)
    #     print("Position: " + str(position))
    #     print("Rotation: " + str(quaternion))
    #     import pdb

    #     pdb.set_trace()
    import pdb

    pdb.set_trace()

    return pose_lists, wound_start


def extend_sutures_psm2(pose_lists):
    total_pose_list = pose_lists[:6]
    suture_motions = pose_lists[6:]
    i = 0
    while i < NUM_SUTURES:
        total_pose_list.extend(suture_motions)
        i += 1
    return total_pose_list


def extend_sutures_psm1(pose_lists, wound_start):
    wound_end = wound_start - WOUND_LENGTH
    suture_ys = np.linspace(wound_start, wound_end, NUM_SUTURES)
    total_pose_list = pose_lists[:5]
    suture_motions = pose_lists[5:]
    pre_insertion_motions = suture_motions[:-9]
    insertion_motions = suture_motions[-9:]

    # Step 1: Extract the initial y value
    initial_y = insertion_motions[0][0][1]  # y-value of the first position

    # Step 2: Modify the list so that the y-values are relative deltas to the initial y
    delta_insertion_motions = []
    for position, quaternion in insertion_motions:
        delta_y = position[1] - initial_y  # Calculate delta relative to initial y
        modified_position = np.array([position[0], delta_y, position[2]])  # Keep x and z unchanged
        delta_insertion_motions.append((modified_position, quaternion))
    for suture_y in suture_ys:
        new_insertion_motions = copy.deepcopy(delta_insertion_motions)
        for position, orientation in new_insertion_motions:
            position[1] += suture_y  # Add the value of suture_input_y to the y-component
        new_suture_motions = copy.deepcopy(pre_insertion_motions)
        new_suture_motions.extend(new_insertion_motions)
        total_pose_list.extend(new_suture_motions)
    return total_pose_list


# Function to add random perturbations to positions and quaternions
def randomize_pose(position, quaternion, max_position_delta=0.0005, max_rotation_degrees=5):
    # Add random displacement in x, y, z within max_position_delta (1mm)
    position_delta = np.random.uniform(-max_position_delta, max_position_delta, 3)  # x, y, z
    new_position = position + position_delta

    # Create a random rotation (up to max_rotation_degrees)
    random_axis = np.random.uniform(-1, 1, 3)  # Random rotation axis (unit vector)
    random_axis /= np.linalg.norm(random_axis)  # Normalize the axis
    random_angle = np.radians(np.random.uniform(-max_rotation_degrees, max_rotation_degrees))  # Random angle in radians
    random_rotation = R.from_rotvec(random_axis * random_angle)  # Rotation from axis-angle representation
    new_quaternion = random_rotation * R.from_quat(quaternion)  # Apply the random rotation to the original quaternion

    # Return the new position and quaternion
    return new_position, new_quaternion.as_quat()


def gen_full_pose_list(num_samples, pose_lists):
    additional_iterations = int(np.ceil((num_samples - len(pose_lists)) / len(pose_lists)))
    total_pose_lists = copy.deepcopy(pose_lists)
    i = 0
    while i < additional_iterations:
        pose_list_copy = copy.deepcopy(pose_lists)
        pose_list_copy = [(randomize_pose(position, quaternion)) for position, quaternion in pose_list_copy]
        total_pose_lists.extend(pose_list_copy)
        i += 1
    return total_pose_lists


if __name__ == "__main__":
    psm_number = input("Which PSM are you calibrating for suturing workspace. Pick 1 or 2 ")
    if psm_number != "1" and psm_number != "2":
        print("Please type 1 or 2")
        exit()
    print(psm_number)

    # Manually found poses for 1 suture
    pose_lists, wound_start = set_suturing_pipeline_input(psm_number)
    if psm_number == "1":
        # Set it up to for 6 sutures
        pose_lists = extend_sutures_psm1(pose_lists, wound_start)
    elif psm_number == "2":
        pose_lists = extend_sutures_psm2(pose_lists)

    pose_lists = gen_full_pose_list(1000, pose_lists)
    q_target, pos_target = gen_sampling(
        pose_lists
    )  # Even tho the paper said 1800, we don't detect like 10-15% of them so we up it to 2k so we can get 1800 good data points
    print(np.shape(q_target))
    print(np.shape(pos_target))
    plot_position(pos_target)
    plot_joint(q_target)
    if not os.path.exists("shallow_and_deep_calibration_outputs"):
        os.makedirs("shallow_and_deep_calibration_outputs")
    np.save("shallow_and_deep_calibration_outputs/prime_psm" + psm_number + "_suture_pipeline_sampled", q_target)
