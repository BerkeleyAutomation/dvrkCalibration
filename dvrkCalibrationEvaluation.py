import os

from datetime import datetime
import numpy as np
import cv2
import time

from dvrk.vision import vision_constants as cst
from dvrk.vision.cameras.Camera import Camera
from dvrk.motion.dvrkArm import dvrkArm
from autolab_core import RigidTransform
from dvrk.motion.dvrkTypes import dvrkTypes
from dvrk.vision.cameras.AlliedVisionUtils import AlliedVisionUtils
import dvrk.utils.CmnUtil as utils

JAW_OPEN_ANGLE = [np.pi / 2]
JAW_CLOSE_ANGLE = [-0.3]  # Angle for closing jaw to grasp

home_joints = None


def random_quaternion():
    """
    TODO: Consider refactoring. New function (fully tested, ready for copy-paste) in psm_random_dance.py
    """
    q = None
    while q is None:
        # Note: comment out first line of each pair for horizontal positions, second line for vertical ones

        # quaternion = utils.euler_to_quaternion(np.random.uniform([np.pi/2 - .1, 0, np.pi/4], [np.pi/2 + .1, np.pi*2, 7*np.pi/4]))
        quaternion = utils.euler_to_quaternion(np.random.uniform([0, 0, 0], [np.pi * 2, np.pi * 2, np.pi * 2]))

        R = utils.quaternion_to_R(quaternion)

        # ax = 2
        ax = 1

        proj = np.array([[0], [0], [0]])
        proj[ax] = 1
        dir = np.dot(R, proj)

        # if dir[ax] > min_z:
        if dir[2] > 0.85:
            q = quaternion

    return q


def plot_psm_pos_on_img(img, psm_p_robot, tf_cam_psm, K, D):
    rvec_cam_psm, _ = cv2.Rodrigues(tf_cam_psm.rotation)

    psm_p_cam, _ = cv2.projectPoints(psm_p_robot, rvec_cam_psm, tf_cam_psm.translation, K, D)
    psm_p_cam = psm_p_cam.squeeze().astype(int)
    img = cv2.circle(img, (psm_p_cam[0], psm_p_cam[1]), 5, (255, 0, 0), -1)
    return img


def single_psm_warmup(psm_number, psm):
    psm.set_jaw(JAW_CLOSE_ANGLE)
    psm_warmup_points = []
    psm_warmup_quats = []
    if psm_number == "1":
        psm_warmup_points = [
            np.array([[0.01], [0.0], [-0.12], [1]]),
            np.array([[0.01], [0.05], [-0.135], [1]]),
            np.array([[0.05], [0.0], [-0.11], [1]]),
            np.array([[0.05], [0.05], [-0.13], [1]]),
        ]
        psm_warmup_quats = [
            np.array([0.32743764, 0.79021532, 0.43636955, 0.27915223]),
            np.array([0.1098817, 0.54847876, 0.7428689, 0.3677538]),
            np.array([0.06069057, 0.49024802, 0.86942407, 0.00867864]),
            np.array([0.63309931, -0.3134932, -0.31293999, 0.63480379]),
        ]
    elif psm_number == "2":
        print("Not implemented yet")
        import pdb

        pdb.set_trace()
        psm_warmup_points = [
            np.array([[-0.01659861], [0.06651666], [-0.09047811], [1]]),
            np.array([[0.00258789], [0.07366597], [-0.10036293], [1]]),
            np.array([[-0.02584431], [0.09027522], [-0.08787239], [1]]),
            np.array([[-0.04668781], [0.04265233], [-0.10490519], [1]]),
        ]
        psm_warmup_quats = [
            np.array([0.72054657, 0.28439698, 0.05348962, 0.63013479]),
            np.array([0.82623543, -0.1924434, 0.0961523, 0.52062971]),
            np.array([0.59117077, -0.50339768, -0.40847555, 0.47984958]),
            np.array([0.20280202, 0.60934815, 0.58818048, 0.4915383]),
        ]
    else:
        print("Please select PSM1 or PSM2")
        exit()
    for p1, q1 in zip(psm_warmup_points, psm_warmup_quats):
        psm1_pos = np.array([p1[0, 0] / p1[3, 0], p1[1, 0] / p1[3, 0], p1[2, 0] / p1[3, 0]])
        psm1_quat = q1
        psm.set_pose(psm1_pos, psm1_quat)
        time.sleep(1)
    psm_pose = np.array([0, 0, -0.13]), np.array([0, 0, 0, 1])
    psm.set_pose(*psm_pose)
    curr_joints = psm.get_current_joint()
    curr_joints[3:] = [0, 0, 0]
    home_joints = np.copy(curr_joints)
    psm.set_joint(joint=curr_joints)
    psm.set_jaw(jaw=JAW_CLOSE_ANGLE)
    return home_joints


root_path = os.path.dirname(os.path.abspath(__file__))

left_P_matrix_fname = os.path.join(
    root_path,
    "experiment/0_trajectory_extraction/allied_vision_calibration_outputs/allied_vision_matrices/left_P_mat.npy",
)
right_P_matrix_fname = os.path.join(
    root_path,
    "experiment/0_trajectory_extraction/allied_vision_calibration_outputs/allied_vision_matrices/right_P_mat.npy",
)
P_av_left = np.load(left_P_matrix_fname)
P_av_right = np.load(right_P_matrix_fname)
D_undistorted = np.zeros((4,), np.float64)

tf_av_left_zivid = RigidTransform.load(
    os.path.join(
        root_path,
        "experiment/0_trajectory_extraction/allied_vision_to_zivid_calibration_outputs/av_zivid_output_matrices/av_left_to_zivid.tf",
    )
)
tf_av_right_zivid = RigidTransform.load(
    os.path.join(
        root_path,
        "experiment/0_trajectory_extraction/allied_vision_to_zivid_calibration_outputs/av_zivid_output_matrices/av_right_to_zivid.tf",
    )
)


psm_number_input = input("Which PSM are you using? (1 or 2)")
tf_zivid_psm = None
pose_lists = []

if psm_number_input == "1":
    tf_zivid_psm = np.load(
        os.path.join(
            root_path, "experiment/0_trajectory_extraction/shallow_and_deep_calibration_outputs/psm1_robot_to_zivid.npy"
        )
    )
    # Move to suture clear pose
    pose_lists.append(
        (np.array([0.11316637, 0.01303404, -0.04916395]), np.array([0.12616457, -0.60540458, -0.22937437, 0.75163502]))
    )

    # Add home pose
    pose_lists.append((np.array([0, 0, -0.13]), np.array([0, 0, 0, 1])))

    # Go to pre-handover pose
    pose_lists.append(
        (np.array([0.08880125, 0.05858506, -0.10344337]), np.array([0.61735759, -0.17638474, 0.31953042, 0.69689192]))
    )

    # Go to handover pose
    pose_lists.append(
        (np.array([0.08738169, 0.08474459, -0.10285583]), np.array([0.61735759, -0.17638474, 0.31953042, 0.69689192]))
    )

    # Correction Step 1
    pose_lists.append(
        (np.array([0.08417361, 0.08523226, -0.10458831]), np.array([0.61557816, -0.19970439, 0.29569177, 0.70267209]))
    )

    # Correction Step 2
    pose_lists.append(
        (np.array([0.08435525, 0.08498609, -0.10476889]), np.array([0.64382882, 0.35338942, -0.22771418, 0.63933295]))
    )

    # Pre-insertion
    pose_lists.append(
        (np.array([0.10880398, 0.08446415, -0.0957803]), np.array([0.64635628, 0.3529954, -0.22424672, 0.63822505]))
    )
    # Insertion poses
    pose_lists.append(
        (np.array([0.1088187, 0.08370875, -0.14635209]), np.array([0.50746778, 0.49848271, -0.49660786, 0.49736514]))
    )
    pose_lists.append(
        (np.array([0.13533885, 0.08416236, -0.14621308]), np.array([0.5720614, 0.41562694, -0.41562694, 0.5720614]))
    )
    pose_lists.append(
        (np.array([0.12961172, 0.08416236, -0.14685837]), np.array([0.61499985, 0.34896301, -0.34896301, 0.61499985]))
    )
    pose_lists.append(
        (np.array([0.12417177, 0.08416236, -0.14876189]), np.array([0.65020432, 0.27791067, -0.27791067, 0.65020432]))
    )
    pose_lists.append(
        (np.array([0.11929178, 0.08416236, -0.13182819]), np.array([0.6772321, 0.20336343, -0.20336343, 0.6772321]))
    )
    pose_lists.append(
        (np.array([0.11521647, 0.08416236, -0.13590351]), np.array([0.69574328, 0.12625879, -0.12625879, 0.69574328]))
    )
    pose_lists.append(
        (np.array([0.11215017, 0.08416236, -0.1407835]), np.array([0.7055051, 0.04756637, -0.04756637, 0.7055051]))
    )
    pose_lists.append(
        (np.array([0.11024665, 0.08416236, -0.14622345]), np.array([0.70639477, -0.03172423, 0.03172423, 0.70639477]))
    )

elif psm_number_input == "2":
    tf_zivid_psm = np.load(
        os.path.join(
            root_path, "experiment/0_trajectory_extraction/shallow_and_deep_calibration_outputs/psm2_robot_to_zivid.npy"
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
            np.array([-0.00326865, 0.0856655, -0.09373203]),
            np.array([7.07144990e-01, 7.97969507e-04, 5.85044373e-05, 7.07068118e-01]),
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
    print("Not implemented yet")
    import pdb

    pdb.set_trace()
else:
    print("Please select 1 or 2")
    exit()
psm_points = [pose[0] for pose in pose_lists]

dvrk_type_input = input(
    "Which PSM driver type are you calibrating. Large SutureCut Needle Driver (1) or Large Needle Driver (2)"
)
dvrk_type = None
if dvrk_type_input == "1":
    dvrk_type = dvrkTypes.LRG_SUTURECUT_NEEDLE_DRIVER
elif dvrk_type_input == "2":
    dvrk_type = dvrkTypes.LARGE_NEEDLE_DRIVER
else:
    print("Please select 1 or 2")
    exit()

psm = dvrkArm("/PSM" + psm_number_input, dvrk_type=dvrk_type, use_rnn=True)

home_joints = single_psm_warmup(psm_number=psm_number_input, psm=psm)
points_in_psm_frame = np.array(psm_points)

tf_zivid_psm = np.linalg.inv(tf_zivid_psm)

tf_zivid_psm = RigidTransform(
    rotation=tf_zivid_psm[:3, :3],
    translation=tf_zivid_psm[:3, 3],
    from_frame="psm" + str(psm_number_input),
    to_frame="zivid",
)

# Compute desired transforms
tf_av_left_psm = tf_av_left_zivid * tf_zivid_psm
tf_av_right_psm = tf_av_right_zivid * tf_zivid_psm

camera_input = input("Which camera do you want to use? (Zivid or AV)")
if camera_input == "AV":
    psm_points_homogenous = [np.append(position, 1) for position in psm_points]
    psm_points_homogenous = np.array(psm_points_homogenous)
    points_in_av_left_frame_homogenous = tf_av_left_psm.matrix @ psm_points_homogenous.T
    points_in_av_left_frame_homogenous = points_in_av_left_frame_homogenous.T
    points_in_av_left_frame = points_in_av_left_frame_homogenous[:, :3] / points_in_av_left_frame_homogenous[
        :, 3
    ].reshape(-1, 1)
    pixels, _ = cv2.projectPoints(
        points_in_av_left_frame,
        rvec=np.array([0.0, 0.0, 0.0]),
        tvec=np.array([0.0, 0.0, 0.0]),
        cameraMatrix=P_av_left[:, :3],
        distCoeffs=D_undistorted,
    )
    pixels = np.round(pixels).astype(int).squeeze()
    av_util = AlliedVisionUtils()
    av_cam = Camera(cst.ALLIED_VISION, zivid_cam_choice="inclined", zivid_capture_type="2d", rectify_img=False)
    time.sleep(2)
    print(" ")
    print("AV Stereo ready!")
    psm_point_index = 0
    while True:
        img_av_left, img_av_right = av_cam.capture()

        if len(img_av_left) == 0 or len(img_av_right) == 0:
            time.sleep(0.1)
            continue
        img_av_left = av_util.rectify_single(img_av_left, is_left=True)
        img_av_right = av_util.rectify_single(img_av_right, is_left=False)
        psm1_p_robot, _ = psm.get_current_pose()

        if psm_point_index - 1 >= 0:
            img_av_left = cv2.circle(img_av_left, pixels[psm_point_index - 1], 5, (0, 255, 0), -1)
        img_av_left = plot_psm_pos_on_img(img_av_left, psm1_p_robot, tf_av_left_psm, P_av_left[:, :-1], D_undistorted)
        cv2.imshow("End Effector Tracking AV_LEFT. Press q to exit.", img_av_left)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            av_cam.stop()
            break
        elif key == ord(" "):
            if psm_point_index >= len(pose_lists):
                av_cam.stop()
                exit()
            psm_pose = pose_lists[psm_point_index]
            psm.set_pose(*psm_pose)
            inverse_model_error = np.linalg.norm(psm.get_current_pose()[0] - psm_pose[0]) * 1000
            print("Inverse model error: " + str(inverse_model_error) + " mm.")
            psm_point_index += 1
elif camera_input == "Zivid":
    points_in_zivid_frame_homogenous = tf_zivid_psm.matrix @ psm_points.T
    points_in_zivid_frame_homogenous = points_in_zivid_frame_homogenous.T
    points_in_zivid_frame = points_in_zivid_frame_homogenous[:, :3] / points_in_zivid_frame_homogenous[:, 3].reshape(
        -1, 1
    )
    zivid_cam = Camera(cst.ZIVID)
    K_zivid = zivid_cam.cam.intrinsics_
    D_zivid = zivid_cam.cam.distortion_coefficients_
    time.sleep(2)
    print(" ")
    print("Zivid ready!")
    psm_point_index = 0
    pixels, _ = cv2.projectPoints(
        points_in_zivid_frame,
        rvec=np.array([0.0, 0.0, 0.0]),
        tvec=np.array([0.0, 0.0, 0.0]),
        cameraMatrix=K_zivid,
        distCoeffs=D_zivid,
    )
    pixels = np.round(pixels).astype(int).squeeze()
    while True:
        img_zivid = zivid_cam.capture()

        psm1_p_robot, _ = psm.get_current_pose()

        if psm_point_index - 1 >= 0:
            img_zivid = cv2.circle(img_zivid, pixels[psm_point_index - 1], 5, (0, 255, 0), -1)
        img_zivid = plot_psm_pos_on_img(img_zivid, psm1_p_robot, tf_zivid_psm, K_zivid, D_zivid)
        cv2.imshow("End Effector Tracking ZIVID. Press q to exit.", img_zivid)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):

            psm.set_joint(joint=np.copy(home_joints))
            time.sleep(1)
            if psm_point_index >= len(psm_points):
                exit()
            psm_pose = pose_lists[psm_point_index]
            psm.set_pose(*psm_pose)
            psm_point_index += 1
else:
    print("Please type Zivid or AV")
    exit()
