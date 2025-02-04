import numpy as np
from dvrk.motion.dvrkKinematics import dvrkKinematics
L1 = 0.4318
L2 = 0.4162
L3 = 0.0091
L4 = 0.0102
psm1_poses = []
psm1_joints = np.load('/home/davinci/dvrkCalibration/experiment/0_trajectory_extraction/shallow_and_deep_calibration_outputs/prime_psm1_random_sampled.npy')
for psm1_joint in psm1_joints:
    psm1_pos,psm1_quat = dvrkKinematics.joint_to_pose(joint=psm1_joint,L1=L1,L2=L2,L3=L3,L4=L4)
    psm1_pose = np.concatenate((psm1_pos,psm1_quat))
    psm1_poses.append(psm1_pose)
np_psm1_poses = np.array(psm1_poses)
np.save('/home/davinci/dvrkCalibration/experiment/0_trajectory_extraction/shallow_and_deep_calibration_outputs/prime_psm1_random_sampled_poses.npy',np_psm1_poses)

psm2_poses = []
psm2_joints = np.load('/home/davinci/dvrkCalibration/experiment/0_trajectory_extraction/shallow_and_deep_calibration_outputs/prime_psm2_random_sampled.npy')
for psm2_joint in psm2_joints:
    psm2_pos,psm2_quat = dvrkKinematics.joint_to_pose(joint=psm2_joint,L1=L1,L2=L2,L3=L3,L4=L4)
    psm2_pose = np.concatenate((psm2_pos,psm2_quat))
    psm2_poses.append(psm2_pose)
np_psm2_poses = np.array(psm2_poses)
np.save('/home/davinci/dvrkCalibration/experiment/0_trajectory_extraction/shallow_and_deep_calibration_outputs/prime_psm2_random_sampled_poses.npy',np_psm2_poses)

psm1_to_zivid = np.load('/home/davinci/dvrkCalibration/experiment/0_trajectory_extraction/shallow_and_deep_calibration_outputs/psm1_robot_to_zivid.npy')
psm2_to_zivid = np.load('/home/davinci/dvrkCalibration/experiment/0_trajectory_extraction/shallow_and_deep_calibration_outputs/psm2_robot_to_zivid.npy')
psm1_to_psm2 = psm1_to_zivid @ np.linalg.inv(psm2_to_zivid)
import pdb
pdb.set_trace()
