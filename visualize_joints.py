from dvrk.motion.dvrkArm import dvrkArm
from autolab_core import RigidTransform
from dvrk.motion.dvrkTypes import dvrkTypes
import numpy as np
import time

JAW_CLOSE_ANGLE = [-0.3]

psm2 = dvrkArm("/PSM2", dvrk_type=dvrkTypes.LARGE_NEEDLE_DRIVER, use_rnn=False)

psm2_pose = np.array([0, 0, -0.13]), np.array([0, 0, 0, 1])
psm2.set_pose(*psm2_pose)
curr_joints = psm2.get_current_joint()
curr_joints[3:] = [0, 0, 0]
psm2.set_joint(joint=curr_joints)
psm2.set_jaw(jaw=JAW_CLOSE_ANGLE)

input("Move joints")
while True:
    curr_joints = psm2.get_current_joint()
    curr_joints[3:] = [-1, 0, 0]
    psm2.set_joint(joint=curr_joints)
    psm2.set_jaw(jaw=JAW_CLOSE_ANGLE)
    time.sleep(1.5)
    curr_joints = psm2.get_current_joint()
    curr_joints[3:] = [1, 0, 0]
    psm2.set_joint(joint=curr_joints)
    psm2.set_jaw(jaw=JAW_CLOSE_ANGLE)
    time.sleep(1.5)
