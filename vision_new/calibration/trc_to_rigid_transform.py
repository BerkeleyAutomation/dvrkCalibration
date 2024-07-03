import os
import numpy as np
from autolab_core import RigidTransform

from dvrk.vision import vision_constants as cst

POSE_SAVE_DIR = os.path.join(cst.DATA_PATH, "aruco_calib_results_rect_new")
TRC_NP_FNAME = "Trc_inclined_PSM2.npy"
TF_ZIVID_PSM_FNAME = "tf_zivid_psm2.tf"
PSM_FRAME = "psm2"


def np_to_rigid_transform(np_tf_mat: np.ndarray, from_frame: str, to_frame: str) -> RigidTransform:
    rigid_transform = RigidTransform(
        rotation=np_tf_mat[0:3, 0:3], translation=np_tf_mat[0:3, 3], from_frame=from_frame, to_frame=to_frame
    )
    return rigid_transform


def main():
    # Load dvrk deep calibration script output and convert to RigidTransform
    zivid2robot_np = np.load(os.path.join(POSE_SAVE_DIR, TRC_NP_FNAME))
    zivid2robot_tf = np_to_rigid_transform(zivid2robot_np, "zivid", PSM_FRAME)
    robot2zivid_tf = zivid2robot_tf.inverse()
    robot2zivid_tf.save(os.path.join(POSE_SAVE_DIR, TF_ZIVID_PSM_FNAME))
    print(robot2zivid_tf)


if __name__ == "__main__":
    main()
