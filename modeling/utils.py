import os.path as osp
import numpy as np


def load_data(data_dir):
    joint_actual = np.load(osp.join(data_dir, "joint_act.npy"))
    joint_desired = np.load(osp.join(data_dir, "joint_des.npy"))
    position_actual = np.load(osp.join(data_dir, "position_act.npy"))
    position_desired = np.load(osp.join(data_dir, "position_des.npy"))
    quaternion_actual = np.load(osp.join(data_dir, "quaternion_act.npy"))
    quaternion_desired = np.load(osp.join(data_dir, "quaternion_des.npy"))

    return {
        "joint_actual": joint_actual,
        "position_actual": position_actual,
        "quaternion_actual": quaternion_actual,
        "joint_desired": joint_desired,
        "position_desired": position_desired,
        "quaternion_desired": quaternion_desired
    }


def join_data(dataset1, dataset2):
    new_dataset = {
    }
    for k in dataset1:
        new_dataset[k] = np.vstack((dataset1[k], dataset2[k]))
    return new_dataset


def flatten_data(dataset, keys):
    ndata = []
    for k in keys:
        ndata.append(dataset[k])
    return np.hstack(ndata)


def incorporate_history(data_arr_in, data_arr_out, horizon):
    if horizon == 1:
        return data_arr_in, data_arr_out
    return (np.hstack([data_arr_in[i:data_arr_in.shape[0] - horizon + i + 1] for i in range(horizon)]),
            data_arr_out[:-horizon+1])
