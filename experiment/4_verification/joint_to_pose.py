import numpy as np
import PyKDL

def fk_position(q1, q2, q3, q4, q5, q6, L1=0, L2=0, L3=0, L4=0):
    xtip = L2 * np.cos(q2) * np.sin(q1) - L1 * np.cos(q2) * np.sin(q1) + q3 * np.cos(q2) * np.sin(q1) + L3 * np.cos(
        q2) * np.cos(q5) * np.sin(
        q1) + L4 * np.cos(q1) * np.cos(q4) * np.sin(q6) - L3 * np.cos(q1) * np.sin(q4) * np.sin(q5) + L4 * np.cos(
        q2) * np.cos(q5) * np.cos(
        q6) * np.sin(q1) - L4 * np.cos(q1) * np.cos(q6) * np.sin(q4) * np.sin(q5) - L3 * np.cos(q4) * np.sin(
        q1) * np.sin(q2) * np.sin(
        q5) - L4 * np.sin(q1) * np.sin(q2) * np.sin(q4) * np.sin(q6) - L4 * np.cos(q4) * np.cos(q6) * np.sin(
        q1) * np.sin(q2) * np.sin(q5)

    ytip = L1 * np.sin(q2) - L2 * np.sin(q2) - q3 * np.sin(q2) - L3 * np.cos(q5) * np.sin(q2) - L3 * np.cos(
        q2) * np.cos(q4) * np.sin(
        q5) - L4 * np.cos(q5) * np.cos(q6) * np.sin(q2) - L4 * np.cos(q2) * np.sin(q4) * np.sin(q6) - L4 * np.cos(
        q2) * np.cos(q4) * np.cos(
        q6) * np.sin(q5)

    ztip = L1 * np.cos(q1) * np.cos(q2) - L2 * np.cos(q1) * np.cos(q2) - q3 * np.cos(q1) * np.cos(q2) - L3 * np.cos(
        q1) * np.cos(q2) * np.cos(
        q5) + L4 * np.cos(q4) * np.sin(q1) * np.sin(q6) - L3 * np.sin(q1) * np.sin(q4) * np.sin(q5) + L3 * np.cos(
        q1) * np.cos(q4) * np.sin(
        q2) * np.sin(q5) + L4 * np.cos(q1) * np.sin(q2) * np.sin(q4) * np.sin(q6) - L4 * np.cos(q6) * np.sin(
        q1) * np.sin(q4) * np.sin(
        q5) - L4 * np.cos(q1) * np.cos(q2) * np.cos(q5) * np.cos(q6) + L4 * np.cos(q1) * np.cos(q4) * np.cos(
        q6) * np.sin(q2) * np.sin(q5)
    return [xtip, ytip, ztip]


def fk_orientation(j1, j2, j3, j4, j5, j6):
    # R08
    r11 = np.cos(j1) * np.cos(j4) * np.cos(j6) - np.cos(j2) * np.cos(j5) * np.sin(j1) * np.sin(j6) - np.cos(
        j6) * np.sin(j1) * np.sin(j2) * np.sin(j4) + np.cos(j1) * np.sin(j4) * np.sin(j5) * np.sin(j6) + np.cos(
        j4) * np.sin(j1) * np.sin(j2) * np.sin(j5) * np.sin(j6)
    r12 = np.cos(j1) * np.cos(j5) * np.sin(j4) + np.cos(j2) * np.sin(j1) * np.sin(j5) + np.cos(j4) * np.cos(
        j5) * np.sin(j1) * np.sin(j2)
    r13 = np.cos(j1) * np.cos(j6) * np.sin(j4) * np.sin(j5) - np.cos(j2) * np.cos(j5) * np.cos(j6) * np.sin(
        j1) - np.cos(j1) * np.cos(j4) * np.sin(j6) + np.sin(j1) * np.sin(j2) * np.sin(j4) * np.sin(j6) + np.cos(
        j4) * np.cos(j6) * np.sin(j1) * np.sin(j2) * np.sin(j5)
    r21 = np.cos(j5) * np.sin(j2) * np.sin(j6) - np.cos(j2) * np.cos(j6) * np.sin(j4) + np.cos(j2) * np.cos(
        j4) * np.sin(j5) * np.sin(j6)
    r22 = np.cos(j2) * np.cos(j4) * np.cos(j5) - np.sin(j2) * np.sin(j5)
    r23 = np.cos(j5) * np.cos(j6) * np.sin(j2) + np.cos(j2) * np.sin(j4) * np.sin(j6) + np.cos(j2) * np.cos(
        j4) * np.cos(j6) * np.sin(j5)
    r31 = np.cos(j4) * np.cos(j6) * np.sin(j1) + np.cos(j1) * np.cos(j2) * np.cos(j5) * np.sin(j6) + np.cos(
        j1) * np.cos(j6) * np.sin(j2) * np.sin(j4) + np.sin(j1) * np.sin(j4) * np.sin(j5) * np.sin(j6) - np.cos(
        j1) * np.cos(j4) * np.sin(j2) * np.sin(j5) * np.sin(j6)
    r32 = np.cos(j5) * np.sin(j1) * np.sin(j4) - np.cos(j1) * np.cos(j2) * np.sin(j5) - np.cos(j1) * np.cos(
        j4) * np.cos(j5) * np.sin(j2)
    r33 = np.cos(j1) * np.cos(j2) * np.cos(j5) * np.cos(j6) - np.cos(j4) * np.sin(j1) * np.sin(j6) - np.cos(
        j1) * np.sin(j2) * np.sin(j4) * np.sin(j6) + np.cos(j6) * np.sin(j1) * np.sin(j4) * np.sin(j5) - np.cos(
        j1) * np.cos(j4) * np.cos(j6) * np.sin(j2) * np.sin(j5)

    R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
    return R

file_path = 'random/raw/'
q_des = np.load(file_path + 'q_des_raw.npy')    # desired joint angles: [q1, ..., q6]
q_act = np.load(file_path + 'q_act_raw.npy')    # actual joint angles: [q1, ..., q6]
t_stamp = np.load(file_path + 't_stamp_raw.npy')    # measured time (sec)
print('data length: ', len(q_des))

L1 = 0.4318  # Rcc (m)
L2 = 0.4162  # tool
L3 = 0.0091  # pitch ~ yaw (m)
L4 = 0.0102  # yaw ~ tip (m)
pos_des = []
pos_act = []
quat_des = []
quat_act = []
for q in q_des:
    pos_des.append(fk_position(q1=q[0], q2=q[1], q3=q[2], q4=0, q5=0, q6=0, L1=L1, L2=L2, L3=0, L4=0))
    R = fk_orientation(q[0], q[1], q[2], q[3], q[4], q[5])
    R_matrix = PyKDL.Rotation(R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2])
    quat_des.append(list(R_matrix.GetQuaternion()))

for q in q_act:
    pos_act.append(fk_position(q1=q[0], q2=q[1], q3=q[2], q4=0, q5=0, q6=0, L1=L1, L2=L2, L3=0, L4=0))
    R = fk_orientation(q[0], q[1], q[2], q[3], q[4], q[5])
    R_matrix = PyKDL.Rotation(R[0,0], R[0,1], R[0,2], R[1,0], R[1,1], R[1,2], R[2,0], R[2,1], R[2,2])
    quat_act.append(list(R_matrix.GetQuaternion()))

np.save('pos_des', pos_des)
np.save('pos_act', pos_act)
np.save('quat_des', quat_des)
np.save('quat_act', quat_act)
print ("pose saved")