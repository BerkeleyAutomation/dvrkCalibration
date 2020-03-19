import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.metrics import mean_squared_error
plt.style.use('seaborn-whitegrid')

def fk_position(joints):
    q1 = joints[0]
    q2 = joints[1]
    q3 = joints[2]
    q4 = joints[3]
    q5 = joints[4]
    q6 = joints[5]
    L1 = 0.4318  # Rcc (m)
    L2 = 0.4162  # tool
    L3 = 0.0091  # pitch ~ yaw (m)
    L4 = 0.0102  # yaw ~ tip (m)
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

def convert_to_cartesian(joints):
    pos = []
    for q in joints:
        pos.append(fk_position(q))
    return pos

def plot_joint(t, q_des, q_act):
    # RMSE = []
    # for i in range(6):
    #     RMSE = np.sqrt(np.sum((q_des[:,i] - q_act[:,i]) ** 2) / len(q_des[:,i]))
    # print("RMSE=", RMSE)

    # Create plot
    # plt.title('joint angle')
    ax = plt.subplot(611)
    plt.plot(t, q_des[:,0]*180./np.pi, 'b-')
    plt.plot(t, q_act[:, 0] * 180. / np.pi, 'r-')
    plt.legend(loc='upper center', bbox_to_anchor=(0.9,2))
    # plt.ylim([35, 62])
    ax.set_xticklabels([])
    plt.ylabel('q1 ($^\circ$)')

    ax = plt.subplot(612)
    plt.plot(t, q_des[:,1]*180./np.pi, 'b-', t, q_act[:, 1] * 180. / np.pi, 'r-')
    # plt.ylim([-10, 12])
    ax.set_xticklabels([])
    plt.ylabel('q2 ($^\circ$)')

    ax = plt.subplot(613)
    plt.plot(t, q_des[:, 2], 'b-', t, q_act[:, 2], 'r-')
    # plt.ylim([0.14, 0.23])
    ax.set_xticklabels([])
    plt.ylabel('q3 (m)')

    ax = plt.subplot(614)
    plt.plot(t, q_des[:, 3]*180./np.pi, 'b-', t, q_act[:, 3]*180./np.pi, 'r-')
    # plt.ylim([-90, 70])
    ax.set_xticklabels([])
    plt.ylabel('q4 ($^\circ$)')

    ax = plt.subplot(615)
    plt.plot(t, q_des[:, 4]*180./np.pi, 'b-', t, q_act[:, 4]*180./np.pi, 'r-')
    # plt.ylim([-60, 60])
    ax.set_xticklabels([])
    plt.ylabel('q5 ($^\circ$)')

    plt.subplot(616)
    plt.plot(t, q_des[:, 5]*180./np.pi, 'b-', t, q_act[:, 5]*180./np.pi, 'r-')
    # plt.ylim([-60, 60])
    # plt.legend(['desired', 'actual'])
    plt.ylabel('q6 ($^\circ$)')
    plt.xlabel('(sample)')
    plt.show()

# Trajectory check
file_path = './result/random_user/wo_model/'
# file_path = './result/random_user/sampling_2_ensemble/'
q_des = np.load(file_path + 'q_des_raw.npy') # desired joint angles: [q1, ..., q6]
q_act = np.load(file_path + 'q_act_raw.npy')

pos_des = np.array(convert_to_cartesian(q_des))
pos_act = np.array(convert_to_cartesian(q_act))

# RMSE error calc
RMSE = np.sqrt(np.sum((pos_des - pos_act) ** 2)/len(pos_des))
print("RMSE=", RMSE, '(m)')

t = range(len(q_des))
plot_joint(t, q_des, q_act)