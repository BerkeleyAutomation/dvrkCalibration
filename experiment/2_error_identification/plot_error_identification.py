import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

plt.style.use('seaborn-whitegrid')
# plt.style.use('bmh')
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=13)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=17)    # legend fontsize
plt.rc('figure', titlesize=10)  # fontsize of the figure title

def plot_joint(t, q_des, q_act):
    RMSE = []
    for i in range(6):
        RMSE = np.sqrt(np.sum((q_des[:,i] - q_act[:,i]) ** 2) / len(q_des[:,i]))
    print("RMSE=", RMSE)

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
    plt.xlabel('sample number')
    plt.show()


def plot_joint_compare(q_des1, q_act1, q_des2, q_act2):
    RMSE1 = []
    for i in range(6):
        RMSE1.append(np.sqrt(np.sum((q_des1[:, i] - q_act1[:, i]) ** 2) / len(q_des1[:, i])))
    print("RMSE1=", RMSE1)

    RMSE2 = []
    for i in range(6):
        RMSE2.append(np.sqrt(np.sum((q_des2[:, i] - q_act2[:, i]) ** 2) / len(q_des2[:, i])))
    print("RMSE2=", RMSE2)

    # Create plot
    # plt.title('joint angle')
    plt.figure(0)
    ax = plt.subplot2grid((6, 2), (0, 0))
    plt.plot(q_des1[:,0]*180./np.pi, 'b-')
    plt.plot(q_act1[:, 0] * 180. / np.pi, 'r-')
    plt.legend(loc='upper center', bbox_to_anchor=(0.9,2))
    ax.set_xticklabels([])
    plt.ylabel('q1 ($^\circ$)')

    ax = plt.subplot2grid((6, 2), (1, 0))
    plt.plot(q_des1[:, 1]*180./np.pi, 'b-')
    plt.plot(q_act1[:, 1]*180./np.pi, 'r-')
    ax.set_xticklabels([])
    plt.ylabel('q2 ($^\circ$)')

    ax = plt.subplot2grid((6, 2), (2, 0))
    plt.plot(q_des1[:, 2], 'b-')
    plt.plot(q_act1[:, 2], 'r-')
    ax.set_xticklabels([])
    plt.ylabel('q3 (m)')

    ax = plt.subplot2grid((6, 2), (3, 0))
    plt.plot(q_des1[:, 3]*180./np.pi, 'b-')
    plt.plot(q_act1[:, 3]*180./np.pi, 'r-')
    ax.set_xticklabels([])
    plt.ylabel('q4 ($^\circ$)')

    ax = plt.subplot2grid((6, 2), (4, 0))
    plt.plot(q_des1[:, 4]*180./np.pi, 'b-')
    plt.plot(q_act1[:, 4]*180./np.pi, 'r-')
    ax.set_xticklabels([])
    plt.ylabel('q5 ($^\circ$)')

    ax = plt.subplot2grid((6, 2), (5, 0))
    plt.plot(q_des1[:, 5]*180./np.pi, 'b-')
    plt.plot(q_act1[:, 5]*180./np.pi, 'r-')
    plt.ylabel('q6 ($^\circ$)')
    plt.xlabel('sample number')

    ax = plt.subplot2grid((6, 2), (0, 1))
    plt.plot(q_des2[:,0]*180./np.pi, 'b-')
    plt.plot(q_act2[:, 0] * 180. / np.pi, 'r-')
    ax.set_xticklabels([])

    ax = plt.subplot2grid((6, 2), (1, 1))
    plt.plot(q_des2[:, 1]*180./np.pi, 'b-')
    plt.plot(q_act2[:, 1]*180./np.pi, 'r-')
    ax.set_xticklabels([])

    ax = plt.subplot2grid((6, 2), (2, 1))
    plt.plot(q_des2[:, 2], 'b-')
    plt.plot(q_act2[:, 2], 'r-')
    ax.set_xticklabels([])

    ax = plt.subplot2grid((6, 2), (3, 1))
    plt.plot(q_des2[:, 3]*180./np.pi, 'b-')
    plt.plot(q_act2[:, 3]*180./np.pi, 'r-')
    ax.set_xticklabels([])

    ax = plt.subplot2grid((6, 2), (4, 1))
    plt.plot(q_des2[:, 4]*180./np.pi, 'b-')
    plt.plot(q_act2[:, 4]*180./np.pi, 'r-')
    ax.set_xticklabels([])

    ax = plt.subplot2grid((6, 2), (5, 1))
    plt.plot(q_des2[:, 5]*180./np.pi, 'b-')
    plt.plot(q_act2[:, 5]*180./np.pi, 'r-')
    plt.xlabel('sample number')
    plt.show()

def plot_joint_compare2(q_des1, q_act1, q_des2, q_act2):
    RMSE1 = []
    for i in range(2):
        RMSE1.append(np.sqrt(np.sum((q_des1[:, i] - q_act1[:, i]) ** 2) / len(q_des1[:, i])))
    print("RMSE1=", RMSE1)

    RMSE2 = []
    for i in range(2):
        RMSE2.append(np.sqrt(np.sum((q_des2[:, i] - q_act2[:, i]) ** 2) / len(q_des2[:, i])))
    print("RMSE2=", RMSE2)

    # Create plot
    # plt.title('joint angle')
    plt.figure(0)
    ax = plt.subplot2grid((6, 2), (1, 0))
    plt.plot(q_des1[:, 4] * 180. / np.pi, 'b-')
    plt.plot(q_act1[:, 4] * 180. / np.pi, 'r-')
    ax.set_xticklabels([])
    plt.ylabel('q5 ($^\circ$)')

    ax = plt.subplot2grid((6, 2), (2, 0))
    plt.plot(q_des1[:, 5] * 180. / np.pi, 'b-')
    plt.plot(q_act1[:, 5] * 180. / np.pi, 'r-')
    plt.ylabel('q6 ($^\circ$)')
    plt.xlabel('sample number')

    ax = plt.subplot2grid((6, 2), (1, 1))
    plt.plot(q_des2[:, 4] * 180. / np.pi, 'b-')
    plt.plot(q_act2[:, 4] * 180. / np.pi, 'r-')
    ax.set_xticklabels([])

    ax = plt.subplot2grid((6, 2), (2, 1))
    plt.plot(q_des2[:, 5] * 180. / np.pi, 'b-')
    plt.plot(q_act2[:, 5] * 180. / np.pi, 'r-')
    # ax.set_xticklabels([])
    plt.xlabel('sample number')
    plt.show()

def plot_hysteresis(q_des, q_act, joint_index):
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set(xlim=(-90, 90), ylim=(-90, 90))
    if joint_index == 2:    # translation
        plt.plot(q_des[:, joint_index], q_act[:, joint_index], 'b-')
        plt.xlabel('desired (mm)')
        plt.ylabel('actual (mm)')
    else:
        plt.plot(q_des[:, joint_index] * 180. / np.pi, q_act[:, joint_index] * 180. / np.pi, 'b-')
        plt.xlabel('desired ($^\circ$)')
        plt.ylabel('actual ($^\circ$)')
    plt.title('Hysteresis of q',joint_index)
    plt.show()


def save_outlier(trajectory, n_data):
    for i,joints in enumerate(trajectory):
        if joints[3]==joints[4]==joints[5]==0.0:
            print ('faulted data: ', i)
            for j,joint in enumerate([joints[3], joints[4], joints[5]]):
                # remove occluded data in data set
                x = np.arange(i-n_data, i+n_data+1)
                y = trajectory[x,j+3]
                index = np.argwhere(y==0.0)
                x = np.delete(x, index)
                y = np.delete(y, index)

                # linear regression (3rd order)
                A = [[xc**3, xc**2, xc, 1] for xc in x]
                b = y
                c = np.linalg.lstsq(A, b, rcond=None)[0]
                trajectory[i][j+3] = c[0]*i**3 + c[1]*i**2 + c[2]*i + c[3]
    return trajectory

# (exp.1) randomly move all joints
file_path = 'exp1/'
q_act1 = np.load(file_path+'q_act_raw.npy')
q_des1 = np.load(file_path+'q_des_raw.npy')
q_act1 = save_outlier(q_act1, 6)

# (exp.2) randomly move q4,q5,q6 with q1,q2,q3 fixed
file_path = 'exp2/'
q_act2 = np.load(file_path + 'q_act_raw.npy')
q_des2 = np.load(file_path + 'q_des_raw.npy')
q_act2[:,0] = 0.0; q_act2[:,1]=0.0; q_act2[:,2]=0.0
q_des2[:,0] = 0.0; q_des2[:,1]=0.0; q_des2[:,2]=0.0
plot_joint_compare(q_des1, q_act1, q_des2, q_act2)


# (exp.3) move q4 only
file_path = 'exp3/'
q_act = np.load(file_path + 'q_act_raw.npy')
q_des = np.load(file_path + 'q_des_raw.npy')
q_act[:,0] = 0.0; q_act[:,1]=0.0; q_act[:,2]=0.0
q_des[:,0] = 0.0; q_des[:,1]=0.0; q_des[:,2]=0.0
t = range(len(q_des))

ax = plt.subplot(311)
plt.plot(t, q_des[:, 3] * 180. / np.pi, 'b-')
plt.plot(t, q_act[:, 3] * 180. / np.pi, 'r-')
plt.ylabel('q4 ($^\circ$)')
ax.set_xticklabels([])

ax = plt.subplot(312)
plt.plot(q_des[:, 4] * 180. / np.pi, 'b-')
plt.plot(q_act[:, 4] * 180. / np.pi, 'r-')
plt.ylabel('q5 ($^\circ$)')
ax.set_xticklabels([])

plt.subplot(313)
plt.plot(q_des[:, 5] * 180. / np.pi, 'b-')
plt.plot(q_act[:, 5] * 180. / np.pi, 'r-')
plt.ylabel('q6 ($^\circ$)')
plt.xlabel('sample number')
plt.show()

# (exp.4) move q5 only
file_path = 'exp4/'
q_act1 = np.load(file_path + 'q_act_raw.npy')
q_des1 = np.load(file_path + 'q_des_raw.npy')
t = range(len(q_des))

# (exp.5) move q6 only
file_path = 'exp5/'
q_act2 = np.load(file_path + 'q_act_raw.npy')
q_des2 = np.load(file_path + 'q_des_raw.npy')
t = range(len(q_des))

plot_joint_compare2(q_des1, q_act1, q_des2, q_act2)

# (exp.6) move q5 with q4=90(deg)

# (exp.7) move q5 with q4=0(deg)