import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from FLSpegtransfer.vision.BallDetection import BallDetection
plt.style.use('seaborn-whitegrid')
# plt.style.use('bmh')
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=13)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=11)    # fontsize of the tick labels
plt.rc('ytick', labelsize=11)    # fontsize of the tick labels
plt.rc('legend', fontsize=17)    # legend fontsize
plt.rc('figure', titlesize=10)  # fontsize of the figure title


# Trajectory check
file_path = 'pick_place/'
# file_path = 'random_sampled/'
# file_path = 'random/filtered_downsampled/'
q_des = np.load(file_path + 'q_des.npy')[:200]    # desired joint angles: [q1, ..., q6]
q_act = np.load(file_path + 'q_act.npy')[:200]    # actual joint angles: [q1, ..., q6]
pos_des = np.load(file_path + 'pos_des.npy')[:200]  # desired position: [x,y,z]
pos_act = np.load(file_path + 'pos_act.npy')[:200]  # actual position: [x,y,z]
quat_des = np.load(file_path + 'quat_des.npy')[:200]  # desired quaternion: [qx,qy,qz,qw]
quat_act = np.load(file_path + 'quat_act.npy')[:200]  # desired quaternion: [qx,qy,qz,qw]
t_stamp = np.load(file_path + 't_stamp.npy')[:200]    # measured time (sec)
print('data length1: ',len(q_des))

# plot position trajectory
RMSE = np.sqrt(np.sum((pos_des - pos_act) ** 2)/len(pos_des))
print("RMSE=", RMSE, '(m)')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(pos_des[:, 0], pos_des[:, 1], pos_des[:, 2], 'b-')
plt.plot(pos_act[:, 0], pos_act[:, 1], pos_act[:, 2], 'r-')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
# plt.legend(['desired', 'actual'])
plt.show()

# plot joint angles
RMSE = []
for i in range(6):
    RMSE.append(np.sqrt(np.sum((q_des[:,i] - q_act[:,i]) ** 2))/len(q_des[:,i]))
print("RMSE=", RMSE)
t = range(len(q_des))
plt.title('joint angle')
plt.subplot(611)
plt.plot(t, q_des[:,0]*180./np.pi, 'b-', t, q_act[:,0]*180./np.pi, 'r-')
plt.ylabel('q1 ($^\circ$)')
plt.subplot(612)
plt.plot(t, q_des[:,1]*180./np.pi, 'b-', t, q_act[:, 1] * 180. / np.pi, 'r-')
plt.ylabel('q2 ($^\circ$)')
plt.subplot(613)
plt.plot(t, q_des[:, 2], 'b-', t, q_act[:, 2], 'r-')
plt.ylabel('q3 (mm)')
plt.subplot(614)
plt.plot(t, q_des[:, 3]*180./np.pi, 'b-', t, q_act[:, 3]*180./np.pi, 'r-')
plt.ylabel('q4 ($^\circ$)')
plt.subplot(615)
plt.plot(t, q_des[:, 4]*180./np.pi, 'b-', t, q_act[:, 4]*180./np.pi, 'r-')
plt.ylabel('q5 ($^\circ$)')
plt.subplot(616)
plt.plot(t, q_des[:, 5]*180./np.pi, 'b-', t, q_act[:, 5]*180./np.pi, 'r-')
plt.ylabel('q6 ($^\circ$)')
plt.xlabel('(step)')
plt.show()
