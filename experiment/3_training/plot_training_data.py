import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from FLSpegtransfer.vision.BallDetection import BallDetection
plt.style.use('seaborn-whitegrid')

def index_outlier(trajectory):
    index = []
    for i,joints in enumerate(trajectory):
        if joints[3]==joints[4]==joints[5]==0.0:
            print ('faulted data: ', i)
            index.append(i)
    return index

# Trajectory check
file_path = 'peg_transfer/'
q_des = np.load(file_path + 'joint_des.npy')    # desired joint angles: [q1, ..., q6]
q_act = np.load(file_path + 'joint_act.npy')    # actual joint angles: [q1, ..., q6]
pos_des = np.load(file_path + 'position_des.npy')  # desired position: [x,y,z]
pos_act = np.load(file_path + 'position_act.npy')  # actual position: [x,y,z]
quat_des = np.load(file_path + 'quaternion_des.npy')  # desired quaternion: [qx,qy,qz,qw]
quat_act = np.load(file_path + 'quaternion_act.npy')  # desired quaternion: [qx,qy,qz,qw]
t_stamp = np.load(file_path + 'time_stamp.npy')    # measured time (sec)
print('data length: ',len(q_des))

# plot position trajectory
RMSE = np.sqrt(np.mean((pos_des - pos_act) ** 2))
print("RMSE=", RMSE, '(mm)')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(pos_des[:, 0], pos_des[:, 1], pos_des[:, 2], 'b--')
plt.plot(pos_act[:, 0], pos_act[:, 1], pos_act[:, 2], 'r-')
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_zlabel('z (mm)')
plt.legend(['desired', 'actual'])
plt.show()

print (t_stamp)

# plot joint angles
RMSE = []
for i in range(6):
    RMSE.append(np.sqrt(np.mean((q_des[:,i] - q_act[:,i]) ** 2)))
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
