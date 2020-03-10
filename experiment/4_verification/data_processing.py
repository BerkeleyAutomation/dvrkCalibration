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

print (0.008*180./np.pi)

# Trajectory check
file_path = 'random/raw/'
q_des = np.load(file_path + 'q_des_raw.npy')    # desired joint angles: [q1, ..., q6]
q_act = np.load(file_path + 'sq_act_raw.npy')    # actual joint angles: [q1, ..., q6]
t_stamp = np.load(file_path + 't_stamp_raw.npy')    # measured time (sec)
print('data length: ', len(q_des))

print (index_outlier(q_act))

# plot joint angles
RMSE = []
for i in range(6):
    RMSE.append(np.sqrt(np.sum((q_des[:,i] - q_act[:,i]) ** 2)/len(q_act)))
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
