import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.metrics import mean_squared_error
plt.style.use('seaborn-whitegrid')

# Trajectory check
file_path = './result/sampling_3/'
q_des = np.load(file_path + 'q_des.npy')    # desired joint angles: [q1, ..., q6]
q_des = q_des[35:125]
q_act = np.load(file_path + 'q_act.npy')
q_act = q_act[35:125]
new_q_des = np.load(file_path + 'new_q_des.npy')[10:]    # desired joint angles: [q1, ..., q6]
new_q_act = np.load(file_path + 'new_q_act.npy')[10:]    # actual joint angles: [q1, ..., q6]
t_stamp = np.load(file_path + 't_stamp.npy')    # measured time (sec)
print('data length: ',len(new_q_des))

# MSE
orig_mse = mean_squared_error(q_des[:,3:], q_act[:,3:])
new_mse = mean_squared_error(q_des[:,3:], new_q_act[:,3:])
print("Orig MSE:", orig_mse)
print("New MSE:", new_mse)

# Plot joint angles
t = range(len(new_q_des))
plt.title('joint angle')
plt.subplot(611)
plt.plot(t, q_des[:,0]*180./np.pi, 'b-', t, new_q_act[:,0]*180./np.pi, 'r-', t, q_act[:,0]*180./np.pi)
plt.ylabel('q1 ($^\circ$)')
plt.subplot(612)
plt.plot(t, q_des[:,1]*180./np.pi, 'b-', t, new_q_act[:, 1] * 180. / np.pi, 'r-', t, q_act[:,1]*180./np.pi)
plt.ylabel('q2 ($^\circ$)')
plt.subplot(613)
plt.plot(t, q_des[:, 2], 'b-', t, new_q_act[:, 2], 'r-', t, q_act[:,2])
plt.ylabel('q3 (mm)')
plt.subplot(614)
plt.plot(t, q_des[:, 3]*180./np.pi, 'b-', t, new_q_act[:, 3]*180./np.pi, 'r-', t, q_act[:,3]*180./np.pi)
plt.ylabel('q4 ($^\circ$)')
plt.subplot(615)
plt.plot(t, q_des[:, 4]*180./np.pi, 'b-', t, new_q_act[:, 4]*180./np.pi, 'r-', t, q_act[:,4]*180./np.pi)
plt.ylabel('q5 ($^\circ$)')
plt.subplot(616)
plt.plot(t, q_des[:, 5]*180./np.pi, 'b-', t, new_q_act[:, 5]*180./np.pi, 'r-', t, q_act[:,5]*180./np.pi)
plt.ylabel('q6 ($^\circ$)')
plt.xlabel('(step)')
plt.show()