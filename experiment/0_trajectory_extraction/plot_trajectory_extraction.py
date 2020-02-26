import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import FLSpegtransfer.utils.CmnUtil as U
from FLSpegtransfer.vision.BallDetection import BallDetection
root = '/home/hwangmh/pycharmprojects/FLSpegtransfer/'
plt.style.use('seaborn-whitegrid')
# plt.style.use('bmh')
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=15)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)    # fontsize of the tick labels
plt.rc('ytick', labelsize=10)    # fontsize of the tick labels
plt.rc('legend', fontsize=15)    # legend fontsize
plt.rc('figure', titlesize=10)  # fontsize of the figure title

# Trajectory check
BD = BallDetection()
file_path = root+'experiment/0_trajectory_extraction/raw/'
arm_traj = np.load(file_path+'training_traj_peg_transfer_raw.npy')
wrist_traj = np.load(file_path+'training_traj_peg_transfer_raw.npy')
arm_traj[:, 2] -= 0.010
# arm_traj = arm_traj[6255-5618:]
# wrist_traj = wrist_traj[1:]
print(arm_traj.shape, wrist_traj.shape)

fc = 10
dt = 0.001
arm_traj = U.LPF(arm_traj, fc, dt)
wrist_traj = U.LPF(wrist_traj, fc, dt)
arm_traj = arm_traj[::3]    # down sampling by 3 for peg transfer and by 2 for random trajectory
wrist_traj = wrist_traj[::3]
print('data length: ',len(arm_traj), len(wrist_traj))
pos = np.array(
    [BD.fk_position(q[0], q[1], q[2], 0, 0, 0, L1=BD.L1, L2=BD.L2, L3=0, L4=0) for q in arm_traj])*1000
q1 = arm_traj[:, 0]
q2 = arm_traj[:, 1]
q3 = arm_traj[:, 2]
q4 = wrist_traj[:, 3]
q5 = wrist_traj[:, 4]
q6 = wrist_traj[:, 5]

# Create plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b.-')
ax.set_xlabel('x (mm)')
ax.set_ylabel('y (mm)')
ax.set_zlabel('z (mm)')
plt.title('Random trajectory of tool position')
plt.show()

# Create plot
plt.title('joint angle')
plt.subplot(611)
plt.plot(q1*180./np.pi, 'b.-')
plt.ylabel('q1 ($^\circ$)')
plt.subplot(612)
plt.plot(q2*180./np.pi, 'b.-')
plt.ylabel('q2 ($^\circ$)')
plt.subplot(613)
plt.plot(q3*1000, 'b.-')
plt.ylabel('q3 (mm)')
plt.subplot(614)
plt.plot(q4*180./np.pi, 'b.-')
plt.ylabel('q4 ($^\circ$)')
plt.subplot(615)
plt.plot(q5*180./np.pi, 'b.-')
plt.ylabel('q5 ($^\circ$)')
plt.subplot(616)
plt.plot(q6*180./np.pi, 'b.-')
plt.ylabel('q6 ($^\circ$)')
plt.xlabel('(step)')
plt.show()

random_traj = np.concatenate((arm_traj[:,:3], wrist_traj[:,3:6]), axis=1)
np.save('new_traj', random_traj)
print('new trajectory saved')