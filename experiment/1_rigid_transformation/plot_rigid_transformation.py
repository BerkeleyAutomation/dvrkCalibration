import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import FLSpegtransfer.utils.CmnUtil as U

root = '/home/hwangmh/pycharmprojects/FLSpegtransfer/'
plt.style.use('seaborn-whitegrid')
# plt.style.use('bmh')
plt.rc('font', size=12)          # controls default text sizes
plt.rc('axes', titlesize=20)     # fontsize of the axes title
plt.rc('axes', labelsize=13)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels
plt.rc('legend', fontsize=17)    # legend fontsize
plt.rc('figure', titlesize=10)  # fontsize of the figure title


file_path = root + 'experiment/1_rigid_transformation/'
pos_des = np.load(file_path + 'pos_des.npy')  # robot's position
pos_act = np.load(file_path + 'pos_act.npy')  # position measured by camera
T = U.get_rigid_transform(np.array(pos_act), np.array(pos_des))
# T = np.load(file_path + 'Trc.npy')  # rigid transform from cam to rob
R = T[:3, :3]
t = T[:3, 3]
print (R, t)
pos_act = np.array([R.dot(p) + t for p in pos_act])

# RMSE error calc
RMSE = np.sqrt(np.sum((pos_des - pos_act) ** 2)/len(pos_des))
print("RMSE=", RMSE, '(m)')

# plot trajectory of des & act position
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(pos_des[:, 0], pos_des[:, 1], pos_des[:, 2], 'b--')
plt.plot(pos_act[:, 0], pos_act[:, 1], pos_act[:, 2], 'r.-')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
# plt.title('Trajectory of tool position')
plt.legend(['desired', 'actual'])
plt.show()