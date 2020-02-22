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

def create_waveform(interp, amp1, amp2, amp3, amp4, freq1, freq2, freq3, freq4, phase, step):
    t = np.arange(0, 1, 1.0 / step)
    waveform1 = amp1*np.sin(2*np.pi*freq1*(t-phase))
    waveform2 = amp2*np.sin(2*np.pi*freq2*(t-phase))
    waveform3 = amp3*np.sin(2*np.pi*freq3*(t-phase))
    waveform4 = amp4*np.sin(2*np.pi*freq4*(t-phase))
    waveform = waveform1 + waveform2 + waveform3 + waveform4
    x = waveform / max(waveform)
    y = (interp[1]-interp[0])/2.0*x + (interp[1]+interp[0])/2.0
    return t, y

def plot_arbitrary_waveform():
    f = 6   # (Hz)
    A = 1   # amplitude
    t, waveform = create_waveform(interp=[0.1, 0.5], amp1=A, amp2=A*3, amp3=A*4, freq1=f, freq2=f*1.8, freq3=f*1.4, phase=0.0, step=200)
    t, waveform = create_waveform(interp=[0.1, 0.5], amp1=A, amp2=A*1.2, amp3=A*4.2, freq1=0.8*f, freq2=f*1.9, freq3=f*1.2, phase=0.5, step=200)
    t, waveform = create_waveform(interp=[0.1, 0.5], amp1=A, amp2=A*1.5, amp3=A*3.5, freq1=f, freq2=f*1.8, freq3=f*1.3, phase=0.3, step=200)

    # f = 6  # (Hz)
    # A = 1  # amplitude
    # step, joint[:,3] = self.create_waveform([self.q4_range[0], self.q4_range[1]], amp1=A, amp2=A*3, amp3=A*4, amp4=A*0.5, freq1=f, freq2=f*1.8, freq3=f*1.4, freq4=f*6, phase=0.0, step=155)
    # step, joint[:,4] = self.create_waveform([self.q5_range[0], self.q5_range[1]], amp1=A, amp2=A*1.2, amp3=A*4.2, amp4=A*0.5, freq1=0.8*f, freq2=f*1.9, freq3=f*1.2, freq4=f*6, phase=0.5, step=155)
    # step, joint[:,5] = self.create_waveform([self.q6_range[0], self.q6_range[1]], amp1=A, amp2=A*1.5, amp3=A*3.5, amp4=A*0.5, freq1=f, freq2=f*1.8, freq3=f*1.3, freq4=f*6, phase=0.3, step=155)

    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set(xlim=(-70, 70), ylim=(-70, 70))

    plt.plot(t, waveform, 'ro-')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.title('arbitrary waveform')
    plt.show()

def plot_joint(t, q_des, q_act):
    RMSE = []
    for i in range(6):
        RMSE.append(np.sqrt(np.mean((q_des[:,i] - q_act[:,i]) ** 2)))
    print("RMSE=", RMSE)

    # Create plot
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
    plt.plot(t, q_des[:, 3]*180./np.pi, 'b-', t, q_act[:, 3]*180./np.pi, 'r.-')
    plt.ylabel('q4 ($^\circ$)')
    plt.subplot(615)
    plt.plot(t, q_des[:, 4]*180./np.pi, 'b-', t, q_act[:, 4]*180./np.pi, 'r.-')
    plt.ylabel('q5 ($^\circ$)')
    plt.subplot(616)
    plt.plot(t, q_des[:, 5]*180./np.pi, 'b-', t, q_act[:, 5]*180./np.pi, 'r.-')
    plt.ylabel('q6 ($^\circ$)')
    plt.xlabel('(step)')
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

def index_outlier(trajectory):
    index = []
    for i,joints in enumerate(trajectory):
        if joints[3]==joints[4]==joints[5]==0.0:
            print ('faulted data: ', i)
            index.append(i)
    return index

from FLSpegtransfer.vision.BallDetection import BallDetection
def plot_error_identification():
    # Trajectory check
    BD = BallDetection()
    file_path = root+'experiment/trajectory/raw/'
    arm_traj = np.load(file_path+'short_traj_random_arm.npy')
    wrist_traj = np.load(file_path+'short_traj_random_wrist.npy')
    arm_traj[:, 2] -= 0.010
    # arm_traj = arm_traj[:6255-637]

    fc = 10
    dt = 0.001
    arm_traj = U.LPF(arm_traj, fc, dt)
    wrist_traj = U.LPF(wrist_traj, fc, dt)
    arm_traj = arm_traj[::5]
    wrist_traj = wrist_traj[::5]
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
    plt.plot(q1[:200]*180./np.pi, 'b.-')
    plt.ylabel('q1 ($^\circ$)')
    plt.subplot(612)
    plt.plot(q2[:200]*180./np.pi, 'b.-')
    plt.ylabel('q2 ($^\circ$)')
    plt.subplot(613)
    plt.plot(q3[:200]*1000, 'b.-')
    plt.ylabel('q3 (mm)')
    plt.subplot(614)
    plt.plot(q4[:200]*180./np.pi, 'b.-')
    plt.ylabel('q4 ($^\circ$)')
    plt.subplot(615)
    plt.plot(q5[:200]*180./np.pi, 'b.-')
    plt.ylabel('q5 ($^\circ$)')
    plt.subplot(616)
    plt.plot(q6[:200]*180./np.pi, 'b.-')
    plt.ylabel('q6 ($^\circ$)')
    plt.xlabel('(step)')
    plt.show()

    random_traj = np.concatenate((arm_traj[:,:3], wrist_traj[:,3:6]), axis=1)
    np.save('training_traj_random', random_traj)
    print('new trajectory saved')

    # (exp.0) get transformation from camera to robot.
    file_path = root+'experiment/exp0/'
    pos_des = np.load(file_path+'pos_des.npy')  # robot's position
    pos_act = np.load(file_path+'pos_act.npy')  # position measured by camera
    T = np.load(file_path+'Trc.npy')  # rigid transform from cam to rob
    R = T[:3, :3]
    t = T[:3, 3]
    pos_act = np.array([R.dot(p) + t for p in pos_act])
    pos_des = pos_des*1000  # (mm)
    pos_act = pos_act*1000  # (mm)

    # RMSE error calc
    RMSE = np.sqrt(np.mean((pos_des - pos_act) ** 2))
    print("RMSE=", RMSE, '(mm)')

    # plot trajectory of des & act position
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(pos_des[:, 0], pos_des[:, 1], pos_des[:, 2], 'b.--')
    plt.plot(pos_act[:, 0], pos_act[:, 1], pos_act[:, 2], 'r.-')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_zlabel('z (mm)')
    # plt.title('Trajectory of tool position')
    plt.legend(['desired', 'actual'])
    plt.show()

    # (exp.1) randomly move all joints
    file_path = root+'experiment/exp1/'
    q_act = np.load(file_path+'q_act.npy')
    q_des = np.load(file_path+'q_des.npy')
    q_act = save_outlier(q_act, 3)

    t = range(len(q_des))
    plot_joint(t, q_des, q_act)

    # (exp.2) randomly move q4,q5,q6 with q1,q2,q3 fixed
    file_path = root + 'experiment/exp2/'
    q_act = np.load(file_path + 'q_act.npy')
    q_des = np.load(file_path + 'q_des.npy')
    t = range(len(q_des))
    plot_joint(t, q_des, q_act)

    # (exp.3) move q4 only
    file_path = root + 'experiment/exp3/'
    q_act = np.load(file_path + 'q_act.npy')
    q_des = np.load(file_path + 'q_des.npy')
    t = range(len(q_des))
    plot_joint(t, q_des, q_act)
    ax = fig.add_subplot(111)
    # ax.set(xlim=(-90, 90), ylim=(-90, 90))
    plt.plot(q_des[:,3]*180./np.pi, q_act[:,3]*180./np.pi, 'b-')
    plt.xlabel('desired ($^\circ$)')
    plt.ylabel('actual ($^\circ$)')
    plt.title('Hysteresis of q4')
    plt.show()

    # (exp.4) move q5 only
    file_path = root + 'experiment/exp4/'
    q_act = np.load(file_path + 'q_act.npy')
    q_des = np.load(file_path + 'q_des.npy')
    t = range(len(q_des))
    plot_joint(t, q_des, q_act)
    ax = fig.add_subplot(111)
    # ax.set(xlim=(-90, 90), ylim=(-90, 90))
    plt.plot(q_des[:,4]*180./np.pi, q_act[:,4]*180./np.pi, 'b-')
    plt.xlabel('desired ($^\circ$)')
    plt.ylabel('actual ($^\circ$)')
    plt.title('Hysteresis of q5')
    plt.show()

    # (exp.5) move q6 only
    file_path = root + 'experiment/exp5/'
    q_act = np.load(file_path + 'q_act.npy')
    q_des = np.load(file_path + 'q_des.npy')
    t = range(len(q_des))
    plot_joint(t, q_des, q_act)
    ax = fig.add_subplot(111)
    # ax.set(xlim=(-90, 90), ylim=(-90, 90))
    plt.plot(q_des[:, 5] * 180. / np.pi, q_act[:, 5] * 180. / np.pi, 'b-')
    plt.xlabel('desired ($^\circ$)')
    plt.ylabel('actual ($^\circ$)')
    plt.title('Hysteresis of q6')
    plt.show()

    # (exp.6) move q5 with randomly moving q4

    # (exp.7) move q5 with randomly moving q6

    # (exp.8) move q6 with randomly moving q4

    # (exp.9) move q6 with randomly moving q5

    # (exp.10) move q4 with randomly moving q5

    # (exp.11) move q4 with randomly moving q6

def plot_training_data_set():
    # Trajectory check
    BD = BallDetection()
    file_path = root+'experiment/trajectory/raw/'
    # arm_traj = np.load(file_path+'training_traj_random_arm.npy')
    # wrist_traj = np.load(file_path+'training_traj_random_wrist.npy')
    arm_traj = np.load(file_path + 'training_traj_peg_transfer.npy')
    wrist_traj = np.load(file_path + 'training_traj_peg_transfer.npy')

    arm_traj[:, 2] -= 0.010
    # arm_traj = arm_traj[:6255-637]

    fc = 50
    dt = 0.001
    arm_traj = U.LPF(arm_traj, fc, dt)
    wrist_traj = U.LPF(wrist_traj, fc, dt)
    arm_traj = arm_traj[::3]
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
    plt.plot(q1[:200]*180./np.pi, 'b.-')
    plt.ylabel('q1 ($^\circ$)')
    plt.subplot(612)
    plt.plot(q2[:200]*180./np.pi, 'b.-')
    plt.ylabel('q2 ($^\circ$)')
    plt.subplot(613)
    plt.plot(q3[:200]*1000, 'b.-')
    plt.ylabel('q3 (mm)')
    plt.subplot(614)
    plt.plot(q4[:200]*180./np.pi, 'b.-')
    plt.ylabel('q4 ($^\circ$)')
    plt.subplot(615)
    plt.plot(q5[:200]*180./np.pi, 'b.-')
    plt.ylabel('q5 ($^\circ$)')
    plt.subplot(616)
    plt.plot(q6[:200]*180./np.pi, 'b.-')
    plt.ylabel('q6 ($^\circ$)')
    plt.xlabel('(step)')
    plt.show()

    random_traj = np.concatenate((arm_traj[:,:3], wrist_traj[:,3:6]), axis=1)
    np.save('training_traj_peg_transfer', random_traj)
    print('new trajectory saved')

    # check the collected data
    file_path = root+'experiment/training_data/peg_transfer/raw/'
    q_act = np.load(file_path+'q_act.npy')
    q_des = np.load(file_path+'q_des.npy')
    t_stamp = np.load(file_path+'t_stamp.npy')
    print(q_des.shape, q_act.shape, t_stamp.shape)

    index = index_outlier(q_act)
    q_act = np.delete(q_act, index, axis=0)
    q_des = np.delete(q_des, index, axis=0)
    t_stamp = np.delete(t_stamp, index, axis=0)
    print(q_des.shape, q_act.shape, t_stamp.shape)

    t = range(len(q_des))
    plot_joint(t, q_des, q_act)
    np.save('q_act', q_act)
    np.save('q_des', q_des)
    np.save('t_stamp', t_stamp)
    print("new file saved")

if __name__ == "__main__":
    # plot_position_3D()
    # plot_hysteresis()
    # q_act = np.load('../vision/q_act.npy')
    # q_des = np.load('../vision/q_des.npy')
    # t = range(len(q_des))
    # plot_joint(t, q_des, q_act)
    # plot()
    # plot_error_identification()
    plot_training_data_set()