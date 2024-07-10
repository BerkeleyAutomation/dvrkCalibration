import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from dvrk.motion.dvrkArm import dvrkArm


dvrk_arm = dvrkArm('/PSM2')
def plot_position(pos_des):
    # plot trajectory of des & act position
    pos_des = np.array(pos_des)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(pos_des[:, 0], pos_des[:, 1], pos_des[:, 2], 'b.--')
    # plt.plot(pos_act[:, 0], pos_act[:, 1], pos_act[:, 2], 'r.-')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_zlabel('z (m)')
    # plt.title('Trajectory of tool position')
    plt.legend(['desired', 'actual'])
    plt.show()

def plot_joint(q_des):
    q_des = np.array(q_des)
    # Create plot
    # plt.title('joint angle')
    ax = plt.subplot(611)
    plt.plot(q_des[:,0]*180./np.pi, 'b-')
    # plt.plot(q_act[:, 0] * 180. / np.pi, 'r-')
    plt.legend(loc='upper center', bbox_to_anchor=(0.9,2))
    ax.set_xticklabels([])
    plt.ylabel('q1 ($^\circ$)')

    ax = plt.subplot(612)
    plt.plot(q_des[:,1]*180./np.pi, 'b-')
    # plt.plot(q_act[:, 1] * 180. / np.pi, 'r-')
    ax.set_xticklabels([])
    plt.ylabel('q2 ($^\circ$)')

    ax = plt.subplot(613)
    plt.plot(q_des[:, 2], 'b-')
    # plt.plot(q_act[:, 2], 'r-')
    ax.set_xticklabels([])
    plt.ylabel('q3 (m)')

    ax = plt.subplot(614)
    plt.plot(q_des[:, 3]*180./np.pi, 'b-')
    # plt.plot(q_act[:, 3]*180./np.pi, 'r-')
    ax.set_xticklabels([])
    plt.ylabel('q4 ($^\circ$)')

    ax = plt.subplot(615)
    plt.plot(q_des[:, 4]*180./np.pi, 'b-')
    # plt.plot(q_act[:, 4]*180./np.pi, 'r-')
    ax.set_xticklabels([])
    plt.ylabel('q5 ($^\circ$)')

    plt.subplot(616)
    plt.plot(q_des[:, 5]*180./np.pi, 'b-')
    # plt.plot(q_act[:, 5]*180./np.pi, 'r-')
    plt.ylabel('q6 ($^\circ$)')
    plt.xlabel('(sample)')
    plt.show()

def ik_position(pos):  # (m)
    x = pos[0]
    y = pos[1]
    z = pos[2]
    L1 = 0.4318  # Rcc (m)
    L2 = 0.4162  # tool
    # L3 = 0.0091  # pitch ~ yaw (m)
    # L4 = 0.0102  # yaw ~ tip (m)

    # Inverse Kinematics
    q1 = np.arctan2(x, -z)  # (rad)
    q2 = np.arctan2(-y, np.sqrt(x ** 2 + z ** 2))  # (rad)
    q3 = np.sqrt(x ** 2 + y ** 2 + z ** 2) + L1 - L2  # (m)
    return q1, q2, q3

def random_sampling(sample_number,psm_number):

    q_target = []
    pos_target = []
    pos_min = []
    pos_max = []
    if(psm_number == '1'):
        #PSM1 for suturing
        pos_min = [0.0, 0.0, -0.125]
        pos_max = [0.11, 0.08, -0.120]
    elif(psm_number == '2'):
        #PSM2 for suturing
        pos_min = [0.0,0.0,-0.106]
        pos_max = [-0.05, 0.05, -0.105]
    else:
        print("Please select PSM1 or 2")
        exit()
    q4_range = np.array([-80, 80])*np.pi/180.
    q5_range = np.array([-60, 60])*np.pi/180.
    q6_range = np.array([-60, 60])*np.pi/180.
    sample_pos_list = [[0.0,0.0,-0.107],[-0.05,0.0,-0.107],[0.0,0.05,-0.107],[-0.05,0.05,-0.107],[0.0,0.0,-0.105],[-0.05,0.0,-0.105],[0.0,0.05,-0.105],[-0.05,0.05,-0.105]]
    for i in range(sample_number):
        pos_rand = np.random.uniform(pos_min, pos_max)
        pos_target.append(pos_rand)
        q1, q2, q3 = ik_position(pos_rand)
        
        # sample_pos = sample_pos_list[i]
        # q1, q2, q3 = ik_position(pos_rand)
        # current_joints = dvrk_arm.get_current_joint()
        # new_joints = np.array([q1,q2,q3,current_joints[3],current_joints[4],current_joints[5]])
        # dvrk_arm.set_joint(new_joints)
        q4 = np.random.uniform(q4_range[0], q4_range[1])
        q5 = np.random.uniform(q5_range[0], q5_range[1])
        q6 = np.random.uniform(q6_range[0], q6_range[1])
        q_target.append([q1,q2,q3,q4,q5,q6])

    return q_target, pos_target

if __name__ == "__main__":
    psm_number = input("Which PSM are you calibrating for suturing workspace. Pick 1 or 2 ")
    print(psm_number)
    q_target, pos_target = random_sampling(1300,psm_number)
    print(np.shape(q_target))
    print(np.shape(pos_target))
    plot_position(pos_target)
    plot_joint(q_target)
    np.save('prime_psm' + psm_number +'_random_sampled', q_target)
    # I'm not saving to true filepath because I don't want to overwrite anything while I'm still testing
    # But eventually save it to this folder: /home/davinci/dvrkCalibration/experiment/0_trajectory_extraction