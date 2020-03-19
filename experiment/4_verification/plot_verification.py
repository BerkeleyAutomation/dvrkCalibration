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


# Trajectory check
file_path = './result/peg_traj_sampled/wo_model/'
# file_path = './result/random_user/sampling_2_ensemble/'
q_des = np.load(file_path + 'new_q_des.npy') # desired joint angles: [q1, ..., q6]
q_act = np.load(file_path + 'new_q_act.npy')

file_path = './result/peg_traj_sampled/RNN_forward_peg_traj/'
new_q_des = np.load(file_path + 'new_q_des.npy')   # desired joint angles: [q1, ..., q6]
new_q_act = np.load(file_path + 'new_q_act.npy')   # actual joint angles: [q1, ..., q6]
t_stamp = np.load(file_path + 't_stamp.npy')    # measured time (sec)
print('data length: ',len(new_q_des))
print(np.shape(q_act))
new_q_act[:,:3] = q_des[:,:3]

pos_des = np.array(convert_to_cartesian(q_des))
pos_act = np.array(convert_to_cartesian(q_act))
new_pos_act = np.array(convert_to_cartesian(new_q_act))

poke_indices = np.argwhere(new_pos_act[:,2] < -0.12)

pos_des = np.squeeze(pos_des[poke_indices])
pos_act = np.squeeze(pos_act[poke_indices])
new_pos_act = np.squeeze(new_pos_act[poke_indices])


thing = np.linalg.norm(pos_des - pos_act, axis=1)
print("orig max:", np.amax(thing))
print("orig min:", np.amin(thing))
print("orig avg:", np.mean(thing))
print("orig med:", np.median(thing))
print("orig std:", np.std(thing))
# print(sum(1 for i in thing if i < .005)/len(thing) * 100)
# print(sum(1 for i in thing if i < .002)/len(thing) * 100)
# print(sum(1 for i in thing if i < .001)/len(thing) * 100)
# print(sum(1 for i in thing if i < .0005)/len(thing) * 100)
percents_1 = []
for i in range(100):
    dist = float(i) / 10000
    percents_1.append(sum(1 for i in thing if i < dist) / len(thing) * 100)
print()
thing = np.linalg.norm(pos_des - new_pos_act, axis=1)
print(" new max:", np.amax(thing))
print(" new min:", np.amin(thing))
print(" new avg:", np.mean(thing))
print(" new med:", np.median(thing))
print(" new std:", np.std(thing))
# print(sum(1 for i in thing if i < .005)/len(thing) * 100)
# print(sum(1 for i in thing if i < .002)/len(thing) * 100)
# print(sum(1 for i in thing if i < .001)/len(thing) * 100)
# print(sum(1 for i in thing if i < .0005)/len(thing) * 100)

percents_2 = []
for i in range(100):
    dist = float(i) / 10000
    percents_2.append(sum(1 for i in thing if i < dist) / len(thing) * 100)

print()

#plt.plot(range(10), percents_1, color='green', marker='o', linestyle='dashed',linewidth=2, markersize=2)
plt.plot(np.arange(0,10,step=.1), percents_1)
plt.plot(np.arange(0,10,step=.1), percents_2)
plt.fill_between(np.arange(0,10,step=.1), percents_1, alpha=.2)
plt.fill_between(np.arange(0,10,step=.1), percents_2, alpha=.2)
plt.xlabel("Dist (mm)")
plt.ylabel("Percent of Samples at Least y Away")
plt.xticks(np.arange(0,10,step=1))
plt.show()


def plot_all(rand=False):
    models = ["wo_model", "RNN_forward_random_traj", "RNN_inverse_random_sampled_traj", "linear_inverse_random_sampled_traj", "RNN_forward_peg_traj", "RNN_inverse_peg_traj", "linear_inverse_peg_traj"]
    model_names = ["Baseline", "$D_{random}$: Forward RNN", "$D_{random}$: Inverse RNN", "$D_{random}$: Inverse Linear", "$D_{pick}$: Forward RNN", "$D_{pick}$: Inverse RNN",  "$D_{pick}$: Inverse Linear"]

    import matplotlib.pylab as pl
    n = len(model_names)
    colors = pl.cm.tab10(np.linspace(0, 1, n))

    file_path = './result/peg_traj_sampled/'
    q_des_orig = np.load(file_path + 'wo_model/new_q_des.npy')  # desired joint angles: [q1, ..., q6]
    pos_des_orig = np.array(convert_to_cartesian(q_des_orig))
    q_act_orig = np.load(file_path + 'wo_model/new_q_act.npy')  # desired joint angles: [q1, ..., q6]
    RMSE = np.sqrt(np.sum((q_des_orig[:,:3] - q_act_orig[:,:3]) ** 2) / len(q_des_orig))
    print(RMSE)

    for model, model_name, color in zip(models, model_names, colors):
        try:
            q_act = np.load(file_path + model + '/new_q_act.npy')  # actual joint angles: [q1, ..., q6]
            q_act[:, :3] = q_des_orig[:, :3]
        except:
            continue

        pos_act = np.array(convert_to_cartesian(q_act))

        poke_indices = np.argwhere(pos_act[:, 2] < -0.12)
        pos_des = np.squeeze(pos_des_orig[poke_indices])
        pos_act = np.squeeze(pos_act[poke_indices])

        distances = np.linalg.norm(pos_des - pos_act, axis=1)
        percents = []
        for i in range(60):
            dist = float(i) / 10000
            percents.append(sum(1 for i in distances if i < dist) / len(distances) * 100)

        if "random" in model_name:
            linestyle = 'dashed'
        elif "pick" in model_name:
            linestyle = 'solid'
        else:
            linestyle = 'dotted'

        plt.plot(np.arange(0, 6, step=.1), percents, label=model_name, alpha=1, linestyle=linestyle, color=color)
        # plt.fill_between(np.arange(0, 5, step=.1), percents, alpha=.5)

        thing = np.linalg.norm(pos_des - pos_act, axis=1)
        print(model_name + " max:", np.amax(thing))
        print(model_name + " min:", np.amin(thing))
        print(model_name + " avg:", np.mean(thing))
        print(model_name + " med:", np.median(thing))
        print(model_name + " std:", np.std(thing))
        print()
        # print(sum(1 for i in thing if i <


    legend = plt.legend(frameon = 1, loc="lower right")
    frame = legend.get_frame()
    frame.set_facecolor('white')
    frame.set_edgecolor('black')
    plt.xlabel("Dist (mm)")
    plt.ylabel("Percent of Samples at Least x Away")
    plt.xticks(np.arange(0, 7, step=1))
    plt.show()

plot_all()

# RMSE error calc
old_RMSE = np.sqrt(np.sum((pos_des - pos_act) ** 2)/len(pos_des))
new_RMSE = np.sqrt(np.sum((pos_des - new_pos_act) ** 2)/len(pos_des))
print("old_RMSE=", old_RMSE, '(m)')
print("new_RMSE=", new_RMSE, '(m)')

# plot trajectory of des & act position
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.plot(pos_des[:, 0], pos_des[:, 1], pos_des[:, 2], 'b.--')
plt.plot(pos_act[:, 0], pos_act[:, 1], pos_act[:, 2], 'g.-')
plt.plot(new_pos_act[:, 0], new_pos_act[:, 1], new_pos_act[:, 2], 'r.-')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('z (m)')
# plt.title('Trajectory of tool position')
plt.legend(['desired', 'actual'])
plt.show()

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

# import mpl_toolkits.mplot3d.axes3d as p3
# import matplotlib.animation as animation
#
# def update_lines(num, dataLines, lines):
#     color = ['b', 'g', 'r']
#     for line, data, c in zip(lines, dataLines, color):
#         # NOTE: there is no .set_data() for 3 dim data...
#         line.set_data(data[0:2, :num])
#         line.set_3d_properties(data[2, :num])
#         # line.set_marker("")
#         line.set_color(c)
#     return lines
#
# # Attaching 3D axis to the figure
# fig = plt.figure()
# ax = p3.Axes3D(fig)
#
# x1, y1, z1 = pos_des[:,0],pos_des[:,1], pos_des[:,2]
# x2, y2, z2 = pos_act[:,0],pos_act[:,1], pos_act[:,2]
# x3, y3, z3 = new_pos_act[:,0],new_pos_act[:,1], new_pos_act[:,2]
# data = np.array([[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]])
#
# # NOTE: Can't pass empty arrays into 3d version of plot()
# lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]
#
# ax.set_xlim(.10,.20)
# ax.set_ylim(-.04,.02)
# ax.set_zlim(-.130,-.100)
# ax.set_xlabel('x (m)')
# ax.set_ylabel('y (m)')
# ax.set_zlabel('z (m)')
# line_ani = animation.FuncAnimation(fig, update_lines, 590, fargs=(data, lines),
#                                    interval=60, blit=True, repeat=True)
#
#
# # line_ani.save('the_movie.mp4', writer = 'imagemagick', fps=30)
# # plt.legend(['desired', 'original act', 'corrected act'])
# #line_ani.save('line_animation_3d_funcanimation.mp4', writer='ffmpeg',fps=1000/100)
# #line_ani.save('line_animation_3d_funcanimation.gif', writer='imagemagick',fps=1000/100)
# #plt.show()