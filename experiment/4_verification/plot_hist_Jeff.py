#!/usr/bin/env python
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
plt.style.use('seaborn-whitegrid')

# Trajectory check
file_path = '3_training/random/'
q_des = np.load(file_path + 'joint_des.npy')    # desired joint angles: [q1, ..., q6]
q_act = np.load(file_path + 'joint_act.npy')    # actual joint angles: [q1, ..., q6]
pos_des = np.load(file_path + 'position_des.npy')  # desired position: [x,y,z]
pos_act = np.load(file_path + 'position_act.npy')  # actual position: [x,y,z]
quat_des = np.load(file_path + 'quaternion_des.npy')  # desired quaternion: [qx,qy,qz,qw]
quat_act = np.load(file_path + 'quaternion_act.npy')  # desired quaternion: [qx,qy,qz,qw]
t_stamp = np.load(file_path + 'time_stamp.npy')    # measured time (sec)
print('data length: ',len(q_des))

qStepDeg = np.array([5.0, 5.0, 0.001, 0.5, 0.5, 0.5])
#qStepDeg = np.array([5.0, 5.0, 0.001, 5.0, 5.0, 5.0])
Xformat = "fixed_distance"

H=50
# Xformat = "samples"

X = []
y = []

skipped = 0

if Xformat == "fixed_distance":
    metric=np.inf
    d2r = np.pi/180.0
    qStep = qStepDeg * np.array([d2r, d2r, 1.0, d2r, d2r, d2r])

    sumOfHistLens = 0
    pltTitle = 'joint angles H=' + str(H) + ", step=" + np.array2string(qStepDeg, precision=2, separator=',', suppress_small=True)
    for i in range(0,len(q_des)):
        qHist = [q_des[i]]
        dPrev = 0.0
        j = i-1
        while j >= 0 and len(qHist) < H:
            dNext = dPrev + np.linalg.norm((q_des[j+1] - q_des[j]) / qStep, ord=metric)
            while len(qHist) < min(H,dNext):
                s = (len(qHist) - dPrev) / (dNext - dPrev)
                #print(len(qHist), dPrev, dNext, s, s*dNext + (1-s)*dPrev)
                qHist.append(q_des[j+1] * (1.0-s) + q_des[j] * s)
            dPrev = dNext
            j = j-1            

        if len(qHist) != H:
            skipped = i+1
        else:
            sumOfHistLens += i - j
            X.append(np.concatenate(qHist))
            y.append(q_act[i])
    #     for j in range(1,H):
    #         d = np.linalg.norm((qHist[j-1] - qHist[j]) / qStep, ord=metric)
    #         if abs(1.0-d) > 1e-3:
    #             print("%d, %d t=%f" % (i, j, d))
    print('skipped %d, mean=%f' % (skipped, sumOfHistLens/(len(q_des) - skipped)))
elif Xformat == "samples":
    skipped = H
    pltTitle = 'joint angles H=' + str(H) + ' (samples)'
    for i in range(H,len(q_des)):
        qHist = []
        for j in range(H+1):
            qHist.append(q_des[i-j])
        X.append(np.concatenate(qHist))
        y.append(q_act[i])
else:
    sys.exit("bad value for Xformat")

X = np.stack(X, axis=0)
y = np.stack(y, axis=0)

print(np.shape(X))
#print(X)
        
    #print(qDelta , q_des[j] - qHist[-1], (q_des[j] - qHist[-1]) / qStep)

# for seed in range(30,43):
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)

    # reg = LinearRegression().fit(X_train, y_train)
reg = LinearRegression().fit(X, y)

    # y_test_pred = reg.predict(X_test)

    # if False:
    #     for j in range(6):
            #print('q%d score: %f' % (j, r2_score(y_test[:,j], y_test_pred[:,j], multioutput='variance_weighted')))
            # print('%d: q%d score: %f' % (seed, j, mean_squared_error(y_test[:,j], y_test_pred[:,j])))
    # print('%d: q345 score: %f' % (seed, mean_squared_error(y_test[:,3:], y_test_pred[:,3:])))





# new data

# Trajectory check
file_path = '4_verification/random/'
q_des = np.load(file_path + 'joint_des.npy')    # desired joint angles: [q1, ..., q6]
q_act = np.load(file_path + 'joint_act.npy')    # actual joint angles: [q1, ..., q6]
pos_des = np.load(file_path + 'pos_des.npy')  # desired position: [x,y,z]
pos_act = np.load(file_path + 'pos_act.npy')  # actual position: [x,y,z]
quat_des = np.load(file_path + 'q_des.npy')  # desired quaternion: [qx,qy,qz,qw]
quat_act = np.load(file_path + 'q_act.npy')  # desired quaternion: [qx,qy,qz,qw]
t_stamp = np.load(file_path + 'time_stamp.npy')    # measured time (sec)
print('data length: ',len(q_des))

# H = 50
# qStepDeg = np.array([5.0, 5.0, 0.001, 0.5, 0.5, 0.5])
#qStepDeg = np.array([5.0, 5.0, 0.001, 5.0, 5.0, 5.0])
# Xformat = "fixed_distance"

# H=5
# Xformat = "samples"

X = []
y = []

skipped = 0


if Xformat == "fixed_distance":
    metric=np.inf
    d2r = np.pi/180.0
    qStep = qStepDeg * np.array([d2r, d2r, 1.0, d2r, d2r, d2r])

    sumOfHistLens = 0
    pltTitle = 'joint angles H=' + str(H) + ", step=" + np.array2string(qStepDeg, precision=2, separator=',', suppress_small=True)
    for i in range(0,len(q_des)):
        qHist = [q_des[i]]
        dPrev = 0.0
        j = i-1
        while j >= 0 and len(qHist) < H:
            dNext = dPrev + np.linalg.norm((q_des[j+1] - q_des[j]) / qStep, ord=metric)
            while len(qHist) < min(H,dNext):
                s = (len(qHist) - dPrev) / (dNext - dPrev)
                #print(len(qHist), dPrev, dNext, s, s*dNext + (1-s)*dPrev)
                qHist.append(q_des[j+1] * (1.0-s) + q_des[j] * s)
            dPrev = dNext
            j = j-1

        if len(qHist) != H:
            skipped = i+1
        else:
            sumOfHistLens += i - j
            X.append(np.concatenate(qHist))
            y.append(q_act[i])
    #     for j in range(1,H):
    #         d = np.linalg.norm((qHist[j-1] - qHist[j]) / qStep, ord=metric)
    #         if abs(1.0-d) > 1e-3:
    #             print("%d, %d t=%f" % (i, j, d))
    print('skipped %d, mean=%f' % (skipped, sumOfHistLens/(len(q_des) - skipped)))
elif Xformat == "samples":
    skipped = H
    pltTitle = 'joint angles H=' + str(H) + ' (samples)'
    for i in range(H,len(q_des)):
        qHist = []
        for j in range(H+1):
            qHist.append(q_des[i-j])
        X.append(np.concatenate(qHist))
        y.append(q_act[i])
else:
    sys.exit("bad value for Xformat")

X = np.stack(X, axis=0)
y = np.stack(y, axis=0)

yPred = reg.predict(X)



t = range(len(y))

# plt.title('joint angle error')
fig, axs = plt.subplots(6, sharex=True)
fig.suptitle(pltTitle)
for j in range(6):
    jAct = q_act[:,j]
    jDes = q_des[:,j]
    #plt.subplot(611 + j)
    if j == 3:
        scale = 1
        yLabel = 'q3 (m)'
    else:
        scale = 180.0/np.pi
        yLabel = 'q%d ($^\circ$)' % (j)
        
    axs[j].plot(t, jAct[skipped:]*scale, 'b-', t, yPred[:,j]*scale, 'r-')
                 
    #axs[j].ylabel(yLabel)
    axs[j].set(ylabel=yLabel)
    
    ax2 = axs[j].twinx()
    ax2.plot(t, (jAct[skipped:] - yPred[:,j]) * scale, 'y-')
    miny, maxy = axs[j].get_ylim()
    dy = (maxy - miny)/2.0
    ax2.set_ylim(-dy, dy)
    ax2.set_ylabel("err")

#plt.xlabel('(step)')
plt.show()
    
#print(qStep, qStep*2 / qStep)

# des:  0   1        2
# val:    0.8      2.4
#       0    1    2
