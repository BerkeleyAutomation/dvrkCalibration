#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
plt.style.use('seaborn-whitegrid')

# Trajectory check
file_path = 'random/'
q_des = np.load(file_path + 'joint_des.npy')    # desired joint angles: [q1, ..., q6]
q_act = np.load(file_path + 'joint_act.npy')    # actual joint angles: [q1, ..., q6]
pos_des = np.load(file_path + 'position_des.npy')  # desired position: [x,y,z]
pos_act = np.load(file_path + 'position_act.npy')  # actual position: [x,y,z]
quat_des = np.load(file_path + 'quaternion_des.npy')  # desired quaternion: [qx,qy,qz,qw]
quat_act = np.load(file_path + 'quaternion_act.npy')  # desired quaternion: [qx,qy,qz,qw]
t_stamp = np.load(file_path + 'time_stamp.npy')    # measured time (sec)
print('data length: ',len(q_des))


# err
err = q_des - q_act
err = np.square(err)
t = range(len(err))

# plt.title('err squared')
# plt.plot(t, err[:,3], 'b-', t, err[:,4], 'r-', t, err[:,5], 'g-')
# plt.show()


# pca = PCA(n_components=6)
# pca.fit(err)
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)

# Ax = b
# 

#prev = None
if True:
    if True:
        delta = q_des[:-1]-q_des[1:]
        err = (q_des - q_act)[1:]
        data = np.hstack((delta[1:-1], err[1:-1]))
    else:
        delta = q_des[:-H]-q_des[H:]
        err = (q_des - q_act)[H:]
        data = np.hstack((delta,err))
        
    d = pd.DataFrame(data=data, columns=(
        "q1", "q2", "q3", "q4", "q5", "q6",
        "err1", "err2", "err3", "err4", "err5", "err6"))
    corr = d.corr()
    # if prev is not None:
    #     tmp = corr
    #     corr = corr - prev
    #     prev = tmp
    # else:
    #     prev = corr
    # mask = np.triu(np.ones_like(corr, dtype=np.bool))
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, annot=True) #.set_title("H = " + str(H))
    #sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, annot=True,
    #            square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()

print(np.linalg.norm((q_des[:1] - q_des[1:]), ord=np.inf, axis=0) * 180./np.pi)
for i in range(6):
    print(np.mean(np.abs((q_des[:1] - q_des[1:])[:,i])*180./np.pi))
    

if False:
    for H in range(0,11):

        err = (q_des - q_act)[1:]
        delta = q_des[:1] - q_des[1:]
        X = delta[H:]
        y = err[H:]
        for i in range(1,H+1):
            #print(H-i, -i)
            X = np.hstack((X, delta[H-i:-i]))


        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        #X = np.hstack((delta[6:], delta[5:-1], delta[4:-2], delta[3:-3], delta[2:-4], delta[1:-5], delta[:-6]))
        reg = LinearRegression().fit(X_train, y_train)

        print("H=" + str(H) + " score: ", r2_score(y_test, reg.predict(X_test), multioutput='variance_weighted'))


        # print("Score: ", reg.score(X_test, y_test));
        #print(reg.coef_);


    yPred = reg.predict(X)

    t = range(len(y))

    if True:
        plt.title('joint angle error')
        for j in range(6):
            jAct = q_act[:,j]
            jDes = q_des[:,j]
            plt.subplot(611 + j)
            scale = 1 if j == 3 else 180.0/np.pi
            plt.plot(t, jAct[H+1:]*180./np.pi, 'b-', t, (jDes[H+1:] - yPred[:,j])*180/np.pi, 'r-')
            plt.ylabel('q' + str(j) + ' ($^\circ$)')

        plt.xlabel('(step)')
        plt.show()
