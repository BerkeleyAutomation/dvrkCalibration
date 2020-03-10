#!/usr/bin/env python
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from sklearn.decomposition import PCA
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression, Lasso, LassoLars
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, FunctionTransformer
plt.style.use('seaborn-whitegrid')

# Trajectory check
train_file_paths = [
    '../training_dataset/peg_transfer/',
#    '../training_dataset/random/',
    '../training_dataset_brijen/peg_transfer/',
#    '../training_dataset_brijen/random/'
    ]
train_q_des = np.vstack(
    [ np.load(path + 'joint_des.npy') for path in train_file_paths ])
train_q_act = np.vstack(
    [ np.load(path + 'joint_act.npy') for path in train_file_paths ])

# for train_file_path in train_file_paths:
#     train_q_des = np.load(train_file_path + 'joint_des.npy')    # desired joint angles: [q1, ..., q6]
#     train_q_act = np.load(train_file_path + 'joint_act.npy')    # actual joint angles: [q1, ..., q6]

verif_file_path = '../verification_dataset/peg_transfer/'
#verif_file_path = train_file_path
verif_q_des = np.load(verif_file_path + 'joint_des.npy')    # desired joint angles: [q1, ..., q6]
verif_q_act = np.load(verif_file_path + 'joint_act.npy')    # actual joint angles: [q1, ..., q6]

class SampleDataHistory:
    def __init__(self, q0, H):
        self.hist = [q0 for i in range(H)]

    def toInput(self, qTarget):
        return np.concatenate([qTarget] + self.hist);

class SampleDataFormat:
    def __init__(self, H=6, inverse=True):
        self.H = H
        self.inverse = inverse
        
    def stack_training_data(self, q_des, q_act):
        train_X = []
        train_y = []
        for i in range(self.H, len(q_des)):
            x_i = [q_act[i] if self.inverse else q_des[i]]
            for j in range(1, self.H+1):
                x_i.append(q_des[i-j])
            train_X.append(np.concatenate(x_i))
            train_y.append(q_des[i] if self.inverse else q_act[i])

        return (np.stack(train_X, axis=0), np.stack(train_y, axis=0))

    def init_qHist(self, q0):
        return SampleDataHistory(q0, self.H)

    def update_qHist(self, tgt, cmd, qHist):
        qHist.hist.insert(0, cmd)
        qHist.hist.pop()
        return qHist

class InterpolatedDataHistory:
    def __init__(self, q0, H, metric, qStep):
        self.H = H
        self.metric = metric
        self.qStep = qStep
        self.hist = [q0]
        
    def toInput(self, qTarget):
        qHist = [qTarget]
        dPrev = 0.0
        j = len(self.hist) - 2
        while j >= 0 and len(qHist) < self.H:
            dNext = dPrev + np.linalg.norm((self.hist[j+1] - self.hist[j]) / self.qStep, ord=self.metric)
            while len(qHist) < min(self.H,dNext):
                s = (len(qHist) - dPrev) / (dNext - dPrev)
                qHist.append(self.hist[j+1] * (1.0-s) + self.hist[j] * s)
            dPrev = dNext
            j = j - 1
            
        while len(qHist) < self.H:
            qHist.append(self.hist[0])
            
        return np.concatenate(qHist)

class InterpolatedDataFormat:
    def __init__(self, H=10, inverse=True):
        self.H = H
        self.qStepDeg = np.array([5.0, 5.0, 0.005, 5.0, 2.5, 2.5])
        d2r = np.pi/180.0
        self.qStep = self.qStepDeg * np.array([d2r, d2r, 1.0, d2r, d2r, d2r])
        self.metric = np.inf
        self.inverse = inverse

    def stack_training_data(self, q_des, q_act):
        sumOfHistLens = 0
        #pltTitle = 'joint angles H=' + str(H) + ", step=" + np.array2string(qStepDeg, precision=2, separator=',', suppress_small=True)
        X = []
        y = []
        for i in range(0,len(q_des)-1):
            qHist = [q_act[i] if self.inverse else q_des[i]]
            dPrev = 0.0
            j = i-2
            while j >= 0 and len(qHist) < self.H:
                dNext = dPrev + np.linalg.norm((q_des[j+1] - q_des[j]) / self.qStep, ord=self.metric)
                while len(qHist) < min(self.H,dNext):
                    s = (len(qHist) - dPrev) / (dNext - dPrev)
                    # print(i, j+1, s)
                    #print(len(qHist), dPrev, dNext, s, s*dNext + (1-s)*dPrev)
                    qHist.append(q_des[j+1] * (1.0-s) + q_des[j] * s)
                dPrev = dNext
                j = j-1
            
            if len(qHist) == self.H:
                sumOfHistLens += i - j
                X.append(np.concatenate(qHist))
                y.append(q_des[i] if self.inverse else q_act[i])
            
        print('mean number of waypoints input vector: %f' % (sumOfHistLens/len(y)))
        return (np.stack(X, axis=0), np.stack(y, axis=0))

    def init_qHist(self, q0):
        return InterpolatedDataHistory(q0, self.H, self.metric, self.qStep)

    def update_qHist(self, tgt, cmd, qHist):
        qHist.hist.append(cmd)
        return qHist

    
class LassoModel:
    def fit(self, X, y):
        self.X_scaler = StandardScaler().fit(X)
        self.y_scaler = StandardScaler().fit(y)
        self.reg = Lasso(alpha=0.00001, normalize=True, max_iter=100000)
        self.reg.fit(self.X_scaler.transform(X), self.y_scaler.transform(y))
        return self
    
    def predict(self, X):
        return self.y_scaler.inverse_transform(
            self.reg.predict(self.X_scaler.transform(X)))

    def plot(self):
        for j in range(6):
            df = pd.DataFrame(data=np.transpose(self.reg.coef_[:,j::6]))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(df, annot=True, cmap=cmap) #.set_title("H = " + str(H))
            plt.show()

class LinearModel:
    def fit(self, X, y):
        self.reg = LinearRegression()
        self.reg.fit(X, y)
        return self

    def predict(self, X):
        return self.reg.predict(X)

    def plot(self):
        for j in range(6):
            df = pd.DataFrame(data=np.transpose(self.reg.coef_[:,j::6]))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            sns.heatmap(df, annot=True, cmap=cmap) #.set_title("H = " + str(H))
            plt.show()        

class RunInverse:
    def __init__(self, dataFormat, model, q0):
        self.dataFormat = dataFormat
        self.model = model
        self.qHist = self.dataFormat.init_qHist(q0)
        
    def step(self, qTarget):
        x = self.qHist.toInput(qTarget) # np.concatenate([qTarget] + self.qHist)
        cmd = self.model.predict(x.reshape(1, -1)).flatten()
        self.qHist = self.dataFormat.update_qHist(qTarget, cmd, self.qHist)
        return cmd

class RunForward:
    def __init__(self, dataFormat, model, q0):
        self.dataFormat = dataFormat
        self.model = model
        self.qHist = self.dataFormat.init_qHist(q0)
        self.alpha = 0.05
        self.iters = 20

    def step(self, qTarget):
        qCmd = qTarget
        for it in range(self.iters):
            x = self.qHist.toInput(qCmd)
            dq = qTarget - self.model.predict(x.reshape(1, -1)).flatten()
            # print(dq)
            qCmd = qCmd + self.alpha * dq
        self.qHist = self.dataFormat.update_qHist(qTarget, qCmd, self.qHist)
        return qCmd

def plot_trajectories(y, yPred, includeError=False):
    # plt.title('joint angle error')
    print("MSE:", mean_squared_error(y[:,3:], yPred[:,3:]))
    fig, axs = plt.subplots(6, sharex=True)
    fig.suptitle("linear_model.py")
    t = range(len(y))
    for j in range(6):
        #jAct = q_act[:,j]
        #jDes = q_des[:,j]
        #plt.subplot(611 + j)
        if j == 3:
            scale = 1
            yLabel = 'q3 (m)'
        else:
            scale = 180.0/np.pi
            yLabel = 'q%d ($^\circ$)' % (j)
        
        axs[j].plot(t, y[:,j]*scale, 'b-', t, yPred[:,j]*scale, 'r-')
                 
        #axs[j].ylabel(yLabel)
        axs[j].set(ylabel=yLabel)

        if includeError:
            ax2 = axs[j].twinx()
            ax2.plot(t, (y[:,j] - yPred[:,j]) * scale, 'y-')
            miny, maxy = axs[j].get_ylim()
            dy = (maxy - miny)/2.0
            ax2.set_ylim(-dy, dy)
            ax2.set_ylabel("err")

    plt.show()

#dataFormat = SampleDataFormat(H=6, inverse=True)
dataFormat = InterpolatedDataFormat(inverse=True)
train_X, train_y = dataFormat.stack_training_data(
    train_q_des, train_q_act)

#model = LinearModel()
model = LassoModel()
model.fit(train_X, train_y)
model.plot()

yPred = model.predict(train_X)
#plot_trajectories(verif_q_act[H:], yPred)

run = RunInverse(dataFormat, model, verif_q_des[0])
verif_cmds = []
for i in range(1, len(verif_q_act)):
    # print("====================================", i)
    cmd = run.step(verif_q_act[i])
    verif_cmds.append(cmd)

plot_trajectories(verif_q_des[1:], np.stack(verif_cmds, axis=0))
plot_trajectories(verif_q_des[1:], verif_q_act[1:])
