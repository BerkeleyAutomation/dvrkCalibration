import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso, LassoLars
from sklearn.preprocessing import StandardScaler, FunctionTransformer
root = '/home/hwangmh/pycharmprojects/FLSpegtransfer/'
# filepath = '/experiment/3_training/pick_place/'
filepath = '/experiment/3_training/random_sampled/'

class LinearModel:
    def __init__(self, H):
        self.inverse = True
        self.H = H
        self.cmdHist = []

        # q_des_1 = np.load(root + filepath + "1/q_des.npy")
        # q_des_2 = np.load(root + filepath + "2/q_des.npy")
        # q_des = np.concatenate((q_des_1, q_des_2))
        #
        # q_act_1 = np.load(root + filepath + "1/q_act.npy")
        # q_act_2 = np.load(root + filepath + "2/q_act.npy")
        # q_act = np.concatenate((q_act_1, q_act_2))

        q_des = np.load(root + filepath + "q_des.npy")[:1800]
        q_act = np.load(root + filepath + "q_act.npy")[:1800]
        print("data length:", len(q_des))
        self.fit(q_des, q_act)

    # X = [ [act0, des1, ... desH],
    #       [act1, des2, ... desH+1],
    #       [act2, des3, ...

    def fit(self, q_des, q_act):
        X = []
        y = []
        for i in range(self.H, len(q_des)):
            x_i = [q_act[i] if self.inverse else q_des[i]]
            for j in range(1, self.H+1):
                x_i.append(q_des[i-j])
            X.append(np.concatenate(x_i))
            y.append(q_des[i] if self.inverse else q_act[i])
        X = np.stack(X, axis=0)
        y = np.stack(y, axis=0)

        self.X_scaler = StandardScaler().fit(X)
        self.y_scaler = StandardScaler().fit(y)
        self.reg = Lasso(alpha=0.00001, normalize=True, max_iter=100000)
        self.reg.fit(self.X_scaler.transform(X), self.y_scaler.transform(y))
        return self

    def _predict(self, X):
        return self.y_scaler.inverse_transform(
            self.reg.predict(self.X_scaler.transform(X)))

    def step(self, qTarget):
        if len(self.cmdHist) < self.H:
            self.cmdHist.insert(0, qTarget)
            return qTarget

        x = np.concatenate([qTarget] + self.cmdHist)
        cmd = self._predict(x.reshape(1, -1)).flatten()
        self.cmdHist.insert(0, cmd)
        self.cmdHist.pop()
        return cmd
