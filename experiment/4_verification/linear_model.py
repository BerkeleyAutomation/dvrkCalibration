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
from cvxopt import spmatrix, matrix, solvers, printing


class LinearModel:
    def __init__(self, q0, H):
        self.inverse = True
        self.H = H
        self.cmdHist = [q0 for i in range(H)]

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
        X = np.stack(train_X, axis=0)
        y = np.stack(train_y, axis=0)

        self.X_scaler = StandardScaler().fit(X)
        self.y_scaler = StandardScaler().fit(y)
        self.reg = Lasso(alpha=0.00001, normalize=True, max_iter=100000)
        self.reg.fit(self.X_scaler.transform(X), self.y_scaler.transform(y))
        return self

    def predict(self, X):
        return self.y_scaler.inverse_transform(
            self.reg.predict(self.X_scaler.transform(X)))

    def step(self, qTarget):
        x = np.concatenate([qTarget] + self.cmdHist)
        cmd = self.model.predict(x.reshape(1, -1)).flatten()
        self.cmdHist.insert(0, cmd)
        self.cmdHist.pop()
        return cmd

    
