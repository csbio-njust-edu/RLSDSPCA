import numpy as np
import pandas as pd
import warnings
import os
import time
import operator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from sklearn.preprocessing import normalize
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")


def rbf(dist, t):
    '''
    rbf kernel function
    '''
    return np.exp(-(dist / t))

def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)
    aa = np.sum(np.multiply(x, x), 1)
    ab = x * x.T
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def cal_rbf_dist(data, n_neighbors, t):
    dist = Eu_dis(data)
    n = dist.shape[0]
    # rbf_dist = rbf(dist, t)
    W_L = np.zeros((n, n))
    for i in range(n):
        index_L = np.argsort(dist[i, :])[1: n_neighbors + 1]
        len_index_L = len(index_L)
        for j in range(len_index_L):
            # W_L[i, index_L[j]] = rbf_dist[i, index_L[j]]
            W_L[i, index_L[j]] = 1
    # W_L = np.multiply(W_L, (W_L > W_L.transpose())) + np.multiply(W_L.transpose(), (W_L.transpose() >= W_L))
    W_L = np.maximum(W_L, W_L.transpose())
    return W_L

def cal_laplace(data):
    N = data.shape[0]
    H = np.zeros_like(data)
    for i in range(N):
        H[i, i] = np.sum(data[i])
    L = H - data  # Laplacian
    return L

def RLSDSPCA_Algorithm(xMat,bMat,laplace,alpha,beta,gamma,k,c,n):
    obj1 = 0
    obj2 = 0
    thresh = 1e-50
    A = np.random.rand(c, k)
    V = np.eye(n)
    vMat = np.mat(V)
    E = np.ones((xMat.shape[0],xMat.shape[1]))
    E = np.mat(E)
    C = np.ones((xMat.shape[0],xMat.shape[1]))
    C = np.mat(C)
    laplace = np.mat(laplace)
    miu = 1
    for m in range(0, 10):
        Z = (-(miu/2) * ((E - xMat + C/miu).T * (E - xMat + C/miu))) - (alpha * bMat.T * bMat) + beta * vMat + gamma * laplace
        # cal Q
        Z_eigVals, Z_eigVects = np.linalg.eig(np.mat(Z))
        eigValIndice = np.argsort(Z_eigVals)
        n_eigValIndice = eigValIndice[0:k]
        n_Z_eigVect = Z_eigVects[:, n_eigValIndice]
        Q = np.array(n_Z_eigVect)
        # cal V
        q = np.linalg.norm(Q, ord=2, axis=1)
        qq = 1.0 / (q * 2)
        VV = np.diag(qq)
        vMat = np.mat(VV)
        qMat = np.mat(Q)
        # cal Y
        Y = (xMat - E - C/miu) * qMat
        # cal A
        A = bMat * qMat
        # cal AA
        AA = xMat - Y * qMat.T - C/miu
        # cal E
        for i in range(E.shape[1]):
            E[:,i] = (np.max((1 - 1.0 / (miu * np.linalg.norm(AA[:,i]))),0)) * AA[:,i]
        # cal C
        C = C + miu * (E - xMat + Y * qMat.T)
        # cal miu
        miu = 1.2 * miu

        obj1 = np.linalg.norm(qMat)
        if m > 0:
            diff = obj2 - obj1
            if diff < thresh:
                break
        obj2 = obj1
    return Y


def cal_projections(X_data,B_data,alpha1,beta1,gamma1,k_d):
    nclass = 4
    n = len(X_data)
    # dist = Eu_dis(X_data)
    # max_dist = np.max(dist)
    # W_L = cal_rbf_dist(X_data, n_neighbors=9, t=max_dist)
    W_L = cal_rbf_dist(X_data, n_neighbors=9, t=1)
    R = W_L
    M = cal_laplace(R)
    Y = RLSDSPCA_Algorithm(X_data.transpose(), B_data.transpose(), M, alpha1, beta1, gamma1, k_d, nclass, n)
    return Y


if __name__ == '__main__':
    X_filepath = '..\\data\\X_original_G.csv'
    X_original = pd.read_csv(X_filepath)
    X_original = X_original.values
    sc = MinMaxScaler()
    # fit_transform Parameters
    # ----------
    # X: numpy array of shape[n_samples, n_features]
    X_original = sc.fit_transform(X_original)
    X_original = X_original.transpose()
    Y_filepath = '..\\data\\gnd4class_4_GAI.csv'
    Y_gnd4class4 = pd.read_csv(Y_filepath)
    Y_gnd4class4 = Y_gnd4class4.values.transpose()
    X = np.mat(X_original)
    B = np.mat(Y_gnd4class4)
    nclass = 4
    k_d = 4
    count = 0
    correctlist = []
    correctlist.clear()
    x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=143, random_state=1)
    knc = KNeighborsClassifier(n_neighbors=1)
    # 5 folds
    KF = KFold(n_splits=5)
    for alpha in np.logspace(-10,10,21):
        for beta in np.logspace(-10,10,21):
            # correct = 0
            print('count = ',count)
            count += 1
            per_correct = 0
            for train_index, validation_index in KF.split(x_train):
                x_train_t = x_train[train_index]
                x_train_v = x_train[validation_index]
                y_train_t = y_train[train_index]
                y_train_v = y_train[validation_index]
                Y_train_proj = cal_projections(x_train_t, y_train_t, alpha, beta, 0, k_d)
                Y_train_proj = np.mat(Y_train_proj)

                Y_train_proj = (((Y_train_proj.T * Y_train_proj).I) * (Y_train_proj.T)).T
                Q_train_proj1 = (np.mat(x_train_t) * Y_train_proj)
                Q_test_proj = (np.mat(x_train_v) * Y_train_proj)
                knc.fit(np.real(Q_train_proj1), y_train_t)
                y_predict = knc.predict(np.real(Q_test_proj))
                per_correct += knc.score(np.real(Q_test_proj), y_train_v)
            print('per accuracy：', per_correct / 5)
            correctlist.append(per_correct / 5)
    mean_correct_rate = np.array(correctlist).reshape(21,21)
    mean_correct_rate_PD = pd.DataFrame(mean_correct_rate)
    datapath1 = '..\\plot\\RLSDSPCA_accuracy_a_b_k4.csv'
    mean_correct_rate_PD.to_csv(datapath1, index=False)