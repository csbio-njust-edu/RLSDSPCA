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
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,fbeta_score
from sklearn.metrics import roc_auc_score,confusion_matrix,normalized_mutual_info_score,matthews_corrcoef
from sklearn.metrics import precision_recall_fscore_support

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
    dist_mat = np.asarray(dist_mat)
    return dist_mat

def cal_rbf_dist(data, n_neighbors, t):
    dist = Eu_dis(data)
    n = dist.shape[0]
    # rbf_dist = rbf(dist, t)
    W_L = np.zeros((n, n))
    for i in range(n):
        index_L = np.argsort(dist[i, :])[1:1 + n_neighbors]
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

def gLSPCA_Algorithm(xMat,laplace,beta,gamma,k,n):
    obj1 = 0
    obj2 = 0
    thresh = 1e-50
    V = np.eye(n)
    vMat = np.mat(V)
    for m in range(0, 10):
        Z = -(xMat.T * xMat) + beta * vMat + gamma * laplace
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
        Y = xMat * qMat
        obj1 = np.linalg.norm(qMat)
        if m > 0:
            diff = obj2 - obj1
            if diff < thresh:
                break
        obj2 = obj1
    return Y

def cal_projections(X_data,B_data,k_d):
    nclass = 4
    n = len(X_data)  # 500
    # dist = Eu_dis(X_data)
    # max_dist = np.max(dist)
    # W_L = cal_rbf_dist(X_data, n_neighbors=9, t=max_dist)
    W_L = cal_rbf_dist(X_data, n_neighbors=9, t=1)
    R = W_L
    M = cal_laplace(R)
    Y = gLSPCA_Algorithm(X_data.transpose(), M, 0.5, 1e2, k_d, n)
    return Y
