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


def SDSPCA_Algorithm(xMat,bMat,alpha,beta,k,c,n):
    obj1 = 0
    obj2 = 0
    thresh = 1e-50
    A = np.random.rand(c, k)
    V = np.eye(n)
    vMat = np.mat(V)
    for m in range(0, 10):
        Z = -(xMat.T * xMat) - (alpha * bMat.T * bMat) + beta * vMat
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
        # cal A
        A = bMat * qMat

        obj1 = np.linalg.norm(qMat)
        if m > 0:
            diff = obj2 - obj1
            if diff < thresh:
                break
        obj2 = obj1
    return Y

def cal_projections(X_data,B_data,k_d):
    nclass = 4
    n = len(X_data)
    Y = SDSPCA_Algorithm(X_data.transpose(), B_data.transpose(), 1e2, 0.5, k_d, nclass, n)
    return Y