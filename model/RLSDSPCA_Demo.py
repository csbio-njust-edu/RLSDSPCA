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

def cal_projections(X_data,B_data,k_d):
    nclass = 4
    n = len(X_data)
    # dist = Eu_dis(X_data)
    # max_dist = np.max(dist)
    # W_L = cal_rbf_dist(X_data, n_neighbors=9, t=max_dist)
    W_L = cal_rbf_dist(X_data, n_neighbors=9, t=1)
    R = W_L
    M = cal_laplace(R)
    Y = RLSDSPCA_Algorithm(X_data.transpose(), B_data.transpose(), M, 1e2, 0.5, 1e2, k_d, nclass, n)
    return Y


if __name__ == '__main__':
    X_filepath = '..\\data\\X_original_G.csv'
    X_original = pd.read_csv(X_filepath,header = None)
    X_original = X_original.values
    sc = MinMaxScaler()
    # fit_transform Parameters
    # ----------
    # X: numpy array of shape[n_samples, n_features]
    X_original = sc.fit_transform(X_original)
    X_original = X_original.transpose()
    Y_filepath = '..\\data\\gnd4class_4_GAI.csv'
    Y_gnd4class4 = pd.read_csv(Y_filepath,header = None)
    Y_gnd4class4 = Y_gnd4class4.values.transpose()
    X = np.mat(X_original)
    B = np.mat(Y_gnd4class4)
    accuracylist = []
    precisionlist = []
    recalllist = []
    f1list = []
    knc = KNeighborsClassifier(n_neighbors=1)
    for k in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        print('k = ', k)
        for per in range(1, 6):
            x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=143, random_state=per)
            Y_train_pro = cal_projections(x_train, y_train, k)
            Y_train_pro = np.mat(Y_train_pro)

            Y_train_pro = (((Y_train_pro.T * Y_train_pro).I) * (Y_train_pro.T)).T
            Q_train_pro1 = (np.mat(x_train) * Y_train_pro)
            Q_test_pro = (np.mat(x_test) * Y_train_pro)  # 100，4
            knc.fit(np.real(Q_train_pro1), y_train)
            y_predict = knc.predict(np.real(Q_test_pro))
            accuracy += accuracy_score(y_test, y_predict)
            precision += precision_score(y_test, y_predict, average='macro')
            recall += recall_score(y_test, y_predict, average='macro')
            f1 += f1_score(y_test, y_predict, average='macro')
        print('per accuracy：', accuracy / 5)
        print('per precision：', precision / 5)
        print('per recall：', recall / 5)
        print('per f1：', f1 / 5)
        accuracylist.append(accuracy / 5)
        precisionlist.append(precision / 5)
        recalllist.append(recall / 5)
        f1list.append(f1 / 5)
        print('---------------------------------------')
    accuracy_mean = np.mean(accuracylist)
    accuracy_var = np.var(accuracylist)
    accuracy_std = np.std(accuracylist)
    print('accuracy_mean = ', accuracy_mean)
    print('accuracy_var = ', accuracy_var)
    print('accuracy_std = ', accuracy_std)
    mean_accuracy_rate_PD = pd.DataFrame(accuracylist)
    datapath1 = '..\\plot\\accuracy_RLSDSPCA.csv'
    mean_accuracy_rate_PD.to_csv(datapath1, index=False)

    precision_mean = np.mean(precisionlist)
    precision_var = np.var(precisionlist)
    precision_std = np.std(precisionlist)
    print('precision_mean = ', precision_mean)
    print('precision_var = ', precision_var)
    print('precision_std = ', precision_std)
    mean_precision_rate_PD = pd.DataFrame(precisionlist)
    datapath2 = '..\\plot\\precision_RLSDSPCA.csv'
    mean_precision_rate_PD.to_csv(datapath2, index=False)

    recall_mean = np.mean(recalllist)
    recall_var = np.var(recalllist)
    recall_std = np.std(recalllist)
    print('recall_mean = ', recall_mean)
    print('recall_var = ', recall_var)
    print('recall_std = ', recall_std)
    mean_recall_rate_PD = pd.DataFrame(recalllist)
    datapath3 = '..\\plot\\recall_RLSDSPCA.csv'
    mean_recall_rate_PD.to_csv(datapath3, index=False)

    f1_mean = np.mean(f1list)
    f1_var = np.var(f1list)
    f1_std = np.std(f1list)
    print('f1_mean = ', f1_mean)
    print('f1_var = ', f1_var)
    print('f1_std = ', f1_std)
    mean_f1_rate_PD = pd.DataFrame(f1list)
    datapath4 = '..\\plot\\f1_RLSDSPCA.csv'
    mean_f1_rate_PD.to_csv(datapath4, index=False)
