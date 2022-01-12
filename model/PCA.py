import numpy as np
import warnings
warnings.filterwarnings("ignore")


def PCA_Algorithm(xMat,k):
    Z = -(xMat.T * xMat)
    # Z = -(xMat.T * xMat) - (alpha * bMat.T * bMat) + beta * vMat  # (643, 643)
    # 计算Q
    Z_eigVals, Z_eigVects = np.linalg.eig(Z)
    # 对特征值从小到大排序
    eigValIndice = np.argsort(Z_eigVals)
    # 最小的k个特征值的下标,
    # k表示降维的个数
    n_eigValIndice = eigValIndice[0:k]
    # 最小的k个特征值对应的特征向量
    n_Z_eigVect = Z_eigVects[:, n_eigValIndice]
    Q = np.array(n_Z_eigVect)  # (643, 3)
    qMat = np.mat(Q)  # (643, 3)
    # 计算Y
    Y = xMat * qMat  # (20502, 3)
    return Y

def PCA_cal_projections(X_data,k_d):
    # alpha [0-1e4]
    # beta [1e-4-1e1]
    # gamma [1e-4-1e1]
    Y = PCA_Algorithm(X_data.T, k_d) #(20502, 643)
    return Y