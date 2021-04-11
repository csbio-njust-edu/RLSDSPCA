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


def PCA_Algorithm(xMat,k):
    Z = -(xMat.T * xMat)
    # cal Q
    Z_eigVals, Z_eigVects = np.linalg.eig(Z)
    eigValIndice = np.argsort(Z_eigVals)
    n_eigValIndice = eigValIndice[0:k]
    n_Z_eigVect = Z_eigVects[:, n_eigValIndice]
    Q = np.array(n_Z_eigVect)
    qMat = np.mat(Q)
    # cal Y
    Y = xMat * qMat
    return Y

def cal_projections(X_data,k_d):
    Y= PCA_Algorithm(X_data.T, k_d)
    return Y