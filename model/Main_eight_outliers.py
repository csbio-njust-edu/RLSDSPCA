import numpy as np
import pandas as pd
import warnings
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from constructW import constructW
from PCA import PCA_cal_projections
from gLPCA import gLPCA_cal_projections
from gLSPCA import gLSPCA_cal_projections
from RgLPCA import RgLPCA_cal_projections
from SDSPCA import SDSPCA_cal_projections
from LSDSPCA import LSDSPCA_cal_projections
from RLSDSPCA import RLSDSPCA_cal_projections
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
warnings.filterwarnings("ignore")
def trans(yy):
    nn = len(yy)
    yy_array = np.zeros((nn,2))
    for i in range(nn):
        if (yy[i] == 1):
            yy_array[i, 1] = 1
        else:
            yy_array[i, 0] = 1
    return yy_array
if __name__ == '__main__':
    X_filepath = rootPath + "/data/gaussxor_eight_outliers.csv"
    X_original_data = pd.read_csv(X_filepath)
    X_original_data = X_original_data.values
    # print(X_original_data)
    X_original = X_original_data[:, 1:]
    X_original = X_original.transpose()
    Y_gnd4class4 = X_original_data[:, 0]
    # print(X_original.shape)
    # print(Y_gnd4class4)
    Y_gnd4class4_array = trans(Y_gnd4class4)
    Y_gnd4class4_array = Y_gnd4class4_array.transpose()
    X = np.mat(X_original)
    B = np.mat(Y_gnd4class4_array)

    accuracylist = []
    precisionlist = []
    recalllist = []
    f1list = []
    knc = KNeighborsClassifier(n_neighbors=1)

    # RLSDSPCA
    print("======================================RLSDSPCA======================================")
    accuracylist.clear()
    precisionlist.clear()
    recalllist.clear()
    f1list.clear()
    for k in [2]:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        print('k = ', k)
        for per in range(1, 6):
            x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=33, random_state=per)
            Y_train_pro = RLSDSPCA_cal_projections(x_train, y_train, 1e-10, 10000000, 10000, k)
            Y_train_pro = np.mat(Y_train_pro)

            Y_train_pro = (((Y_train_pro.T * Y_train_pro).I) * (Y_train_pro.T)).T
            Q_train_pro1 = (np.mat(x_train) * Y_train_pro)
            Q_test_pro = (np.mat(x_test) * Y_train_pro)

            knc.fit(np.real(Q_train_pro1), y_train)
            y_predict = knc.predict(np.real(Q_test_pro))
            y_predict1 = knc.predict_proba(np.real(Q_test_pro))
            # print(y_predict1)
            accuracy += accuracy_score(y_test, y_predict)
            precision += precision_score(y_test, y_predict, average='macro')
            recall += recall_score(y_test, y_predict, average='macro')
            f1 += f1_score(y_test, y_predict, average='macro')
        accuracylist.append(accuracy / 5)
        precisionlist.append(precision / 5)
        recalllist.append(recall / 5)
        f1list.append(f1 / 5)
        print('---------------------------------------')
    accuracy_mean = np.mean(accuracylist)
    accuracy_var = np.var(accuracylist)
    accuracy_std = np.std(accuracylist)
    print('RLSDSPCA accuracy_mean = ', accuracy_mean)
    print('RLSDSPCA accuracy_var = ', accuracy_var)
    print('RLSDSPCA accuracy_std = ', accuracy_std)

    precision_mean = np.mean(precisionlist)
    precision_var = np.var(precisionlist)
    precision_std = np.std(precisionlist)
    print('RLSDSPCA precision_mean = ', precision_mean)
    print('RLSDSPCA precision_var = ', precision_var)
    print('RLSDSPCA precision_std = ', precision_std)

    recall_mean = np.mean(recalllist)
    recall_var = np.var(recalllist)
    recall_std = np.std(recalllist)
    print('RLSDSPCA recall_mean = ', recall_mean)
    print('RLSDSPCA recall_var = ', recall_var)
    print('RLSDSPCA recall_std = ', recall_std)

    f1_mean = np.mean(f1list)
    f1_var = np.var(f1list)
    f1_std = np.std(f1list)
    print('RLSDSPCA f1_mean = ', f1_mean)
    print('RLSDSPCA f1_var = ', f1_var)
    print('RLSDSPCA f1_std = ', f1_std)


    # SDSPCA
    print("======================================SDSPCA======================================")
    accuracylist.clear()
    precisionlist.clear()
    recalllist.clear()
    f1list.clear()
    for k in [2]:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        print('k = ', k)
        for per in range(1, 6):
            x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=33, random_state=per)
            Y_train_pro = SDSPCA_cal_projections(x_train, y_train, 1e-7, 1e4, k)
            Y_train_pro = np.mat(Y_train_pro)

            Y_train_pro = (((Y_train_pro.T * Y_train_pro).I) * (Y_train_pro.T)).T
            Q_train_pro1 = (np.mat(x_train) * Y_train_pro)
            Q_test_pro = (np.mat(x_test) * Y_train_pro)

            knc.fit(np.real(Q_train_pro1), y_train)
            y_predict = knc.predict(np.real(Q_test_pro))
            y_predict1 = knc.predict_proba(np.real(Q_test_pro))
            # print(y_predict1)
            accuracy += accuracy_score(y_test, y_predict)
            precision += precision_score(y_test, y_predict, average='macro')
            recall += recall_score(y_test, y_predict, average='macro')
            f1 += f1_score(y_test, y_predict, average='macro')
        accuracylist.append(accuracy / 5)
        precisionlist.append(precision / 5)
        recalllist.append(recall / 5)
        f1list.append(f1 / 5)
        print('---------------------------------------')
    accuracy_mean = np.mean(accuracylist)
    accuracy_var = np.var(accuracylist)
    accuracy_std = np.std(accuracylist)
    print('SDSPCA accuracy_mean = ', accuracy_mean)
    print('SDSPCA accuracy_var = ', accuracy_var)
    print('SDSPCA accuracy_std = ', accuracy_std)

    precision_mean = np.mean(precisionlist)
    precision_var = np.var(precisionlist)
    precision_std = np.std(precisionlist)
    print('SDSPCA precision_mean = ', precision_mean)
    print('SDSPCA precision_var = ', precision_var)
    print('SDSPCA precision_std = ', precision_std)

    recall_mean = np.mean(recalllist)
    recall_var = np.var(recalllist)
    recall_std = np.std(recalllist)
    print('SDSPCA recall_mean = ', recall_mean)
    print('SDSPCA recall_var = ', recall_var)
    print('SDSPCA recall_std = ', recall_std)

    f1_mean = np.mean(f1list)
    f1_var = np.var(f1list)
    f1_std = np.std(f1list)
    print('SDSPCA f1_mean = ', f1_mean)
    print('SDSPCA f1_var = ', f1_var)
    print('SDSPCA f1_std = ', f1_std)

    # RgLPCA
    print("======================================RgLPCA======================================")
    accuracylist.clear()
    precisionlist.clear()
    recalllist.clear()
    f1list.clear()
    for k in [2]:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        print('k = ', k)
        for per in range(1, 6):
            x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=33, random_state=per)
            Y_train_pro = RgLPCA_cal_projections(x_train, 1, k)
            Y_train_pro = np.mat(Y_train_pro)

            Y_train_pro = (((Y_train_pro.T * Y_train_pro).I) * (Y_train_pro.T)).T
            Q_train_pro1 = (np.mat(x_train) * Y_train_pro)
            Q_test_pro = (np.mat(x_test) * Y_train_pro)

            knc.fit(np.real(Q_train_pro1), y_train)
            y_predict = knc.predict(np.real(Q_test_pro))
            y_predict1 = knc.predict_proba(np.real(Q_test_pro))
            # print(y_predict1)
            accuracy += accuracy_score(y_test, y_predict)
            precision += precision_score(y_test, y_predict, average='macro')
            recall += recall_score(y_test, y_predict, average='macro')
            f1 += f1_score(y_test, y_predict, average='macro')
        accuracylist.append(accuracy / 5)
        precisionlist.append(precision / 5)
        recalllist.append(recall / 5)
        f1list.append(f1 / 5)
        print('---------------------------------------')
    accuracy_mean = np.mean(accuracylist)
    accuracy_var = np.var(accuracylist)
    accuracy_std = np.std(accuracylist)
    print('RgLPCA accuracy_mean = ', accuracy_mean)
    print('RgLPCA accuracy_var = ', accuracy_var)
    print('RgLPCA accuracy_std = ', accuracy_std)

    precision_mean = np.mean(precisionlist)
    precision_var = np.var(precisionlist)
    precision_std = np.std(precisionlist)
    print('RgLPCA precision_mean = ', precision_mean)
    print('RgLPCA precision_var = ', precision_var)
    print('RgLPCA precision_std = ', precision_std)

    recall_mean = np.mean(recalllist)
    recall_var = np.var(recalllist)
    recall_std = np.std(recalllist)
    print('RgLPCA recall_mean = ', recall_mean)
    print('RgLPCA recall_var = ', recall_var)
    print('RgLPCA recall_std = ', recall_std)

    f1_mean = np.mean(f1list)
    f1_var = np.var(f1list)
    f1_std = np.std(f1list)
    print('RgLPCA f1_mean = ', f1_mean)
    print('RgLPCA f1_var = ', f1_var)
    print('RgLPCA f1_std = ', f1_std)

    # gLSPCA
    print("======================================gLSPCA======================================")
    accuracylist.clear()
    precisionlist.clear()
    recalllist.clear()
    f1list.clear()
    for k in [2]:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        print('k = ', k)
        for per in range(1, 6):
            x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=33, random_state=per)
            Y_train_pro = gLSPCA_cal_projections(x_train, 1e4, 1e-9, k)
            Y_train_pro = np.mat(Y_train_pro)

            Y_train_pro = (((Y_train_pro.T * Y_train_pro).I) * (Y_train_pro.T)).T
            Q_train_pro1 = (np.mat(x_train) * Y_train_pro)
            Q_test_pro = (np.mat(x_test) * Y_train_pro)

            knc.fit(np.real(Q_train_pro1), y_train)
            y_predict = knc.predict(np.real(Q_test_pro))
            y_predict1 = knc.predict_proba(np.real(Q_test_pro))
            # print(y_predict1)
            accuracy += accuracy_score(y_test, y_predict)
            precision += precision_score(y_test, y_predict, average='macro')
            recall += recall_score(y_test, y_predict, average='macro')
            f1 += f1_score(y_test, y_predict, average='macro')
        accuracylist.append(accuracy / 5)
        precisionlist.append(precision / 5)
        recalllist.append(recall / 5)
        f1list.append(f1 / 5)
        print('---------------------------------------')
    accuracy_mean = np.mean(accuracylist)
    accuracy_var = np.var(accuracylist)
    accuracy_std = np.std(accuracylist)
    print('gLSPCA accuracy_mean = ', accuracy_mean)
    print('gLSPCA accuracy_var = ', accuracy_var)
    print('gLSPCA accuracy_std = ', accuracy_std)

    precision_mean = np.mean(precisionlist)
    precision_var = np.var(precisionlist)
    precision_std = np.std(precisionlist)
    print('gLSPCA precision_mean = ', precision_mean)
    print('gLSPCA precision_var = ', precision_var)
    print('gLSPCA precision_std = ', precision_std)

    recall_mean = np.mean(recalllist)
    recall_var = np.var(recalllist)
    recall_std = np.std(recalllist)
    print('gLSPCA recall_mean = ', recall_mean)
    print('gLSPCA recall_var = ', recall_var)
    print('gLSPCA recall_std = ', recall_std)

    f1_mean = np.mean(f1list)
    f1_var = np.var(f1list)
    f1_std = np.std(f1list)
    print('gLSPCA f1_mean = ', f1_mean)
    print('gLSPCA f1_var = ', f1_var)
    print('gLSPCA f1_std = ', f1_std)

    # gLPCA
    print("======================================gLPCA======================================")
    accuracylist.clear()
    precisionlist.clear()
    recalllist.clear()
    f1list.clear()
    for k in [2]:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        print('k = ', k)
        for per in range(1, 6):
            x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=33, random_state=per)
            Y_train_pro = gLPCA_cal_projections(x_train, 0.1, k)
            Y_train_pro = np.mat(Y_train_pro)

            Y_train_pro = (((Y_train_pro.T * Y_train_pro).I) * (Y_train_pro.T)).T
            Q_train_pro1 = (np.mat(x_train) * Y_train_pro)
            Q_test_pro = (np.mat(x_test) * Y_train_pro)

            knc.fit(np.real(Q_train_pro1), y_train)
            y_predict = knc.predict(np.real(Q_test_pro))
            y_predict1 = knc.predict_proba(np.real(Q_test_pro))
            # print(y_predict1)
            accuracy += accuracy_score(y_test, y_predict)
            precision += precision_score(y_test, y_predict, average='macro')
            recall += recall_score(y_test, y_predict, average='macro')
            f1 += f1_score(y_test, y_predict, average='macro')
        accuracylist.append(accuracy / 5)
        precisionlist.append(precision / 5)
        recalllist.append(recall / 5)
        f1list.append(f1 / 5)
        print('---------------------------------------')
    accuracy_mean = np.mean(accuracylist)
    accuracy_var = np.var(accuracylist)
    accuracy_std = np.std(accuracylist)
    print('gLPCA accuracy_mean = ', accuracy_mean)
    print('gLPCA accuracy_var = ', accuracy_var)
    print('gLPCA accuracy_std = ', accuracy_std)

    precision_mean = np.mean(precisionlist)
    precision_var = np.var(precisionlist)
    precision_std = np.std(precisionlist)
    print('gLPCA precision_mean = ', precision_mean)
    print('gLPCA precision_var = ', precision_var)
    print('gLPCA precision_std = ', precision_std)

    recall_mean = np.mean(recalllist)
    recall_var = np.var(recalllist)
    recall_std = np.std(recalllist)
    print('gLPCA recall_mean = ', recall_mean)
    print('gLPCA recall_var = ', recall_var)
    print('gLPCA recall_std = ', recall_std)

    f1_mean = np.mean(f1list)
    f1_var = np.var(f1list)
    f1_std = np.std(f1list)
    print('gLPCA f1_mean = ', f1_mean)
    print('gLPCA f1_var = ', f1_var)
    print('gLPCA f1_std = ', f1_std)

    # PCA
    print("======================================PCA======================================")
    accuracylist.clear()
    precisionlist.clear()
    recalllist.clear()
    f1list.clear()
    for k in [2]:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        print('k = ', k)
        for per in range(1, 6):
            x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=33, random_state=per)
            Y_train_pro = PCA_cal_projections(x_train, k)
            Y_train_pro = np.mat(Y_train_pro)

            Y_train_pro = (((Y_train_pro.T * Y_train_pro).I) * (Y_train_pro.T)).T
            Q_train_pro1 = (np.mat(x_train) * Y_train_pro)
            Q_test_pro = (np.mat(x_test) * Y_train_pro)

            knc.fit(np.real(Q_train_pro1), y_train)
            y_predict = knc.predict(np.real(Q_test_pro))
            y_predict1 = knc.predict_proba(np.real(Q_test_pro))
            # print(y_predict1)
            accuracy += accuracy_score(y_test, y_predict)
            precision += precision_score(y_test, y_predict, average='macro')
            recall += recall_score(y_test, y_predict, average='macro')
            f1 += f1_score(y_test, y_predict, average='macro')
        accuracylist.append(accuracy / 5)
        precisionlist.append(precision / 5)
        recalllist.append(recall / 5)
        f1list.append(f1 / 5)
        print('---------------------------------------')
    accuracy_mean = np.mean(accuracylist)
    accuracy_var = np.var(accuracylist)
    accuracy_std = np.std(accuracylist)
    print('PCA accuracy_mean = ', accuracy_mean)
    print('PCA accuracy_var = ', accuracy_var)
    print('PCA accuracy_std = ', accuracy_std)

    precision_mean = np.mean(precisionlist)
    precision_var = np.var(precisionlist)
    precision_std = np.std(precisionlist)
    print('PCA precision_mean = ', precision_mean)
    print('PCA precision_var = ', precision_var)
    print('PCA precision_std = ', precision_std)

    recall_mean = np.mean(recalllist)
    recall_var = np.var(recalllist)
    recall_std = np.std(recalllist)
    print('PCA recall_mean = ', recall_mean)
    print('PCA recall_var = ', recall_var)
    print('PCA recall_std = ', recall_std)

    f1_mean = np.mean(f1list)
    f1_var = np.var(f1list)
    f1_std = np.std(f1list)
    print('PCA f1_mean = ', f1_mean)
    print('PCA f1_var = ', f1_var)
    print('PCA f1_std = ', f1_std)
