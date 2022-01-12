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

if __name__ == '__main__':
    COAD_sample1_filepath = rootPath + "/data/COAD_sample1.csv"
    COAD_sample1 = pd.read_csv(COAD_sample1_filepath)
    COAD_sample1 = COAD_sample1.values

    COAD_sample2_filepath = rootPath + "/data/COAD_sample2.csv"
    COAD_sample2 = pd.read_csv(COAD_sample2_filepath)
    COAD_sample2 = COAD_sample2.values

    COAD_sample3_filepath = rootPath + "/data/COAD_sample3.csv"
    COAD_sample3 = pd.read_csv(COAD_sample3_filepath)
    COAD_sample3 = COAD_sample3.values

    COAD_sample4_filepath = rootPath + "/data/COAD_sample4.csv"
    COAD_sample4 = pd.read_csv(COAD_sample4_filepath)
    COAD_sample4 = COAD_sample4.values

    X_original_G_sample = np.vstack((COAD_sample1, COAD_sample2, COAD_sample3, COAD_sample4))
    X_original = X_original_G_sample
    # X_original_G_sample_PD = pd.DataFrame(X_original_G_sample)
    # datapath1 = rootPath + "/data/COAD_sample.csv"
    # X_original_G_sample_PD.to_csv(datapath1, index=False)

    # X_filepath = rootPath + "/data/COAD_GE.csv"
    # X_filepath = 'D:\\MachineLearning\\Python\\pyCharmProjects\\ML\\csbio\\data\\X_original_G.csv'
    # X_original = pd.read_csv(X_filepath)
    # X_original = X_original.values
    sc = MinMaxScaler()
    # fit_transform Parameters
    # ----------
    # X: numpy array of shape[n_samples, n_features]
    X_original = sc.fit_transform(X_original)
    X_original = X_original.transpose()

    Y_filepath = rootPath + "/data/COADSampleCategory1.csv"
    # Y_filepath = 'D:\\MachineLearning\\Python\\pyCharmProjects\\ML\\csbio\\data\\gnd4class_4_GAI.csv'
    Y_gnd4class4 = pd.read_csv(Y_filepath)
    Y_gnd4class4 = Y_gnd4class4.values.transpose()
    X = np.mat(X_original)
    B = np.mat(Y_gnd4class4)
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
    for k in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        print('k = ', k)
        for per in range(1, 6):
            x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=61, random_state=per)
            Y_train_pro = RLSDSPCA_cal_projections(x_train, y_train, 1e5, 60, 3, k)
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
        print('RLSDSPCA per accuracy：', accuracy / 5)
        print('RLSDSPCA per precision：', precision / 5)
        print('RLSDSPCA per recall：', recall / 5)
        print('RLSDSPCA per f1：', f1 / 5)
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

    # LSDSPCA
    print("======================================LSDSPCA======================================")
    accuracylist.clear()
    precisionlist.clear()
    recalllist.clear()
    f1list.clear()
    for k in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        print('k = ', k)
        for per in range(1, 6):
            x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=61, random_state=per)
            Y_train_pro = LSDSPCA_cal_projections(x_train, y_train, 1e5, 60, 3, k)
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
        print('LSDSPCA per accuracy：', accuracy / 5)
        print('LSDSPCA per precision：', precision / 5)
        print('LSDSPCA per recall：', recall / 5)
        print('LSDSPCA per f1：', f1 / 5)
        accuracylist.append(accuracy / 5)
        precisionlist.append(precision / 5)
        recalllist.append(recall / 5)
        f1list.append(f1 / 5)
        print('---------------------------------------')
    accuracy_mean = np.mean(accuracylist)
    accuracy_var = np.var(accuracylist)
    accuracy_std = np.std(accuracylist)
    print('LSDSPCA accuracy_mean = ', accuracy_mean)
    print('LSDSPCA accuracy_var = ', accuracy_var)
    print('LSDSPCA accuracy_std = ', accuracy_std)

    precision_mean = np.mean(precisionlist)
    precision_var = np.var(precisionlist)
    precision_std = np.std(precisionlist)
    print('LSDSPCA precision_mean = ', precision_mean)
    print('LSDSPCA precision_var = ', precision_var)
    print('LSDSPCA precision_std = ', precision_std)

    recall_mean = np.mean(recalllist)
    recall_var = np.var(recalllist)
    recall_std = np.std(recalllist)
    print('LSDSPCA recall_mean = ', recall_mean)
    print('LSDSPCA recall_var = ', recall_var)
    print('LSDSPCA recall_std = ', recall_std)

    f1_mean = np.mean(f1list)
    f1_var = np.var(f1list)
    f1_std = np.std(f1list)
    print('LSDSPCA f1_mean = ', f1_mean)
    print('LSDSPCA f1_var = ', f1_var)
    print('LSDSPCA f1_std = ', f1_std)


    # SDSPCA
    print("======================================SDSPCA======================================")
    accuracylist.clear()
    precisionlist.clear()
    recalllist.clear()
    f1list.clear()
    for k in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        print('k = ', k)
        for per in range(1, 6):
            x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=61, random_state=per)
            Y_train_pro = SDSPCA_cal_projections(x_train, y_train, 1e5, 60, k)
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
        print('SDSPCA per accuracy：', accuracy / 5)
        print('SDSPCA per precision：', precision / 5)
        print('SDSPCA per recall：', recall / 5)
        print('SDSPCA per f1：', f1 / 5)
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
    for k in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        print('k = ', k)
        for per in range(1, 6):
            x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=61, random_state=per)
            Y_train_pro = RgLPCA_cal_projections(x_train, 3, k)
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
        print('RgLPCA per accuracy：', accuracy / 5)
        print('RgLPCA per precision：', precision / 5)
        print('RgLPCA per recall：', recall / 5)
        print('RgLPCA per f1：', f1 / 5)
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
    for k in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        print('k = ', k)
        for per in range(1, 6):
            x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=61, random_state=per)
            Y_train_pro = gLSPCA_cal_projections(x_train, 60, 3, k)
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
        print('gLSPCA per accuracy：', accuracy / 5)
        print('gLSPCA per precision：', precision / 5)
        print('gLSPCA per recall：', recall / 5)
        print('gLSPCA per f1：', f1 / 5)
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
    for k in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        print('k = ', k)
        for per in range(1, 6):
            x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=61, random_state=per)
            Y_train_pro = gLPCA_cal_projections(x_train, 3, k)
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
        print('gLPCA per accuracy：', accuracy / 5)
        print('gLPCA per precision：', precision / 5)
        print('gLPCA per recall：', recall / 5)
        print('gLPCA per f1：', f1 / 5)
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
    for k in [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]:
        accuracy = 0
        precision = 0
        recall = 0
        f1 = 0
        print('k = ', k)
        for per in range(1, 6):
            x_train, x_test, y_train, y_test = train_test_split(X.T, B.T, test_size=61, random_state=per)
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
        print('PCA per accuracy：', accuracy / 5)
        print('PCA per precision：', precision / 5)
        print('PCA per recall：', recall / 5)
        print('PCA per f1：', f1 / 5)
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