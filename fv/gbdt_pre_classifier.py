from sklearn import ensemble
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn import decomposition
from sklearn.ensemble import GradientBoostingClassifier
import os
from sklearn.externals import joblib


def getDataSet1(n):
    testName_all = []
    rainfall_all = []
    dataSet_all = []
    dataSet_label = []
    i = 0
    k = 1000 * n
    f = open("E:\\Users\kingdom\TIANCHI\data\\train.txt", 'r')
    for line in f.readlines():
        if (i >= k):
            line = line.split()
            labelSet = line[0].split(",")
            testName = labelSet[0]
            rainfall = labelSet[1]
            line[0] = labelSet[2]
            dataSet = list(map(int, line))
            rainfall = float(rainfall)
            if (rainfall <= 19 and rainfall >= 0):
                testName_all.append(testName)
                rainfall_all.append(rainfall)
                dataSet_all.append(dataSet)
                dataSet_label.append(1)
            if (rainfall > 19 ):
                testName_all.append(testName)
                rainfall_all.append(rainfall)
                dataSet_all.append(dataSet)
                dataSet_label.append(2)
            print(i)
            if (i == k + 999):
                break
        i += 1
    return testName_all, rainfall_all, dataSet_all, dataSet_label


def getDataSet2(n, label_num):
    testName_all = []
    rainfall_all = []
    dataSet_all = []
    dataSet_label = []
    i = 0
    k = 1000 * n
    f = open('E:\\Users\kingdom\TIANCHI\data\\train.txt', 'r')
    for line in f.readlines():
        if (i >= k):
            line = line.split()
            labelSet = line[0].split(",")
            testName = labelSet[0]
            rainfall = labelSet[1]
            line[0] = labelSet[2]
            dataSet = list(map(int, line))
            rainfall = float(rainfall)
            if (rainfall <= 19 and rainfall >= 0 and label_num == 0):
                testName_all.append(testName)
                rainfall_all.append(rainfall)
                dataSet_all.append(dataSet)
                dataSet_label.append(1)
            if (rainfall > 19  and label_num == 1):
                testName_all.append(testName)
                rainfall_all.append(rainfall)
                dataSet_all.append(dataSet)
                dataSet_label.append(2)
            print(i)
            if (i == k + 999):
                break
        i += 1
    return testName_all, rainfall_all, dataSet_all, dataSet_label


def getDataSet3(n):
    testName_all = []
    rainfall_all = []
    dataSet_all = []
    dataSet_label = []
    i = 0
    k = 1000 * n
    f = open('E:\\Users\kingdom\TIANCHI\data\\testA.txt', 'r')
    for line in f.readlines():
        if (i >= k):
            line = line.split()
            labelSet = line[0].split(",")
            testName = labelSet[0]
            rainfall = labelSet[1]
            line[0] = labelSet[2]
            dataSet = list(map(int, line))
            rainfall = float(rainfall)
            testName_all.append(testName)
            rainfall_all.append(rainfall)
            dataSet_all.append(dataSet)
            print(i)
            if (i == k + 999):
                break
        i += 1
    return testName_all, rainfall_all, dataSet_all, dataSet_label


def pre_train():
    dataSet_all = []
    rainfall_all = []
    dataSet_label_all = []
    print("aaaaaaaaaaaaaaaaa")
    os.chdir("E:\\Users\kingdom\TIANCHI\data\\machine_model")
    ipca = joblib.load("rain_ipca_model.m")
    print("bbbbbbbbbbbbbbbb")
    for i in range(10):
        testName, rainfall, dataSet, dataSet_label = getDataSet1(i)
        dataSet = ipca.transform(dataSet)
        dataSet_all.extend(dataSet)
        rainfall_all.extend(rainfall)
        dataSet_label_all.extend(dataSet_label)

    print(len(dataSet_all))
    print(len(dataSet_label_all))

    gbc = GradientBoostingClassifier()
    gbc.fit(dataSet_all, dataSet_label_all)
    joblib.dump(gbc, "rain_pretrain_GBDR_class=2.m")
    print("ddddddddddddddd")


def train():
    dataSet_all = [[], [], [], [], []]
    rainfall_all = [[], [], [], [], []]
    dataSet_label_all = [[], [], [], [], []]
    print("aaaaaaaaaaaaaaaaa")
    os.chdir("E:\\Users\kingdom\TIANCHI\data\\machine_model")
    ipca = joblib.load("rain_ipca_model.m")
    print("bbbbbbbbbbbbbbbb")
    for k in range(2):
        for i in range(10):
            testName, rainfall, dataSet, dataSet_label = getDataSet2(i, k)
            dataSet = ipca.transform(dataSet)
            dataSet_all[k].extend(dataSet)
            rainfall_all[k].extend(rainfall)
            # dataSet_label_all[k].extend(dataSet_label) #此处不需要
            print("cccccccccc")

        print(len((dataSet_all[k])))
        print(len(rainfall_all[k]))
        print(dataSet_all[k])
        print(rainfall_all[k])

        x = dataSet_all[k]
        x = np.array(x)
        print(type(x))
        y = rainfall_all[k]
        y = np.array(y)
        gbdt = ensemble.GradientBoostingRegressor()
        gbdt.fit(x, y)  # 训练数据来学习，不需要返回值
        os.chdir("E:\\Users\kingdom\TIANCHI\data\\machine_model")
        joblib.dump(gbdt, "rain_train_GBDT_" + str(k) + "_class=2.m")
        print("done = ", k)
        print("ddddddddddddddd")


def predict():
    os.chdir("E:\\Users\kingdom\TIANCHI\data\\machine_model")
    ipca = joblib.load("rain_ipca_model.m")
    gbc = joblib.load("rain_pretrain_GBDR_class=2.m")
    gbdt1 = joblib.load("rain_train_GBDT_0_class=2.m")
    gbdt2 = joblib.load("rain_train_GBDT_1_class=2.m")
    result = []
    for i in range(2):
        print("1")
        testName, rainfall, dataSet, dataSet_label = getDataSet3(i)
        print("2")
        dataSet = ipca.transform(dataSet)
        print("2.5")
        predict_label = gbc.predict(dataSet)
        print("3")
        for k in range(1000):
            if (predict_label[k] == 1):
                y = gbdt1.predict(dataSet[k])
            if (predict_label[k] == 2):
                y = gbdt2.predict(dataSet[k])
            result.append(y)

    print("4")
    save_file = "E:\\Users\kingdom\TIANCHI\data\\" + "GBDT_result_2" + ".txt"
    fl = open(save_file, 'w')
    for k in result:
        fl.write(str(k[0]))
        fl.write("\n")
    fl.close()
    print("5")


if __name__ == "__main__":
    pre_train()
    print("pretrain done!")
    train()
    print("train done!")
    predict()
    print("finish!")























