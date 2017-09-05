# -*- coding: utf-8 -*-
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import sys
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model,preprocessing # scikit-learn の Linear Regression を利用します
# from sklearn.datasets.samples_generator import make_regression # 回帰用のサンプルデータ


def predict_Y_by_scv(train_csv,test_csv):

  for i in range(3,17):
    if i == 10:
      continue
    l = []
    l.append(i)
    for j in range(i+1,17):
      if j == 10:
        continue
      l.append(j)
      print(l)
      # [2, 3, 4, 5, 6, 7, 8, 10, 11, 12]　の時に多かった　
      # x_train = np.loadtxt(train_csv,dtype=(float),delimiter=',',usecols=(1,2,3,4,5,6,7,9,10,11,12,13))
      # y_train = np.loadtxt(train_csv,dtype=(float),delimiter=',',usecols=14)
      # x_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=(1,2,3,4,5,6,7,9,10,11,12,13))
      # y_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=14)
      x_train = np.loadtxt(train_csv,dtype=(float),delimiter=',',usecols=l)
      y_train = np.loadtxt(train_csv,dtype=(float),delimiter=',',usecols=17)
      x_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=l)
      y_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=17)
      # x_train = np.loadtxt(train_csv,dtype=(float),delimiter=',',usecols=l)
      # y_train = np.loadtxt(train_csv,dtype=(float),delimiter=',',usecols=14)
      # x_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=l)
      # y_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=14)
      # print("線形回帰")
      # regr = linear_model.LinearRegression()
      # regr.fit(x_train,y_train)  # 訓練する
      # y_predict = regr.predict(x_test)
      # print(scipy.stats.spearmanr(y_predict,y_test))
      # print("ロジスティック回帰")
      # regr = linear_model.LogisticRegression()
      # regr.fit(x_train,y_train)  # 訓練する
      # y_predict = regr.predict(x_test)
      # print(scipy.stats.spearmanr(y_predict,y_test))
      print("サポートベクターマシーン")
      clf = svm.SVC()
      clf.fit(x_train,y_train)  # 訓練する
      y_predict = clf.predict(x_test)
      print(scipy.stats.spearmanr(y_predict,y_test))
      print("Neural Network")
      clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
      clf.fit(x_train,y_train)    
      y_predict = clf.predict(x_test)
      print(scipy.stats.spearmanr(y_predict,y_test))
      # print("線形サポーターベクターマシーン")
      # svc = svm.SVC(kernel='linear')
      # svc.fit(x_train,y_train)  # 訓練する
      # y_predict = svc.predict(x_test)
      # print(scipy.stats.spearmanr(y_predict,y_test))
      # print("ランダムフォレスト")
      # clf = RandomForestRegressor()
      # clf.fit(x_train,y_train)  # 訓練する
      # y_predict = clf.predict(x_test)
      # print(scipy.stats.spearmanr(y_predict,y_test))
      # print("k-近傍方")
      # knn = neighbors.KNeighborsClassifier()
      # knn.fit(x_train,y_train)  # 訓練する
      # y_predict = knn.predict(x_test)
      # print(scipy.stats.spearmanr(y_predict,y_test))

if __name__ == '__main__':
  argvs = sys.argv
  # check_argvs(argvs)
  #xmlとsubstitution読み込み
  # print(argvs)
  train_csv = argvs[1]
  test_csv =  argvs[2]
  # try:
  predict_Y_by_scv(train_csv,test_csv)
  #  rint("Oops!  That was mistakes.  Try again...")