# -*- coding: utf-8 -*-
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingClassifier
import itertools
from sklearn.svm import SVR
from sklearn import neighbors
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import sys
import csv 
from sklearn.neural_network import MLPClassifier 
from sklearn import linear_model,preprocessing # scikit-learn の Linear Regression を利用します
# from sklearn.datasets.samples_generator import make_regression # 回帰用のサンプルデータ
####
# yの結果を組合せたものをcsvに落とし込むファイル
####

def create_fix_y_csv(train_csv,test_csv,name,l):
  #LR 
  # l = [2, 3]
  # l = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12]
  #svm
  # l =  [2, 3]
  #randomforest[2, 3, 4]
  x_train = np.loadtxt(train_csv,dtype=(float),delimiter=',',usecols=l)
  y_train = np.loadtxt(train_csv,dtype=(float),delimiter=',',usecols=17)
  x_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=l)
  y_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=17)
  # regr = linear_model.LinearRegression()
  # regr.fit(x_train,y_train)  # 訓練する
  # y_test = regr.predict(x_test)
  model = MLPRegressor()
  # model = RandomForestRegressor()
  model.fit(x_train,y_train)  # 訓練する
  y_test = model.predict(x_test)
  file = open(name,'w')
  csvWriter = csv.writer(file)

  l = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
  x_test = np.loadtxt(test_csv,dtype=(str),delimiter=',',usecols=l)
  y_test = list(y_test)
  y_test.reverse()
  for i,line in enumerate(x_test):
    p = y_test.pop()
    line = list(line) + [p] 
    csvWriter.writerow(line)

  # csvWriter.writerows(f)

def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    return np.asarray(X_new), np.asarray(y_new).ravel()


class RankSVM(svm.LinearSVC):
    """Performs pairwise ranking with an underlying LinearSVC model
    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.
    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """

    def fit(self, X, y):
        """
        Fit a pairwise ranking model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)
        Returns
        -------
        self
        """
        X_trans, y_trans = transform_pairwise(X, y)
        super(RankSVM, self).fit(X_trans, y_trans)
        return self

    def predict(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            np.argsort(np.dot(X, self.coef_.T))
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X, y):
        """
        Because we transformed into a pairwise problem, chance level is at 0.5
        """
        X_trans, y_trans = transform_pairwise(X, y)
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)


def create_fix_y_pair_csv(train_csv,test_csv,name,l):
  #LR 
  # l = [2, 3]
  # l = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12]
  #svm
  # l =  [2, 3]
  #randomforest[2, 3, 4]
  x_train = np.loadtxt(train_csv,dtype=(float),delimiter=',',usecols=l)
  y_train = np.loadtxt(train_csv,dtype=(float),delimiter=',',usecols=17)
  x_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=l)
  y_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=17)
  # regr = linear_model.LinearRegression()
  # regr.fit(x_train,y_train)  # 訓練する
  # y_test = regr.predict(x_test)

  # # print("サポートベクターマシーン")
  # clf = svm.SVC()
  # clf.fit(x_train,y_train)  # 訓練する
  # y_predict = clf.predict(x_test)

  # rank_svm = RankSVM().fit(x_train, y_train)
  # y_predict = rank_svm.predict(x_test)
  # print(scipy.stats.spearmanr(y_predict,y_test))
  # print("Neural Network")
  clf = GradientBoostingClassifier()
  clf.fit(x_train,y_train)
  # clf = MLPClassifier(solver='sgd' , alpha=1e-5,\
  #   hidden_layer_sizes=(5, 2), random_state=1,)
  # clf.fit(x_train,y_train)    
  y_predict = clf.predict(x_test)
  # print(scipy.stats.spearmanr(y_predict,y_test))
  
  file = open(name,'w')
  csvWriter = csv.writer(file)

  l = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
  x_test = np.loadtxt(test_csv,dtype=(str),delimiter=',',usecols=l)
  y_predict = list(y_predict)
  y_predict.reverse()
  for i,line in enumerate(x_test):
    p = y_predict.pop()
    line = list(line) + [p] 
    csvWriter.writerow(line)

  # y_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=17)

if __name__ == '__main__':
  argvs = sys.argv
  # check_argvs(argvs)
  #xmlとsubstitution読み込み
  # print(argvs)
  train_csv = argvs[1]
  test_csv =  argvs[2]
  name = argvs[3]
  # try:
  predict_Y_by_scv(train_csv,test_csv,name) 
  #  rint("Oops!  That was mistakes.  Try again...")