# -*- coding: utf-8 -*-
import itertools
from sklearn.svm import SVR
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import sys
from sklearn import linear_model,preprocessing # scikit-learn の Linear Regression を利用します
# from sklearn.datasets.samples_generator import make_regression # 回帰用のサンプルデータ

from sklearn import svm, linear_model, cross_validation


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


def predict_Y_by_scv(train_csv,test_csv):

  for i in range(2,15):
    if i == 9:
      continue
    l = []
    l.append(i)
    for j in range(i+1,15):
      if j == 9:
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
      rank_svm = RankSVM().fit(x_train, y_train)
      print('Performance of ranking ', rank_svm.score(x_test,y_test))
      print("線形回帰")
      regr = linear_model.LinearRegression()
      regr.fit(x_train,y_train)  # 訓練する
      y_predict = regr.predict(x_test)
      print(scipy.stats.spearmanr(y_predict,y_test))
      # print("ロジスティック回帰")
      # regr = linear_model.LogisticRegression()
      # regr.fit(x_train,y_train)  # 訓練する
      # y_predict = regr.predict(x_test)
      # print(scipy.stats.spearmanr(y_predict,y_test))
      print("サポートベクターマシーン")
      svc = SVR()
      svc.fit(x_train,y_train)  # 訓練する
      y_predict = svc.predict(x_test)
      print(scipy.stats.spearmanr(y_predict,y_test))
      # print("線形サポーターベクターマシーン")
      # svc = svm.SVC(kernel='linear')
      # svc.fit(x_train,y_train)  # 訓練する
      # y_predict = svc.predict(x_test)
      # print(scipy.stats.spearmanr(y_predict,y_test))
      print("ランダムフォレスト")
      clf = RandomForestRegressor()
      clf.fit(x_train,y_train)  # 訓練する
      y_predict = clf.predict(x_test)
      print(scipy.stats.spearmanr(y_predict,y_test))
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