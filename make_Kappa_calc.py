# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import sys
import os
from rank_scorer import getScore,getSystemRankings
from sklearn import linear_model,preprocessing # scikit-learn の Linear Regression を利用します
# from sklearn.datasets.samples_generator import make_regression # 回帰用のサンプルデータ
from make_pred_test_csv import create_fix_y_csv
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


def spareman(train_csv,test_csv,l):
      x_train = np.loadtxt(train_csv,dtype=(float),delimiter=',',usecols=l)
      y_train = np.loadtxt(train_csv,dtype=(float),delimiter=',',usecols=17)
      x_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=l)
      y_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=17)
      rank_svm = RankSVM().fit(x_train, y_train)
      y_predict = rank_svm.predict(x_test)


      # print('Performance of ranking ', rank_svm.score(x_test,y_test))
      # print("線形回帰")
      # clf = RandomForestRegressor()
      # clf.fit(x_train,y_train)  # 訓練する
      # y_predict = clf.predict(x_test)
      # print(scipy.stats.spearmanr(y_predict,y_test))
      # regr = linear_model.LinearRegression()
      # regr.fit(x_train,y_train)  # 訓練する
      # y_predict = regr.predict(x_test)
      return scipy.stats.spearmanr(y_predict,y_test)[0]
      # # print("ロジスティック回帰")
      # # regr = linear_model.LogisticRegression()
      # # regr.fit(x_train,y_train)  # 訓練する
      # # y_predict = regr.predict(x_test)
      # # print(scipy.stats.spearmanr(y_predict,y_test))
      # print("サポートベクターマシーン")
      # svc = SVR()
      # svc.fit(x_train,y_train)  # 訓練する
      # y_predict = svc.predict(x_test)
      # print(scipy.stats.spearmanr(y_predict,y_test))
      # # print("線形サポーターベクターマシーン")
      # # svc = svm.SVC(kernel='linear')
      # # svc.fit(x_train,y_train)  # 訓練する
      # # y_predict = svc.predict(x_test)
      # # print(scipy.stats.spearmanr(y_predict,y_test))
      # print("ランダムフォレスト")
      
def spareman_value(test_csv,test_fix_csv,l):
      y_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=17)
      y_predict = np.loadtxt(test_fix_csv,dtype=(float),delimiter=',',usecols=17)
      return scipy.stats.spearmanr(y_test,y_predict)[0]
  

    
def predict_Y_by_scv(train_csv,test_csv,l):
  #LR 
  # l = [2, 3]
  # l = [2, 3, 4, 5, 6, 7, 8, 10, 11, 12]
  #svm
  # l =  [2, 3]
  #randomforest[2, 3, 4]
  x_train = np.loadtxt(train_csv,dtype=(float),delimiter=',',usecols=l)
  y_train = np.loadtxt(train_csv,dtype=(float),delimiter=',',usecols=17)
  x_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=l)
  # y_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=17)
  regr = linear_model.LinearRegression()
  regr.fit(x_train,y_train)  # 訓練する
  y_test = regr.predict(x_test)
  # print("ランダムフォレスト")
  # clf = RandomForestRegressor()
  # clf.fit(x_train,y_train)  # 訓練する
  # y_test = clf.predict(x_test)
  # print(scipy.stats.spearmanr(y_predict,y_test))
  x_train = np.loadtxt(train_csv,dtype=(str),delimiter=',',usecols=(0,1))
  x_test = np.loadtxt(test_csv,dtype=(str),delimiter=',',usecols=(0,1))
  # x_test2 = np.loadtxt(test_csv,dtype=(str),delimiter=',',usecols=1)
  text = ''
  cnt = 0
  # print(int(x_test[len(x_test)-1][0])+2)
  for i in range(int(x_test[0][0]),int(x_test[len(x_test)-1][0])+1):
    total_score = []
    l = 0
    # print("Sentence "+str(i)+" rankings: ")
    #tokenize
    # if(cnt > len(x_test)-1):
    #     break
    text+="Sentence "+str(i + int(x_train[len(x_train)-1][0])+2)+" rankings: " 
    # while(float(x_test[cnt][0]) == float(i)-1):
    # print(float(i))
    if(cnt < len(x_test)-1):
      while(float(x_test[cnt][0]) == float(i)):
        total_score.append([x_test[cnt][1],y_test[cnt]])
        cnt += 1
        l += 1
        if(cnt > len(x_test)-1):
          break
    total_score = sorted(total_score,key=lambda x: x[1],reverse = True) #　タプルの数字が小さい順にソート
    for j in range(0,l):
      # print(total_score[j][0])
          # retrieve phrase indicate meaning (not a word or an idiom)
          # method find return -1 if pharase don't include designate character 
          # if(total_score[j][1].find('_',0,len(total_score[j][1])) != -1): continue
      text += "{" + total_score[j][0] + "}"
      if(len(total_score)-1 != j): text += " "
    if(i < int(x_test[len(x_test)-1][0]) - int(x_test[0][0])): text+='\n'

    # csvWriter.writerow(all_listData)
  return text

if __name__ == '__main__':
  argvs = sys.argv
  # check_argvs(argvs)
  #xmlとsubstitution読み込み
  # print(argvs)
  s = set([])
  train_csv = argvs[1]
  test_csv =  argvs[2]
  x = []; y=[]
  max_spare = 0
  max_kvalue = 0
  l_spareman = []
  l_kvalue = []
  for i in range(2,15):
    l = []
    l.append(i)
    for j in range(i+1,15):
      l.append(j)
      print(l)
      # l = [2,3]
      systemRespFile = open(argvs[3],'w')
      systemRespFile.write("")
      processFile = open("fix_test.csv",'w')
      create_fix_y_csv(train_csv,test_csv,"fix_test.csv",l)
          # spare = spareman(train_csv,test_csv,l)

      spare = spareman_value(test_csv,"fix_test.csv",l)
      print(spare)
      os.remove("./fix_test.csv") #delete csv file
      systemRespFile.write(predict_Y_by_scv(train_csv,test_csv,l))
      x.append(spare)
      systemRespFile = open(argvs[3])
      goldFile = open(argvs[4])
      
      if(max_spare < spare): 
        max_spare = spare
        l_spareman = l
    
      #get system rankings and store in structure
      systemRankings = getSystemRankings(systemRespFile)
      goldRankings = getSystemRankings(goldFile)
      # print(systemRankings)
      # print(goldRankings)

      # try:
      kvalue = getScore(systemRankings,goldRankings,0)
      # s.add((spare,score))
      if(max_kvalue < kvalue): 
        max_kvalue = kvalue
        l_kvalue = l
      systemRespFile.close()
      systemRespFile = open(argvs[3], 'r')
      #   file = open("best_result.txt", 'w')
      #   for row in systemRespFile:
      #     file.write(row)
      #   file.close()
      y.append(kvalue)
      os.remove("./output.txt")
      # print('Normalized system score:', score)
      # print("Oops!  That was mistakes.  Try again...")

  # x = []; y=[]
  # for point in s:
  #    x.append(point[0])
  #    y.append(point[1])
  # print(len(x))
  # print(len(y))
  # print(x)
  # print(y)
  # x = sorted(x)
  # print(x[len(x)-1])
  # # y = sorted(y)
  # print(y[len(y)-1])
  print("sparemanが最大の時の配列の要素",l_spareman)
  print("K値が最大の時の配列の要素",l_kvalue)
  print("spareman最大値",max_spare)
  print("K値最大値",max_kvalue)
  plt.scatter(x,y)
  plt.title('corelation')
  plt.xlabel('spareman')
  plt.ylabel('K-value')
  from sklearn.tree import DecisionTreeClassifier             # 決定木用
  clf = DecisionTreeClassifier(max_depth=2, random_state = 0) # インスタンス作成 max_depth:木の深さ
  visualize_tree(clf, X, y)
  plt.show()
# 通常
# sparemanが最大の時の配列の要素 [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# K値が最大の時の配列の要素 [11, 12, 13, 14]
# spareman最大値 0.478370714305
# K値最大値 0.5369993601560604
# 全体正規化
# sparemanが最大の時の配列の要素 [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# K値が最大の時の配列の要素 [11, 12, 13, 14]
# spareman最大値 0.478435965872
# K値最大値 0.5369993601560604