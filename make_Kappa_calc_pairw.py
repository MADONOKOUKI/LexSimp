# -*- coding: utf-8 -*-
from sklearn.svm import SVR
from sklearn import svm
from sklearn import neighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import scipy 
import matplotlib.pyplot as plt
import sys
from sklearn.neural_network import MLPClassifier
from sklearn import linear_model,preprocessing # scikit-learn の Linear Regression を利用します
from collections import OrderedDict
# from sklearn.datasets.samples_generator import make_regression # 回帰用のサンプルデータ
from rank_scorer import getScore,getSystemRankings
from sklearn import linear_model,preprocessing # scikit-learn の Linear Regression を利用します
# from sklearn.datasets.samples_generator import make_regression # 回帰用のサンプルデータ
from make_pred_test_csv import create_fix_y_pair_csv
import os

def predict_Y_by_scv(train_csv,test_csv,l):
  #svc
  # l = [4, 5]
  #neural network
  # l = [4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
  # l = [3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]
  x_train = np.loadtxt(train_csv,dtype=(float),delimiter=',',usecols=l)
  y_train = np.loadtxt(train_csv,dtype=(float),delimiter=',',usecols=17)
  x_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=l)
  # y_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=17)
  clf = GradientBoostingClassifier()
  clf.fit(x_train,y_train)
  # clf = MLPClassifier(solver='sgd', alpha=1e-5,\
  #   hidden_layer_sizes=(5, 2), random_state=1,)
  # clf.fit(x_train,y_train)    
  y_test = clf.predict(x_test)
  # clf = svm.SVC()
  # clf.fit(x_train,y_train)  # 訓練する
  # y_test = clf.predict(x_test)
  # regr = linear_model.LinearRegression()
  # regr.fit(x_train,y_train)  # 訓練する
  # y_test = regr.predict(x_test)
  x_train = np.loadtxt(train_csv,dtype=(str),delimiter=',',usecols=(0,1,2))
  x_test = np.loadtxt(test_csv,dtype=(str),delimiter=',',usecols=(0,1,2))
  # x_test2 = np.loadtxt(test_csv,dtype=(str),delimiter=',',usecols=1)
  text = ''
  cnt = 0
  for i in  range(int(x_test[0][0]),int(x_test[len(x_test)-1][0])+1):
    total_score = []
    l = 0
    # print("Sentence "+str(i)+" rankings: ")
    #tokenize
    # if(cnt > len(x_test)-1):
    #     break
    rank = dict()
    text+="Sentence "+str(str(i + int(x_train[len(x_train)-1][0])+2))+" rankings: " 
    # print("Sentence "+str(str(i + int(x_train[len(x_train)-1][0])+2))+" rankings: ")
    # print(cnt)
    # while(float(x_test[cnt][0]) == float(i)-1):
    while(float(x_test[cnt][0]) == float(i)):
      # print(cnt)
      # print(len(x_test))
      if(x_test[cnt][1] == x_test[cnt][2]):
        cnt += 1
        rank[x_test[cnt][1]] = 0
        break
      if(y_test[cnt] == 1):
        if x_test[cnt][1] not in rank:
          rank[x_test[cnt][1]] = 1
        if x_test[cnt][2] not in rank:
          rank[x_test[cnt][2]] = 0
        else :
          rank[x_test[cnt][1]] += 1
          if x_test[cnt][2] not in rank:
            rank[x_test[cnt][2]] = 0
      elif(y_test[cnt] == -1):
        if x_test[cnt][2] not in rank:
          rank[x_test[cnt][2]] = 1
          if x_test[cnt][1] not in rank:
            rank[x_test[cnt][1]] = 0
        else :
          rank[x_test[cnt][2]] += 1
          if x_test[cnt][1] not in rank:
            rank[x_test[cnt][1]] = 0
      cnt += 1
      if(cnt > len(x_test)-1):
        break
    # total_score = sorted(total_score,key=lambda x: x[1],reverse = True) #　タプルの数字が小さい    
    # print("OK")
    rank = sorted(rank.items(), key=lambda x: x[1],reverse = True)
    # OrderedDict(sorted(rank.items(),reverse = True))
    # for j in range(0,l):
    flag = 0
    # print(rank)
    for j in range(0,len(rank)):
      text += "{" + rank[j][0] + "}" 
      # print(i)
      # if(flag > i): continue
      # if(i < len(rank) - 1):
      #   # print(rank[i][0])
      #   if(rank[i][1] == rank[i+1][1]):
      #     text += "{" 
      #     if(rank[i][1] == rank[i+1][1]):
      #       while(rank[i][1] == rank[i+1][1]):
      #         text += rank[i][0] + ', ' 
      #         i += 1
      #         flag = i
      #         if(i > len(rank)-2): break
      #         # print(rank[i][0])
              
      #       text += rank[i][0] + "}"
      #       if(len(rank)-1 > i): text += " "
      #       if(flag > len(rank)-1): break
      #       i += 1
      #       flag = i
      #     # print(i)
      #   else:
      #     text += "{" + rank[i][0] + "}"
      # else:
      #     text += "{" + rank[i][0] + "}"
      if(len(rank)-1 != j): text += " "
    text+='\n'
    
    # csvWriter.writerow(all_listData)
  # print(text)
  return text
def spareman_value(test_csv,test_fix_csv,l):
      y_test = np.loadtxt(test_csv,dtype=(float),delimiter=',',usecols=17)
      y_predict = np.loadtxt(test_fix_csv,dtype=(float),delimiter=',',usecols=17)
      return scipy.stats.spearmanr(y_test,y_predict)[0]
  

if __name__ == '__main__':
  argvs = sys.argv
  # check_argvs(argvs)
  #xmlとsubstitution読み込み
  # print(argvs)
  train_csv = argvs[1]
  test_csv =  argvs[2]

  x = []; y=[]
  max_spare = 0
  max_kvalue = 0
  l_spareman = []
  l_kvalue = []

  for i in range(3,17):
    l = []
    l.append(i)
    for j in range(i+1,17):
      l.append(j)
      print(l)
      # l = [2,3]
      systemRespFile = open(argvs[3],'w')
      systemRespFile.write("")
      processFile = open("fix_test.csv",'w')
      create_fix_y_pair_csv(train_csv,test_csv,"fix_test.csv",l)
          # spare = spareman(train_csv,test_csv,l)

      spare = spareman_value(test_csv,"fix_test.csv",l)
      print(spare)
      # os.remove("./fix_test.csv") #delete csv file
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
      print(kvalue)
      # os.remove("./output.txt")
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
  plt.title('Pairwise-SVC')
  plt.xlabel('spareman')
  plt.ylabel('K-value')
  plt.show()
  # try:
  print(predict_Y_by_scv(train_csv,test_csv))
  #  rint("Oops!  That was mistakes.  Try again...")