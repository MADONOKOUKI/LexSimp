import re

def cand_list(line):
  '''
  In: Sentence 3 rankings: {colourful} {bright} {brilliant} {gleam, luminous}
  Out: [['colourful'], ['bright'], ['brilliant'], ['gleam', 'luminous']]
  '''
  # ^ : 行の先頭にmatchさせている
  # . : 改行以外の任意の文字にmatchする
  # + : 直前のREに作用してREを一回以上繰り返したものにmatchを行う
  # \ : それを正規表現としてでなく文字列として扱っている
  # () : 丸括弧にどんな正規表現があってもmatchする
  # ? : 直線の文字を0または1回だけ繰り返したものを返す
  # $ : 文字列の末尾、文字列の末尾の改行の直線とmatchする

  m = re.match('^Sentence .+ rankings: (\{.+\})', line)
  cands = m.group(1)
  clist = []
  while True:
    m = re.match('\{(.+?)\} (.+)$', cands)
    if not m:
      m = re.match('\{(.+?)\}$', cands)
      tie = m.group(1).split(', ')
      clist.append(tie)
      return clist
    else:
      tie = m.group(1).split(', ')
      clist.append(tie)
      cands = m.group(2)


def read_gold_rankings(infile):
  '''
  cand_lists = read_gold_rankings(ranking_f)
  '''
  f = open(infile, 'r')
  cand_lists = []
  for l in f.readlines():
    c_list = cand_list(l)
    cand_lists.append(c_list)
  return cand_lists

### Main #####
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import sys
import csv
from sklearn import preprocessing

lemmatizer = nltk.WordNetLemmatizer()

if  __name__ == "__main__":
  # 引数読み込み
  argvs = sys.argv
  # check_argvs(argvs)
  #xmlとsubstitution読み込み
  # print(argvs)
  test_f = argvs[1]
  answer_f =  argvs[2]
  test = read_gold_rankings(test_f)
  answer = read_gold_rankings(answer_f)

  ans = 0
  for i in range(0,len(test)):
    if(test[i][0] == answer[i][0]):
      ans += 1

  print(ans/len(test))
