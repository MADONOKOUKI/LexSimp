# -*- coding: utf-8 -*-
def make_ordered_pairs(clist):
  r = []
  for tie_pair in itr.combinations(clist, 2):
    #print list(itr.product(tie_pair[0], tie_pair[1]))
    r += itr.product(tie_pair[0], tie_pair[1])
  return r

def make_all_pairs(clist):
  r = []
  for tie_pair in itr.permutations(clist, 2):
    r += itr.product(tie_pair[0], tie_pair[1])
  return r

def make_equal_pairs(clist):
  r = []
  for tie in clist:
    if len(tie) > 1:
      ties = itr.combinations(tie, 2)
      r += ties
  return r

def pos_neg(clist):
  pos = make_ordered_pairs(clist)
  eq = make_equal_pairs(clist)
  neg = []
  for x in make_all_pairs(clist):
    if not x in pos:
      neg.append(x)
  # if len(make_all_pairs(clist)) == 0 and len(clist) == 1:
  #   one = clist
  return pos, neg, eq
import make_feature_data as sp

# enter target word and candidate word tgt, cand
def w2v_sim(w1, w2): #行列のコサイン距離を計算している
  if w1 in w2v_model and w2 in w2v_model:
    # print(w2v_model[w1])
    return cossim(w2v_model[w1], w2v_model[w2])
  else:
    return 0.0

def w2v_ctx_sim(w1, w2): #200次元の行列
  if w1 in w2v_model and w2 in w2v_model:
    # print(w2v_model.vocab[w1].index)
    # print(w2v_model.syn1neg) # すべての単語のベクトルを管理している
    # print(w2v_model.syn1neg[w2v_model.vocab[w1].index])
    return cossim(w2v_model.syn1neg[w2v_model.vocab[w1].index],
            w2v_model.syn1neg[w2v_model.vocab[w2].index])
  else:
    return 0.0
 
def w2v_snt_sim(tokens, ind, x): #1文字を変えた時にどれくらいの近似値を出すか
  tokens_ = list(tokens)
  tokens_[ind] = x
  l = w2v_seq_sim(tokens, tokens_)
  if l != 'nan':
    return l
  else:
    return 0

def w2v_seq_sim(seq1, seq2):
  # print(w2v_model.vector_size)
  v1 = np.zeros(w2v_model.vector_size) #ベクトルサイズの0行列を作成する
  for x in seq1: v1 += w2v_vector(x)
  v2 = np.zeros(w2v_model.vector_size)
  for x in seq2: v2 += w2v_vector(x)
  return cossim(v1, v2)

def w2v_vector(x):
  # print(x)
  # look for みたいな時にlook_forみたいな感じにしてすくう
  if not x in w2v_model: return np.zeros(w2v_model.vector_size)
  return w2v_model[x]


def ext_sem_feat(tokens, tgt, ind, cand):
  return [w2v_sim(tgt, cand), w2v_ctx_sim(tgt, cand),
      w2v_snt_sim(tokens, ind, cand), \
      synsets_Dice(tgt, cand), synsets_Jaccard(tgt, cand), \
      synsets_Tversky(tgt, cand), synsets_Tversky(cand, tgt)]

#ベクトル同士がどれくらい同じ方向を向いているかを表す指標
def cossim(v1, v2):
  return np.dot(v1, v2) / (np.sqrt(np.dot(v1, v1)) * np.sqrt(np.dot(v2, v2)))


def synsets_Dice(w1, w2):
  s1 = set(wn.synsets(w1))
  s2 = set(wn.synsets(w2))
  # intersection は　s1とs2の行列内の一致している文字の数
  return (2.0 * len(s1.intersection(s2))) / (len(s1) + len(s2))

def synsets_Jaccard(w1, w2):
  s1 = set(wn.synsets(w1))
  s2 = set(wn.synsets(w2))
  return float(len(s1.intersection(s2))) / len(s1.union(s2))

def synsets_Tversky(w1, w2):
  s1 = set(wn.synsets(w1))
  s2 = set(wn.synsets(w2))
  if not s1: return 0.0
  return float(len(s1.intersection(s2)))/(len(s1))

#



def load_resources():
  print('loading ngrams...')
  sp.load_ngrams()
  load_w2v()

import gensim
import numpy as np

def load_w2v():
  global w2v_model
  #
  print("loading w2v....")
  w2v_model = gensim.models.Word2Vec.load('./w2v/ew9.w5_sg.model.300')
  # w2v_model.syn1neg[model.vocab[word].index]


import math

def ext_w_feat(w):
  freq_wn = sum([x.count() for x in wn.lemmas(w)])
  freq_ew = sp.unigram[w]
  # w の言葉の　すべての意味
  synsets = wn.synsets(w)
  n_words = sum([len(syn.lemmas()) for syn in synsets])
  # for syn in synsets:
  #   print(syn.lemmas())
  # 長さ・2種類のfrequency・一つの単語のいろいろな意味、その意味それぞれに対して幾らだけ同じ意味の単語が存在するか
  # print(synsets)
  return [len(w), math.log(freq_wn+1), math.log(freq_ew+1), \
      len(synsets), n_words]


def truncate_tokens(tokens, ind, tgt, wsize):
  tokens_ = tokens[max(0, ind-wsize):min(len(tokens), ind+wsize+1)]
  if ind-wsize < 0:
    ind = ind
  else:
    ind = wsize
  return  ind, tokens_

def get_candidate(ranking):
  r = []
  for x in ranking: r += x
  return r

### Reads contexts.xml ###
from lxml import etree
import codecs

def read_contexts(infile, parser = etree.XMLParser(recover=True)):
  '''
  sentences = read_contexts(ctx_f)
  '''
  root = etree.parse(infile, parser)
  contexts = root.findall('.//context')
  sentences = []
  for i, c in enumerate(contexts):
    pos = c.getparent().getparent().values()[0]
    if c.text:
      left = c.text.split()
    else:
      left = []
    target = c[0].text.split()
    if c[0].tail:
      right = c[0].tail.split()
    else:
      right = []
    target_index = len(left)
    line = [i, target_index, target[0], pos] + left + target + right
    sentences.append(line)
  return sentences

### Reads substitutions.gold-rankings ###
import re
import itertools as itr

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

def tokens_p(tokens, ind, x):
  tokens_ = list(tokens)
  tokens_[ind] = x
  return sp.tokens_p(tokens_)

#引数の長さ確認

def check_argvs(argvs):
  if(len(argvs) < 3):
    print("insufficient files you selected")
    quit()
  if(len(argvs) > 3):
    print("too many files you selected")
    quit()

def ins_list(cand,w_feat,sem_feat,sentence_prob,rank):
  l = [cand]
  r = [rank]
  mid = w_feat + sem_feat + [sentence_prob]
  # print(l + w_feat + sem_feat + sentence_prob + r)
  # # preprocessing the vector
  return  mid + r
  # return l + w_feat + sem_feat + mid + r

def is_nan(list_obj):
  for i,obj in enumerate(list_obj):
    if(obj != obj):
      list_obj[i] = 0

  return list_obj


### Main #####
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import sys
import csv
from sklearn import preprocessing

lemmatizer = nltk.WordNetLemmatizer()

if  __name__ == "__main__":
  load_resources()
  load_w2v()
  # 引数読み込み
  argvs = sys.argv
  # check_argvs(argvs)
  #xmlとsubstitution読み込み
  # print(argvs)
  ctx_f = argvs[1]
  ranking_f =  argvs[2]
  sentences = read_contexts(ctx_f)
  cand_lists = read_gold_rankings(ranking_f)
  # print(sentences)
  verbose = False
  features = []
  golds = []
  all_cand_list = []
  pw_list = []
  file1 = open(argvs[3],'w') 
  file2 = open(argvs[4],'w') 
  csvWriter1 = csv.writer(file1)
  csvWriter2 = csv.writer(file2)
  index = []
  word = []
  index_pair = []
  word_pair_x = []
  word_pair_y = []
  word_pair_res = []
  for i, s in enumerate(sentences):
    #
    if verbose: print('Sentence', i, '---')
    ind, tgt = s[1:3]
    tgt_pos = s[3].split('.')[1]
    tgt_lemma = lemmatizer.lemmatize(tgt, tgt_pos) 
    if verbose: print("Subst target/pos/lemma:", tgt, tgt_pos, tgt_lemma)
    #4行目から文章の内容に入るため
    tokens = [x.lower() for x in s[4:]]
    if verbose: print('Tokens:', tokens)
    wsize=3
    if wsize > 0:
      ind, tokens = truncate_tokens(list(tokens), ind, tgt, wsize)
      # if verbose: print('Shorten tokens:', tokens)
    #
    ranking = cand_lists[i]
    candidates = get_candidate(ranking)
    # print(candidates)
    # verbose = True
    w_feat = {}
    sem_feat = {}
    sentence_prob = {}
    # csvWriter = csv.writer(file)
    one = []
    if(len(candidates) == 1):
      one = candidates
    for rank,cand in enumerate(candidates):
      if verbose: print("cand:", cand)
      if not cand in w_feat:
        w_feat[cand] = ext_w_feat(cand)
        # if verbose: print('w_feat:', w_feat[cand])
      if not (tgt, cand) in sem_feat:
        sem_feat[(tgt, cand)] = ext_sem_feat(tokens, tgt_lemma, ind, cand)
        # if verbose: print('sem_feat:', sem_feat[(tgt, cand)])
      if not (i, ind, cand) in sentence_prob:
        sentence_prob[(i, ind, cand)] = tokens_p(tokens, ind, cand)
        # if verbose: print('sent prob:', sentence_prob[(i, ind, cand)])
      # print(rank)
      index.append(i)
      word.append(cand)
      cand_list = ins_list(cand,w_feat[cand],sem_feat[(tgt, cand)],sentence_prob[(i, ind, cand)],(rank+1)) + [100-rank*10] + [100/(rank+1)]
      # cand_list[np.isnan(cand_list)] = 0.0
      fix_cand = []
      for x in cand_list:
        if x == x:
          fix_cand.append(x)
        else:
          fix_cand.append(0)
      # fix_cand = [x for x in cand_list if x != 'nan']
      # all_cand_list.append(cand_list)
      all_cand_list.append(fix_cand)
    positives, negatives, ties  = pos_neg(ranking)
    # print(positives)
    # print(negatives)
    # print(ties)
    # csvWriter.writerows(all_cand_list)

    for x, y in positives:
      x_feat = w_feat[x]+sem_feat[(tgt, x)]+[sentence_prob[(i, ind, x)]]
      y_feat = w_feat[y]+sem_feat[(tgt, y)]+[sentence_prob[(i, ind, y)]]
      tv = synsets_Tversky(x, y)
      xy_feat = [x - y for (x, y) in zip(x_feat,y_feat)] + [tv]
      # if verbose: print(x, '>', y)
      # if verbose: print(x_feat + y_feat + [tv])
      features.append(xy_feat)
      golds.append(1)
      index_pair.append(i)
      word_pair_x.append(x)
      word_pair_y.append(y)
      word_pair_res.append(1)
      # # preprocessing the vector
      cand_list = xy_feat 
      fix_cand = []
      for x in cand_list:
        if x == x:
          fix_cand.append(x)
        else:
          fix_cand.append(0)
      pw_list.append(fix_cand)
      # pw_list.append(cand_list)
    #
    for x, y in negatives:
      x_feat = w_feat[x]+sem_feat[(tgt, x)]+[sentence_prob[(i, ind, x)]]
      y_feat = w_feat[y]+sem_feat[(tgt, y)]+[sentence_prob[(i, ind, y)]]
      tv = synsets_Tversky(x, y)
      xy_feat = [x - y for (x, y) in zip(x_feat,y_feat)] + [tv] 
      # if verbose: print(x, '<', y)
      # if verbose: print(x_feat + y_feat + [tv])
      features.append(xy_feat)
      golds.append(-1)
      # cand_list = [i] + [x] + [y] + xy_feat + [-1]
      # # preprocessing the vector
      index_pair.append(i)
      word_pair_x.append(x)
      word_pair_y.append(y)
      word_pair_res.append(-1)
      # # preprocessing the vector
      cand_list = xy_feat 
      fix_cand = []
      for x in cand_list:
        if x == x:
          fix_cand.append(x)
        else:
          fix_cand.append(0)
      pw_list.append(fix_cand)
      # pw_list.append(cand_list)
    #
    for x, y in ties:
      x_feat = w_feat[x]+sem_feat[(tgt, x)]+[sentence_prob[(i, ind, x)]]
      y_feat = w_feat[y]+sem_feat[(tgt, y)]+[sentence_prob[(i, ind, y)]]
      tv = synsets_Tversky(x, y)
      xy_feat = [x - y for (x, y) in zip(x_feat,y_feat)] + [tv] 
      # if verbose: print(x, '=', y)
      # if verbose: print(x_feat + y_feat + [tv])
      features.append(xy_feat)
      golds.append(0)
      # cand_list = [i] + [x] + [y] + xy_feat + [0]
      # # preprocessing the vector
      index_pair.append(i)
      word_pair_x.append(x)
      word_pair_y.append(y)
      word_pair_res.append(0)
      # # preprocessing the vector
      cand_list = xy_feat 
      fix_cand = []
      for x in cand_list:
        if x == x:
          fix_cand.append(x)
        else:
          fix_cand.append(0)
      pw_list.append(fix_cand)
      # pw_list.append(cand_list)
    for x in one:
      # x_feat = w_feat[x]+sem_feat[(tgt, x)]+[sentence_prob[(i, ind, x)]]
      # y_feat = w_feat[x]+sem_feat[(tgt, x)]+[sentence_prob[(i, ind, x)]]
      # tv = synsets_Tversky(x, x)
      # xy_feat = [x - x for (x, x) in zip(x_feat,x_feat)] + [tv] 
      # if verbose: print(x, '=', y)
      # if verbose: print(x_feat + y_feat + [tv])
      xy_feat = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
      features.append(xy_feat)
      golds.append(0)
      # cand_list = [i] + [x] + [y] + xy_feat + [0]
      # # preprocessing the vector
      print(x)
      index_pair.append(i)
      word_pair_x.append(x)
      word_pair_y.append(x)
      word_pair_res.append(0)
      # # preprocessing the vector
      cand_list = xy_feat 
      fix_cand = []
      for x in cand_list:
        if x == x:
          fix_cand.append(x)
        else:
          fix_cand.append(0)
      pw_list.append(fix_cand)
    if verbose: print
  
  # # preprocessing the vector
  # all_cand_list = [ preprocessing.scale(all_cand_list,axis=i) for i in range(2,len(all_cand_list[0])+1)]
  # pw_list = [ preprocessing.scale(pw_list,axis=i) for i in range(3,len(all_cand_list[0])+1)]
  min_max_scaler = preprocessing.MinMaxScaler()
  all_cand_list = min_max_scaler.fit_transform(all_cand_list)
  # all_cand_list = preprocessing.scale(all_cand_list)
  pw_list = min_max_scaler.fit_transform(pw_list)
  # pw_list = preprocessing.scale(pw_list)

  all_cand_list = np.array(all_cand_list)
  pw_list = np.array(pw_list)

  index = np.array(index)
  word  = np.array(word)
  index_pair = np.array(index_pair)
  word_pair_x = np.array(word_pair_x)
  word_pair_y = np.array(word_pair_y)
  word_pair_res = np.array(word_pair_res)
  all_cand =[]
  pw = []
  for i in range(0,len(index)):
    all_cand.append(np.concatenate([[index[i]],[word[i]],all_cand_list[i]]).tolist())
    # csvWriter1.writerows(np.concatenate([[index[i]],[word[i]],all_cand_list[i]]))
  for i in range(0,len(index_pair)):
    pw.append(np.concatenate([[index_pair[i]],[word_pair_x[i]],[word_pair_y[i]],pw_list[i],[word_pair_res[i]]]).tolist())
    # csvWriter2.writerows(np.concatenate([[index_pair[i]],[word_pair_x[i]],[word_pair_y[i]],pw_list[i],[word_pair_res[i]]]))
  print(all_cand)
  csvWriter1.writerows(all_cand)
  csvWriter2.writerows(pw)

  features = np.array(features)
  # features[np.isnan(features)] = 0.0 # why NaNs exist?
  golds = np.array(golds)
  #