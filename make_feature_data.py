#!/usr/bin/python
# -*- coding: utf-8 -*-
# Last modified: 2017-08-04 10:54:17 hayashi

#
global unigram, bigram, triram

import nltk

# Reading enwik9 file
ew9_f = 0
# ew9_f = open('./enwik/enwik9.txt', 'r')

def make_ngrams(f = ew9_f):
  global unigram, bigram, trigram
  print('Reading and constucting n-grams...')
  ew9_words = f.read().split()
  ew9_bigrams = nltk.bigrams(ew9_words)
  ew9_trigrams = nltk.trigrams(ew9_words)

  print('Making FreqDist for n-grams...')
  print('  unigrams...')
  unigram = nltk.FreqDist([(x) for x in ew9_words])
  print('  bigrams...')
  bigram = nltk.FreqDist(ew9_bigrams)
  print('  trigrams...')
  trigram = nltk.FreqDist(ew9_trigrams)

#
import _pickle as cPickle
#
def save_ngrams(u='./w2v/ew9.uni,pkl', b='./w2v/ew9.bi.pkl', t='./w2v/ew9.tri.pkl'):
  global unigram, bigram, trigram
  data = [unigram, bigram, trigram]
  for i, x in enumerate([u, b, t]):
    f = open(x, 'wb')
    print('Saving...', x)
    cPickle.dump(data[i], f)
    f.close()

#
def load_ngrams():
  global unigram, bigram, trigram
  global uni_tcount, uni_vsize, bi_tcount, bi_vsize, tri_tcount, tri_vsize
  #
  print('.loading unigrams...')
  f = open('./w2v/ew9.uni.pkl', 'rb')
  unigram = cPickle.load(f)
  uni_tcount = unigram.N()
  uni_vsize = unigram.B()
  f.close()
  #
  print('..loading bigrams...')
  f = open('./w2v/ew9.bi.pkl', 'rb')
  bigram = cPickle.load(f)
  bi_tcount = bigram.N()
  bi_vsize = bigram.B()
  f.close()
  #
  print('...loading trigrams...')
  f = open('./w2v/ew9.tri.pkl', 'rb')
  trigram = cPickle.load(f)
  tri_tcount = trigram.N()
  tri_vsize = trigram.B()
  f.close()
  
#
def uni_p(w):
  return ngram_p((w), unigram, uni_tcount, uni_vsize)

def bi_p(w1, w2):
  return ngram_p((w1, w2), bigram, bi_tcount, bi_vsize)

def tri_p(w1, w2, w3):
  return ngram_p((w1, w2, w3), trigram, tri_tcount, tri_vsize)

def ngram_p(w, tbl, tcount, vsize):
  if not w in tbl:
    count = 1
  else:
    count = tbl[w]
  #return float(count)/(tbl.N()+tbl.B())
  return float(count)/(tcount+vsize)

#
import math

def log2(x):
  return math.log(x, 2)

# tokens probability
import numpy as np

def tokens_p(tokens):
  # could be tuned...
  lambdas = np.array([0.7, 0.2, 0.1])
  #
  log_p = 0.0
  for i in range(len(tokens)):
    u = uni_p((tokens[i]))
    b = get_bi_p(tokens, i)
    t = get_tri_p(tokens, i)
    log_p += log2(np.dot(lambdas, np.array([t, b, u])))
  #return math.pow(2, log_p)
  return log_p

def get_bi_p(tokens, i):
  if (i+1) >= len(tokens):
    return 0.0
  else:
    return bi_p(tokens[i], tokens[i+1])

def get_tri_p(tokens, i):
  if (i+2) >= len(tokens):
    return 0.0
  else:
    return tri_p(tokens[i], tokens[i+1], tokens[i+2])
