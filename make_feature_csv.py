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


#引数の長さ確認

def check_argvs(argvs):
  if(len(argvs) < 3):
    print("insufficient files you selected")
    quit()
  if(len(argvs) > 3):
    print("too many files you selected")
    quit()

### Main #####
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
import sys

lemmatizer = nltk.WordNetLemmatizer()

if  __name__ == "__main__":
  # 引数読み込み
  argvs = sys.argv
  check_argvs(argvs)

  #xmlとsubstitution読み込み
  print(argvs)
  ctx_f = argvs[1]
  ranking_f =  argvs[2]
  try:
    sentences = read_contexts(ctx_f)
    cand_lists = read_gold_rankings(ranking_f)
  except:
    print("Oops!  That was mistakes.  Try again...")
  print(sentences)
  verbose = True
  for i, s in enumerate(sentences):
    #
    if verbose: print('Sentence', i, '---')
    ind, tgt = s[1:3]
    tgt_pos = s[3].split('.')[1]
    tgt_lemma = lemmatizer.lemmatize(tgt, tgt_pos) 
    if verbose: print("Subst target/pos/lemma:", tgt, tgt_pos, tgt_lemma)
    #