#!/usr/bin/env python
# encoding: utf-8
# scorer.py
# Created by Sujay Kumar Jauhar on 2011-11-27.

import sys
import getopt
import re
import itertools


help_message = '''
$ python rank-scorer.py -i <systemFile> -g <goldFile> 
-i or --input to specify path to system generated responses
-g or --gold to specify path to gold file
-v or --verbose to set verbose to True (verbose is False by default)
-h or --help (this message is displayed)
'''


class Usage(Exception):
  def __init__(self, msg):
    self.msg = msg


def readCommandLineInput(argv):
  try:
    try:
      #specify the possible option switches
      opts, args = getopt.getopt(argv[1:], "hi:g:v", ["help", "input=", "gold=", "verbose"])
    except:
      raise Usage(msg)
    
    #default values
    inputFile = None
    goldFile = None
    verbose = False
    
    # option processing
    # print(opts)
    for option, value in opts:
      if option == "-v":
        verbose = True
      elif option in ("-h", "--help"):
        raise Usage(help_message)
      elif option in ("-i", "--input"):
        inputFile = value
      elif option in ("-g", "--gold"):
        goldFile = value
      else:
        raise Usage(help_message)
        
    if (inputFile==None) or (goldFile==None):
      raise Usage(help_message)
    else:
      return (inputFile,goldFile,verbose)
  
  except:
    # print(str(err.msg))
    return 2

#function to read system produced ranking file
def getSystemRankings(file):
  #pattern to recognize rankings in output file
  pattern = re.compile('.*?\{(.*?)\}(.*)')
  
  #extract rankings
  allContextRankings = []
  for i,line in enumerate(file):
    rest = line
    # print(i)
    currentContextRanking = {}
    counter = 0
    while rest:
      match = pattern.search(rest)
      # print(match)
      currentRank = match.group(1)
      # print(currentRank)
      individualWords = currentRank.split(', ')
      # print(individualWords)
      for word in individualWords:
        # print(word)
        # can't understand
        word = re.sub('\s$','',word)
        # print(word)
        currentContextRanking[word] = counter
      rest = match.group(2)
      # print(rest)
      counter += 1
    
    allContextRankings.append(currentContextRanking)
    # print(currentContextRanking)
  return allContextRankings

#comparator function
def compare(val1, val2):
  if (val1 < val2):
    return -1
  elif (val1 > val2):
    return 1
  else:
    return 0

#function to score system with reference to gold
#function is based on kappa with rank correlation
def getScore(system, gold, verbose):
  
  #intialize vars
  totalPairCount = 0
  agree = 0
  equalAgree = 0
  #greaterAgree = 0
  #lesserAgree = 0
  
  contextCount = 0
  #go through contexts parallely
  for (sysContext, goldContext) in zip(system, gold):
    contextCount += 1
    if verbose:
      print('Looking at context', contextCount)
    #go through each combination of substitutions

    for pair in itertools.permutations(sysContext.keys(), 2):
      totalPairCount += 1
      #find ranking order given by system and gold for current pair
      # print(pair[0])
      # print(pair[1])
      # print(sysContext)
      # print(goldContext)
      # print(type(goldContext))
      if(pair[0] in goldContext.keys() and pair[1] in goldContext.keys()):
        sysCompareVal = compare(sysContext[pair[0]],sysContext[pair[1]])
        goldCompareVal = compare(goldContext[pair[0]],goldContext[pair[1]])

        # print(pair[0])
        # print(pair[1])

        # print(sysCompareVal)
        # print(goldCompareVal)
        
        #print verbose information
        if verbose:
          print('\tCurrent pair of words: "' + pair[0] + '" & "' + pair[1] + '"')
          print('\t\tSystem says rank of: "' + pair[0] + '" is',)
          if sysCompareVal == -1:
            print('lesser than "' + pair[1] + '"')
          elif sysCompareVal == 1:
            print('greater than "' + pair[1] + '"')
          else:
            print('equal to "' + pair[1] + '"')
            
          print('\t\tGold says rank of: "' + pair[0] + '" is',)
          if goldCompareVal == -1:
            print('lesser than "' + pair[1] + '"')
          elif goldCompareVal == 1:
            print('greater than "' + pair[1] + '"')
          else:
            print('equal to "' + pair[1] + '"')
        
        #system and gold agree
        #add appropriate counts to agree count
        if (sysCompareVal) == (goldCompareVal):
          agree += 1
          if verbose:
            print("\tAgreement count incremented by 1")
            
        #add count if at least one of them tied candidate pair
        if sysCompareVal == 0:
            equalAgree += 1
            if verbose:
              print("\tEqualTo count incremented by 1")
        if goldCompareVal == 0:
            equalAgree += 1
            if verbose:
              print("\tEqualTo count incremented by 1")
              
        if verbose:
          print('\n')
  
  # if(totalPairCount == 0): equalAgreeProb = 0
  # else: 
  equalAgreeProb = float(equalAgree)/float(totalPairCount*2)
  
  #P(A) and P(E) values 
  # if(totalPairCount == 0): absoluteAgreement = 0
  # else: 
  absoluteAgreement = float(agree)/float(totalPairCount)
  chanceAgreement = (3*pow(equalAgreeProb,2)-2*equalAgreeProb+1.0)/2.0
  
  #return kappa score
  print(absoluteAgreement)
  print(chanceAgreement)
  print(absoluteAgreement - chanceAgreement)

  return (absoluteAgreement - chanceAgreement)/(1.0 - chanceAgreement)
  

if __name__ == "__main__":
  #parse command line input
  commandParse = readCommandLineInput(sys.argv)
  #failed command line input
  if commandParse==2:
    sys.exit(2)
  
  #try opening the specified files  
  # try:
  systemRespFile = open(commandParse[0])
  goldFile = open(commandParse[1])
  # except:
    # print("ERROR opening files. One of the paths specified was incorrect."
    # sys.exit(2)
  
  #get system rankings and store in structure
  systemRankings = getSystemRankings(systemRespFile)
  goldRankings = getSystemRankings(goldFile)
  # print(systemRankings)
  # print("---------------------")
  # print(goldRankings)
  # print(commandParse)
  score = getScore(systemRankings, goldRankings, commandParse[2])
  
  print('Normalized system score:', score)
