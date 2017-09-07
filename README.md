# Project - Substitution-ranking-in-lexical-simplification
This project includes: (libraries)

| gensim | math | lxml | nltk | sklearn | scipy |
| :------: | :-------: | :------: | :------: | :------: | :------: |
| analyze text | Calculation | xml file | Natural Language | Machine Learning | math, science|


## Test Environment
-  python3.6
-  conda 4.3.17


## Update Flow



### Develop
### Ordinal way
0. install virtualenv
```
$ sudo easy_install virtualenv
```

```
Installed /Library/Python/2.7/site-packages/virtualenv-1.11.6-py2.7.egg
Processing dependencies for virtualenv
Finished processing dependencies for virtualenv
$
```
1. create project
```
$ mkdir Lexical-Simplification
$ virtualenv --no-site-packages  Lexical-Simplification
```
2. start
```
$ cd Lexical-Simplification
$ source bin/activate
```

```
(Lexical-Simplification)% deactivate                                                         
%
```

### Today

0. download (https://repo.continuum.io/archive/Anaconda3-4.4.0-MacOSX-x86_64.pkg)

1. create environment

```
conda create -n py36 python=3.6 anaconda
```

2. check whether you created the environment

```
conda info -e
```

more information (http://conda.pydata.org/docs/using/envs.html#share-an-environment)

3. conda install (libraries)


### When Push
1. Check your branch latest version.
2. Check your checked out branch correct.

## Construct Development Environment
When we develop this project, we are using atom - [atom](https://atom.io/).
We recommend you to use atom if you collaborate with us.

```
$ git clone git@github.com:MADONOKOUKI/Substitution-ranking-in-lexical-simplification.git
```
- source activate (python3.6's environment)

- fix file

__Create csv file__

- python3 make_feature_csv.py test/test.xml(train/train.xml) test/test.gold_ranking(train/train.gold_ranking) [csv file(pointwise)]   [csv file(pairwise)]

__Excecure python file (for result)__

- python3 make_Kappa_calc.py(make_Kappa_calc_pairw.py) (train's csv file) (test's csv file)  (temporal result file) gold_test(answer file)

- Get result and can watch a picture( graph )

[if you wan to use w2v data] ask : (madonomadonorunning@gmail.com)
