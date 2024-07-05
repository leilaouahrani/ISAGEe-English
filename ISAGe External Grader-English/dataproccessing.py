##### Data Preprocessing
 
import codecs as c
import numpy as np
import pandas as pd
import io
import os 
import sys
from nltk import word_tokenize
from io import StringIO
import csv
from nltk import word_tokenize
import re
from math import sqrt
from math import *
import linecache as cc
from collections import Counter
from nltk.stem import SnowballStemmer, PorterStemmer
import string
from nltk.corpus import stopwords
import pickle
import json
import sys
from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import GlobalVar as glob

####### Get Semantic Space, StopWords,Tf-Minmax weights ... for Arabic,

# Word Embeddings English load
def loadModelWeEnglish(path):
    # for Skip
    model = KeyedVectors.load_word2vec_format(path, binary=True, unicode_errors='ignore')

    words = [word for word in model.vocab]
    word_vectors = np.array([model[w] for w in words])

    model = {}

    vecDic = {}
    for i in range(0, len(words)):
        vecDic[words[i]] = word_vectors[i]
    return vecDic

#get StopWords , SemanticSpace,Tf-Minmax
def generateModelsForLanguage():
    getSemanticSpace()
    generateStopWords()
    glob.pathMinMax = glob.pathMinMaxEnglish
    glob.pathDictionnaire = glob.pathDictionnaireEnglish

def getSemanticSpace():
    glob.data = pd.read_csv(glob.pathESEnglish,header=None).__array__()
    glob.words = []
    glob.EsPath = glob.pathESEnglish
    glob.EsLength = len(glob.data)-1
    glob.DictioWE = loadModelWeEnglish(glob.pathWEEnglish)

    with io.open(glob.pathWordsEnglish, 'r', encoding="utf-8") as f:
        for i, line in enumerate(f):
            glob.words.append(line.replace("\n", ""))
#########

def generateStopWords():
    glob.stopWords = []
    with io.open(glob.pathStopWordsEnglish, encoding='utf-8') as f:
           for word in f:
               glob.stopWords.append(word.replace("\n", ""))


### Data  cleaning data
def clean_data_english(sentence):
    sentence = re.sub(
        "[\(|=|.|,|;|،|\)|<|>|!|?|\$|£|€|:|\ufeff|»|«|/|\+|\*|\(\)|\-|\[|\]|\(|\)|\{|\}|_|é|ù|è|؛|–|’\|/|؛|'\|…|ـ|&|؟|%|\“|\"|—|\”|@|\‘َ|ُ|ْ|َِِ|ّ|ً|ٌ]",
        " ", sentence)
    sentence = re.sub("[\n|\r]", " ", sentence)
    return sentence.lower()

def sentenceRemoveStop(sentence):
    s = ""
     
    for w in word_tokenize(sentence):
        if w not in glob.stopWords:
            s = s + w + " "
    return s


def StemSentenceEnglish(Sentence, isSnow=False):
    stemmer = SnowballStemmer("english")
    porterstemmer = PorterStemmer()
    Words = Sentence.split()
    phrase = ""
    for word in Words:
        if isSnow:
            stemWord = stemmer.stem(word)
        else:
            stemWord = porterstemmer.stem(word)
        phrase = phrase + " " + stemWord
    return phrase

########## Stemming et cleaning  data #########

#Stemming  Answers ex: [['Answer1 for Question 1','Answer2 for Question 1','Answer3 for Question 1'],['Answer1 for Question 2','Answer2 for Question 2'] ...]
def stemAllCorpus(AllCorpus,isLight=True):
    newAll=[]
    for i in AllCorpus:
        corpus=[]
        for j in i:

               corpus.append(StemSentenceEnglish(sentenceRemoveStop(clean_data_english(j))))
              
        newAll.append(corpus)
        corpus=[]
    return newAll

#Stem set of Model Answers exemple of set: ['Model Answer for Question 1','Model Answer for Question 2' ...]
def stemAllModel(AllCorpus,isLight=True):
    newAll=[]
    for j in AllCorpus:
            newAll.append(StemSentenceEnglish(clean_data_english(sentenceRemoveStop(j))))
            
    return newAll
