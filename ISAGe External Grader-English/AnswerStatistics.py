###### Answer Length statistics features.
import Similarities as ss
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import codecs as c
import numpy as np
import pandas as pd
import io
import os
import sys
import xml.etree.ElementTree as etree
from nltk import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from io import StringIO
import csv
from nltk import word_tokenize
import re
from math import sqrt
from scipy.stats.stats import pearsonr
from math import *
import linecache as cc
from collections import Counter
from nltk.stem import SnowballStemmer, PorterStemmer
import string
from nltk.corpus import stopwords
import json

def getOccurences(sentence):                ##### Occurrence calculation
    try:
        vec = CountVectorizer()
        X = vec.fit_transform([sentence]).toarray()
        lengthSentence = np.sum(X[0])
        occurenceGreatToOne = []
        for i in X[0]:
            if i > 1:
                occurenceGreatToOne.append(i)
    except:
        occurenceGreatToOne=0
        lengthSentence=1
    return occurenceGreatToOne,lengthSentence

def getRedondanceFreq(sentence):
    occurences,length=getOccurences(sentence)
    return np.sum(occurences)/length

####### Redundancy frequency feature
def getRedondanceFreqResponses(AllResponsesCorpus):
    AllFrequencies=[]
    QuestionFrequencies=[]
    for questionCorpus in AllResponsesCorpus:
        for response in questionCorpus:
            QuestionFrequencies.append(getRedondanceFreq(response))
        AllFrequencies.append(QuestionFrequencies)
        QuestionFrequencies=[]

    return AllFrequencies
##### Answer length Feature
def getResPonsesLength(AllQuestionCorpus):
    AllResponsesLength = []
    QuestionLengths = []
    for questionCorpus in AllQuestionCorpus:
        for response in questionCorpus:
            QuestionLengths.append(len(response))
        AllResponsesLength.append(QuestionLengths)
        QuestionLengths=[]
    return AllResponsesLength

##### Difference length between Reference answer and student answer Feature
def getResPonsesDiffLength(AllQuestionCorpus,ResponsesModels):
    AllResponsesLength = []
    QuestionLengths = []
    for i,questionCorpus in enumerate(AllQuestionCorpus):
        for response in questionCorpus:
            QuestionLengths.append(abs(len(response)-len(ResponsesModels[i])))
        AllResponsesLength.append(QuestionLengths)
        QuestionLengths=[]
    return AllResponsesLength