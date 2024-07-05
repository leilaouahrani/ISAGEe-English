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
from nltk.stem.isri import ISRIStemmer
from nltk import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from io import StringIO
import csv
from nltk import word_tokenize
import re
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.stats.stats import pearsonr
from math import *
# from stanfordcorenlp import StanfordCoreNLP
import linecache as cc
from collections import Counter
from nltk.stem import SnowballStemmer, PorterStemmer
import string
from nltk.corpus import stopwords
import pickle
from sklearn.preprocessing import PolynomialFeatures
import json
import sys
import GlobalVar as glob
import dataproccessing


#this file for Specific/General domain caracteristic such as SemanticSpace and WordEmbeddings (Semantic Similarities)

#this function returns a context vector of word from SemanticSpace matrix
def getWordContext(word):
    index = []
    try:
        index.append(glob.words.index(word))
    except ValueError:
      if len(word) > 2:
        for i, item in enumerate(glob.words[0:glob.EsLength+1]):
            if item.find(word) != -1:
                index = [i]
                break

    try:
        f = StringIO(glob.data[index[0]][0])
        vec1 = np.array([np.loadtxt(f)])
    except:
        vec1 = [0 for i in range(glob.EsLength + 1)]

    return vec1

#this function returns a context vector of sentence from SemanticSpace matrix
def getVectorContextOfSentence(sentence):
    wordss = word_tokenize(sentence)
    sentenceContext = [[0 for i in range(glob.EsLength + 1)]]
    for word in wordss:
        sentenceContext = np.add(sentenceContext, getWordContext(word))
    return sentenceContext

#this function calculate the cosine similarity between the context vector of StudentAnswer and ModelAnswer
def CosineSimForQuestionES(CorpusReponses, ReponseModel):
    sim = []
    responseModelCaontext = getVectorContextOfSentence(ReponseModel)

    for i in range(len(CorpusReponses)):
        response = CorpusReponses[i]
        sim.append(
            cosine_similarity(getVectorContextOfSentence(response), responseModelCaontext)[0][0])
    return sim




###########  WE Section ###############

#this function return the sentence vector of context using WE for all answers
def WE_AllQuestionCorpus(AllQuestionCorpus, dictionnaireWE, isPos=False):
        return WE_AllQuestionCorpusENG(AllQuestionCorpus, dictionnaireWE, isPos=False)

#this function return the sentence vector of context using WE for modal answers
def WE_ModelResponses(ModelResponses, dictionnaireWE, isPos=False):

        return WE_ModelResponsesENG(ModelResponses, dictionnaireWE, isPos=False)


#this function return the sentence vector of context using English WE   for all answers
def WE_AllQuestionCorpusENG(AllQuestionCorpus, dictionnaireWE, isPos=False):
    vecteur_zero = np.zeros((300))
    AllQuestionCorpusWE = []
    for i in range(len(AllQuestionCorpus)):
        WE = []
        for phrases in AllQuestionCorpus[i]:
            somme = np.zeros((300))
            Mots = word_tokenize(phrases)
            for word in list(Mots):
                if word in stopwords.words('english'):
                    Mots.remove(word)
            for mot in Mots:
                if mot in dictionnaireWE.keys():
                    if isPos:
                        somme = np.add(somme, getWE_PosTagPonderation_english(mot, dictionnaireWE.get(mot)))
                    else:
                        somme = np.add(somme, dictionnaireWE.get(mot))
                else:
                    somme = np.add(somme, vecteur_zero)
            WE.append(somme)
        AllQuestionCorpusWE.append(WE)

    return AllQuestionCorpusWE

#this function return the sentence vector of context using English WE  for modal answers
def WE_ModelResponsesENG(ModelResponses, dictionnaireWE, isPos=False):
    vecteur_zero = np.zeros(300)
    AllModelResponsesWE = []
    for phrase in ModelResponses:
        somme = np.zeros(300)
        mots = word_tokenize(phrase)
        for word in list(mots):
            if word in stopwords.words('english'):
                mots.remove(word)

        for mot in mots:
            if mot in dictionnaireWE.keys():
                if isPos:
                    somme = np.add(somme, getWE_PosTagPonderation_english(mot, dictionnaireWE.get(mot)))
                else:
                    somme = np.add(somme, dictionnaireWE.get(mot))
            else:
                somme = np.add(somme, vecteur_zero)

        AllModelResponsesWE.append(somme)

    return AllModelResponsesWE


def Cosinus_We(ResponsesCorpus, ModelResponse):
    sim = []
    for response in ResponsesCorpus:

        sim.append(cosine_similarity(response.reshape(1, -1), ModelResponse.reshape(1,-1).reshape(1,-1))[0][0])
    return sim


#Returns the context vector of sentence from WE 
def getVecteurSentenceWE(Sentence, dictWE):
    Mots = word_tokenize(Sentence)
    vecteurzero = np.zeros(300)
    Somme = np.zeros(300)
    for mot in Mots:
        if (mot in dictWE.keys()):
            Somme = np.add(Somme, dictWE.get(mot))
        else:
            Somme = np.add(Somme, vecteurzero)

    return Somme
 










