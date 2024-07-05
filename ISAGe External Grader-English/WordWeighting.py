############# Word Weighting features
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
from math import sqrt
from math import *
import linecache as cc
from collections import Counter
from nltk.stem import SnowballStemmer, PorterStemmer
import string
from nltk.corpus import stopwords
import pickle
from sklearn.preprocessing import PolynomialFeatures
import json
import sys
import  nltk
from DomainKnowledge import getWordContext , getVectorContextOfSentence
import GlobalVar as glob

#################### TF-MinMax Ponderation Semantic Space (TFlog) ##############

#this function returns the tf-minmax frequency from the dictionary
def getWordFrequency(word, dictionaire):
    try:
        return dictionaire[word];
    except:
        if len(word) > 2:
            for d in dictionaire:
                if d.find(word):
                    return dictionaire[d]

    return 0

#load the dictionary from path
def getDictionnaireTFIDF(pathDictio):
  with io.open(pathDictio)as json_file:
    dictio = json.load(json_file)
  return dictio

#this function returns the tf-minmax frequency from the dictionary
def getWordTfMinMaxPonderation(dictionnaire,tfminmax, word):
    try:
        key=list(dictionnaire.keys())[list(dictionnaire.values()).index(word)]
        pond=tfminmax[key]
    except:
        for key , val in dictionnaire.items():
            if val.find(word):
                pond=tfminmax[key]
                break

    return pond

#this function return the Semantic Space vector with  tf-minmax ponderation
def getWordVectorContextMinMaxPonderation(dictionnaire,tfminmax,word):
    word_context=getWordContext(word)
    pond=getWordTfMinMaxPonderation(dictionnaire,tfminmax,word)
    return np.dot(pond,word_context)

#this function returns the Semantic Space sentence vector with  tf-minmax ponderation
def getSentenceVectorContextMinMaxPonderation(dictionnaire,tfminmax,sentence):
    words = word_tokenize(sentence)
    #print(words)
    sentenceContext = [[0 for i in range(glob.EsLength + 1)]]
    for word in words:
        sentenceContext = np.add(sentenceContext, getWordVectorContextMinMaxPonderation(dictionnaire,tfminmax,word))
    return sentenceContext

#this function returns the set of question answers with Tf-MinMax ponderation for SemanticSpace
def getSimForQuestionMinMaxPond(CorpusReponses,ResponseModel):
    dictionnaire,tfidfminmax=getDictionnaireTFIDF(glob.pathDictionnaire),getDictionnaireTFIDF(glob.pathMinMax)
    modelContext=getSentenceVectorContextMinMaxPonderation(dictionnaire,tfidfminmax,ResponseModel)
    sim = []
    for i in range(len(CorpusReponses)):
        response = CorpusReponses[i]
        sim.append(
            cosine_similarity(getSentenceVectorContextMinMaxPonderation(dictionnaire,tfidfminmax,response),modelContext)[0][0])
    return sim


################# POS_Tag Ponderation for Semantic Space ###############

#this function returns the type of word using PosTagger
    
def pos_tag_english(mot):
    mot=mot.split()
    tag=nltk.pos_tag(mot)[0]
    if(tag[1] == 'VBP' or tag[1] =='VBD' or tag[1] =='VBZ' or tag[1] =='VBG'or tag[1]=='VB' ):
        return 'V'
    elif(tag[1] == 'NN' or tag[1] =='NNS' or tag[1] =='NNP' or tag[1] =='DTNN' or tag[1] =='DTNNP' or tag[1] =='DTJJ' or tag[1] =='DTNNS' or tag[1] =='JJ' or tag[1] =='VN' ): #si le tag est un nom
        return 'N'
    else : #other tag
        return 'A'

#this function return the word context with postag ponderation from Semantic Space
def getWordContextPonderation(word):
    
    ponderation = 0
    pos_tagg = pos_tag_english(word)
        
    if pos_tagg == 'V':
        ponderation = 0.5
    elif pos_tagg == 'N':
        ponderation = 0.3
    else:
        ponderation = 0.2
    vec1 = getWordContext(word)
    return np.dot(vec1, ponderation)

#this function returns the sentence context with postag ponderation from Semantic Space
def getVectorContextOfSentencePonderation(sentence):
    words = word_tokenize(sentence)
    sentenceContext = [[0 for i in range(glob.EsLength + 1)]]
    for word in words:
        sentenceContext = np.add(sentenceContext, getWordContextPonderation(word))
    return sentenceContext

#this function return the set of question answers with Tf-MinMax ponderation for Semantic Space
def CosineSimForQuestionES_WithPonderation(CorpusReponses, ReponseModel):
    sim = []
    responseModelPond = getVectorContextOfSentencePonderation(ReponseModel)

    for i in range(len(CorpusReponses)):
        response = CorpusReponses[i]
        sim.append(
            cosine_similarity(getVectorContextOfSentencePonderation(response), responseModelPond)[0][0])
    return sim


########### Ponderation WE #########
def getWEMinMaxPonderation(dictionnaire, tfminmax, word, We_dict):
    if (word in We_dict.keys() and word in dictionnaire.values()):
        word_We = We_dict.get(word)
        pond = getWordTfMinMaxPonderation(dictionnaire, tfminmax, word)
        return np.dot(pond, word_We)
    else:
        return np.zeros(300)

#this function return the WE sentence vector with  tf-minmax ponderation
def getSentenceWEtMinMaxPonderation(dictionnaire, tfminmax, sentence, We_dict):
    words = word_tokenize(sentence)
    sentenceContext = np.zeros(300)
    for word in words:
        sentenceContext = np.add(sentenceContext, getWEMinMaxPonderation(dictionnaire, tfminmax, word, We_dict))
    return sentenceContext

def getSimWEForQuestionMinMaxPond(CorpusReponses, ResponseModel, We_dict, dictionnaire, tfidfminmax):
    modelContext = getSentenceWEtMinMaxPonderation(dictionnaire, tfidfminmax, ResponseModel, We_dict)
    sim = []
    for i in range(len(CorpusReponses)):
        response = CorpusReponses[i]
        sim.append(cosine_similarity(
            (getSentenceWEtMinMaxPonderation(dictionnaire, tfidfminmax, response, We_dict)).reshape(1, -1),
            modelContext.reshape(1, -1))[0][0])
    return sim

#this function return the English WE sentence vector with  tf-minmax ponderation
def getWE_PosTagPonderation_english(word, vect_word):
    tag = (pos_tag_english(word))
    poid = 0
    if (tag == 'V'):
        poid = 0.5
    elif (tag == 'N'):
        poid = 0.3
    else:
        poid = 0.2

    return np.dot(vect_word, poid)

def getWE_PosTagPonderation(word, vect_word):
    tag = (pos_tag_english(word))
    poid = 0
    if (tag == 'V'):
        poid = 0.5
    elif (tag == 'N'):
        poid = 0.3
    else:
        poid = 0.2

    return np.dot(vect_word, poid)

#this function return the sentence vector of context using WE with posta for student answers
def WE_AllQuestionCorpusPos(AllQuestionCorpus, dictionnaireWE, isPos=True):
    vecteur_zero = np.zeros((300))
    AllQuestionCorpusWE = []
    for i in range(len(AllQuestionCorpus)):
        WE = []
        for phrases in AllQuestionCorpus[i]:
            somme = np.zeros((300))
            Mots = word_tokenize(phrases)
            for word in list(Mots):
                if word in list(glob.stopWords):
                    Mots.remove(word)
            for mot in Mots:
                if mot in dictionnaireWE.keys():
                    if isPos:
                        somme = np.add(somme, getWE_PosTagPonderation(mot, dictionnaireWE.get(mot)))
                    else:
                        somme = np.add(somme, dictionnaireWE.get(mot))
                else:
                    somme = np.add(somme, vecteur_zero)
            WE.append(somme)
        AllQuestionCorpusWE.append(WE)

    return AllQuestionCorpusWE

#this function returns the sentence vector of context using WE with posTag for reference answers
def WE_ModelResponsesPos(ModelResponses, dictionnaireWE, isPos=True):
    vecteur_zero = np.zeros(300)
    AllModelResponsesWE = []
    for phrase in ModelResponses:
        somme = np.zeros(300)
        mots = word_tokenize(phrase)
        for word in list(mots):
            if word in list(glob.stopWords):
                mots.remove(word)

        for mot in mots:
            if mot in dictionnaireWE.keys():
                if isPos:
                    somme = np.add(somme, getWE_PosTagPonderation(mot, dictionnaireWE.get(mot)))
                else:
                    somme = np.add(somme, dictionnaireWE.get(mot))
            else:
                somme = np.add(somme, vecteur_zero)

        AllModelResponsesWE.append(somme)

    return AllModelResponsesWE