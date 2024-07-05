# coding=utf-8
#coding=utf-8
### Python 3.7
### We have adopted the same approach as for Arabic.
### The same model is trained with the same features.
### The difference tudiantlies in the data processing (stemming, stopwords, Postagging, ...).

########## Main External Grader( : Trained Model = Linear Ridge PolynomialFeatures d° 2 ###########
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import PolynomialFeatures
import codecs as c
import numpy as np
import pandas as pd
import nltk
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
from sklearn.metrics import mean_squared_error
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
import dataproccessing as prep
import GlobalVar as glob
import DomainKnowledge
import AnswerStatistics
import WordWeighting
import GapQuestion

########## this function predicts the student grade included in [0..5] or a similarity betwenn 0 and 1
def getScore(Question,ReferenceAnswer,StudentAnswer,DiffulteQuestion):

    # Load the specific domain model(Semantic Space) and the General domain model(Word Embeddings)
    prep.generateModelsForLanguage()

    #Stemming answers
    StudentAnswerH = prep.stemAllCorpus(StudentAnswer, isLight=False)                # Heavy stem
    ReferenceAnswerH = prep.stemAllModel(ReferenceAnswer, isLight=False)             # Heavy stem

    StudentAnswerL = prep.stemAllCorpus(StudentAnswer, isLight=True)                 # light stem
    ReferenceAnswerL = prep.stemAllModel(ReferenceAnswer, isLight=True)              # light stem

#######################  Features Extraction   ################"

    ########### 1. Answer Length statistics features#########
    ##### Redundancy frequency
    AllQuestionRedondanceFreq = AnswerStatistics.getRedondanceFreqResponses(StudentAnswer)

    ##### Length Difference between Reference(model) answer and the student answer
    AllQuestionDiffLength = AnswerStatistics.getResPonsesDiffLength(StudentAnswer, ReferenceAnswerH)

    ##### student answer's Length
    AllQuestionLength = AnswerStatistics.getResPonsesLength(StudentAnswer)

    ########### 2. Lexical Similarities Features #########

    #### Jaccard Similarity
    print("ORDER SIMILARITY:")
    AllQuestionSimOrder = []
    for i in range(len(StudentAnswerH)):
        simES = AnswerStatistics.ss.SimOrderForQuestion(StudentAnswerH[i], ReferenceAnswerH[i])
        AllQuestionSimOrder.append(simES)
        print(simES)

    #### Jaccard Similarity
    print("JACCARD SIMILARITY:")
    AllQuestionSimJacc = []
    for i in range(len(StudentAnswerH)):
        simES = AnswerStatistics.ss.SimJaccardForQuestion(StudentAnswerH[i], ReferenceAnswerH[i])
        AllQuestionSimJacc.append(simES)
    print(simES)
    
    #### Dice Similarity
    print("DICE SIMILARITY:")
    AllQuestionSimDice = []
    for i in range(len(StudentAnswerH)):
        simES = AnswerStatistics.ss.SimDiceForQuestion(StudentAnswerH[i], ReferenceAnswerH[i])
        AllQuestionSimDice.append(simES)
    print(simES)

    #### Jaro Similarity
    print("JARO SIM:")
    AllQuestionSimJaro = []
    for i in range(len(StudentAnswerH)):
        sim = AnswerStatistics.ss.SimJaroForQuestion(StudentAnswerH[i], ReferenceAnswerH[i])
        AllQuestionSimJaro.append(sim)
    print(sim)
 
    #### STS Similarity
    print("STS SIM:")
    AllQuestionSimSTS = []
    for i in range(len(StudentAnswerH)):
        sim = AnswerStatistics.ss.SimStsForQuestion(StudentAnswerH[i], ReferenceAnswerH[i])
        AllQuestionSimSTS.append(sim)
    print(sim)

    ###### 3. Semantic Space Model for learning domain-specific features combined with Word Weighting features #########

    #### Cosine(ModelAnswer,StudentAnswer) using Semantic Space
    print("#Cosine(ModelAnswer,StudentAnswer) using SemanticSpace")
    AllQuestionSimES = []
    for i in range(len(StudentAnswerL)):
        simES = DomainKnowledge.CosineSimForQuestionES(StudentAnswerL[i], ReferenceAnswerL[i])
        AllQuestionSimES.append(simES)
    print(simES)

    #### Cosine(ModelAnswer,StudentAnswer) using Semantic Space with POS Tagging Ponderation
    print("# Cosine(ModelAnswer,StudentAnswer) using SemanticSpace with PosTag Ponderation")
    AllQuestionSimPos = []
    for i in range(len(StudentAnswerL)):
        simES = WordWeighting.CosineSimForQuestionES_WithPonderation(StudentAnswerL[i], ReferenceAnswerL[i])
        AllQuestionSimPos.append(simES)
        print(simES)
        
    #### Cosine(ModelAnswer,StudentAnswer) using Semantic Space with NTFLog Ponderation [NTFlog (w) =TFlog (w)/Max (TFlog)] named here as TFMINMAX
    print("# Cosine(ModelAnswer,StudentAnswer) using SemanticSpace with TF-MinMax Ponderation")
    AllQuestionSimES_TFMINMAX = []
    for i in range(len(StudentAnswerL)):
        simES = WordWeighting.getSimForQuestionMinMaxPond(StudentAnswerL[i], ReferenceAnswerL[i])
        AllQuestionSimES_TFMINMAX.append(simES)
        print(simES)

    ####### 4. Word Embeddings Model for learning domain-general knowledge Text Similarity Features combined with Word Weighting features #############
    WE_ResponsesVectors = DomainKnowledge.WE_AllQuestionCorpus(StudentAnswer, glob.DictioWE)
    WE_ModelsVectors = DomainKnowledge.WE_ModelResponses(ReferenceAnswer, glob.DictioWE)

    #### Cosine(ModelAnswer,StudentAnswer) using WE
    print("# Cosine(ModelAnswer,StudentAnswer) using WE")
    AllQuestionSimCosinusWE = []
    for i in range(len(WE_ResponsesVectors)):
        simCos = DomainKnowledge.Cosinus_We(WE_ResponsesVectors[i], WE_ModelsVectors[i])
        AllQuestionSimCosinusWE.append(simCos)
        print(simCos)

    #### Cosine(ModelAnswer,StudentAnswer) using WE with TFMinMax Ponderation
    print("# Cosine(ModelAnswer,StudentAnswer) using WE with TF-MinMax Ponderation")
    AllSimCosinusMinMaxWE = []
    for i in range(len(StudentAnswer)):  # tfminmax/ tfidfminmax
        sim = WordWeighting.getSimWEForQuestionMinMaxPond(StudentAnswer[i], ReferenceAnswer[i], glob.DictioWE, WordWeighting.getDictionnaireTFIDF(
            glob.pathDictionnaire), WordWeighting.getDictionnaireTFIDF(glob.pathMinMax))
        AllSimCosinusMinMaxWE.append(sim)
    print(sim)
    
    WE_ResponsesVectors = WordWeighting.WE_AllQuestionCorpusPos(StudentAnswer, glob.DictioWE,isPos=True)
    WE_ModelsVectors = WordWeighting.WE_ModelResponsesPos(ReferenceAnswer, glob.DictioWE,isPos=True)

    #### Cosine(ModelAnswer,StudentAnswer) using WE with POS Tagging  Ponderation
    print(" # Cosine(ModelAnswer,StudentAnswer) using WE with PosTag Ponderation")
    AllQuestionSimCosinusWEPos = []
    for i in range(len(WE_ResponsesVectors)):
        simCos = DomainKnowledge.Cosinus_We(WE_ResponsesVectors[i], WE_ModelsVectors[i])
        AllQuestionSimCosinusWEPos.append(simCos)
    print(simCos)

    ################# 5. Gap (Qestion-deviation) Features using Semantic Space
    print("#Gap (Qestion-deviation) caractéristics using SemanticSpace")
    ESExpectExprimEtoile, ESExpectExprimMoins, ESEcartInfoEtoile, ESEcartInfoMoins, ESGapPndEtoile, ESGapPndMoins = GapQuestion.GapForAllResponses(
        StudentAnswerL, ReferenceAnswerL, Question)
    print(ESExpectExprimEtoile, ESExpectExprimMoins, ESEcartInfoEtoile, ESEcartInfoMoins, ESGapPndEtoile, ESGapPndMoins)

    ################# 6. Gap (Qestion-deviation) Features using WE : The same features are calculated using WE vectors
    print("Gap (Qestion-deviation) caractéristics using WE")
    ExpectExprimEtoileWE, ExpectExprimMoinsWE, EcartInfoEtoileWE, EcartInfoMoinsWE, GapPndEtoileWE, GapPndMoinsWE = [], [], [], [], [], []
    for i in range(len(StudentAnswer)):
        gap1, gap2, gap3, gap4, gap5, gap6 = [], [], [], [], [], []
        for phrase in StudentAnswer[i]:
            WEExpectExprimEtoile, WEExpectExprimMoins, WEEcartInfoEtoile, WEEcartInfoMoins, WEGapPndEtoile, WEGapPndMoins = GapQuestion.GapCalculWE(
                phrase, ReferenceAnswer[i], "", glob.DictioWE)  # questions[i]

            gap1.append(WEExpectExprimEtoile)
            gap2.append(WEExpectExprimMoins)

            gap3.append(WEEcartInfoEtoile)
            gap4.append(WEEcartInfoMoins)

            gap5.append(WEGapPndEtoile)
            gap6.append(WEGapPndMoins)

        ExpectExprimEtoileWE.append(gap1)
        ExpectExprimMoinsWE.append(gap2)
        EcartInfoEtoileWE.append(gap3)
        EcartInfoMoinsWE.append(gap4)
        GapPndEtoileWE.append(gap5)
        GapPndMoinsWE.append(gap6)

        print(gap1, gap2, gap3, gap4, gap5, gap6)

    ############### Loading trained model
    model = pickle.load(open(glob.pathModelMLEnglish, "rb"))

    Col_X = [AllQuestionSimSTS[0][0], AllQuestionSimJacc[0][0], AllQuestionSimDice[0][0], AllQuestionSimJaro[0][0],AllQuestionSimCosinusWEPos[0][0],AllQuestionSimCosinusWE[0][0],AllSimCosinusMinMaxWE[0][0],
                AllQuestionSimES[0][0],
                AllQuestionSimPos[0][0],
                ESExpectExprimEtoile[0][0],
                ESExpectExprimMoins[0][0],
                ESEcartInfoEtoile[0][0],
                ESEcartInfoMoins[0][0],
                ESGapPndMoins[0][0],
                ESGapPndEtoile[0][0],
                GapPndMoinsWE[0][0],
                GapPndEtoileWE[0][0],
                ExpectExprimEtoileWE[0][0],
                ExpectExprimMoinsWE[0][0],
                EcartInfoEtoileWE[0][0],
                EcartInfoMoinsWE[0][0],
                DiffulteQuestion,AllQuestionLength[0][0],
                AllQuestionDiffLength[0][0],
                AllQuestionRedondanceFreq[0][0]]

    poly_reg = PolynomialFeatures(degree=2)
    Col_X = np.array([Col_X])
    X = poly_reg.fit_transform(Col_X)

    #####  Grade prediction
    resultat = model.predict(X)
    score=round(resultat[0], 3)

    if score > 5:
        score=5
    elif score <0:
        score=0
    ####### score=score/5   to transform the grade to similarity
    return score

######## Calling the grader function : an example
######## When the plugin is installed the call from the LMS is directed to the cloud and the main() is run

######## The grader can be used and tested in a desktop version :
######## Introduce (question, student answer, reference answer and difficulty) to the getScore function as in the example


StudentAnswer =[['To simulate the behaviour of portions of the desired software product.']]
#StudentAnswer=[['To find problem and errors in a program before it is finalized']]
ReferenceAnswer =['To simulate the behaviour of portions of the desired software product.']
Question=['What is the role of a prototype program in problem solving?']

#Here you must indicate the question level , int(1)=easy , int(2)=middle , int(3)=hard
DifficulteQuestion = int(2)

##### call the scoring function : getScore
grade = getScore(Question,ReferenceAnswer,StudentAnswer,DifficulteQuestion)
print("grade is :", grade, " (Student Answer and Reference Answer are similars at:", grade*100/5, "%)")









