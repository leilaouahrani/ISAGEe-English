#####################Information gap between question and answers################################
from DomainKnowledge import getVectorContextOfSentence
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import codecs as c
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from math import *
from DomainKnowledge import getVecteurSentenceWE
import sys


#this file is dedicated for the calculation of the Gap-deviation-question features
################# sentence features's calculation (S: student answer, R: Reference answer, Q: Question ######################

#this function returns all gap features using semantic space
def GapCalcul(Response, ModelResponse, Question):
    # 'ESExpectExprim*','ESExpectExprim-','ESEcartInfo*','ESEcartInfo-','ESGapPnd-','ESGapPnd*'
    ESExpectExprimEtoile, ESExpectExprimMoins, ESEcartInfoEtoile, ESEcartInfoMoins, ESGapPndEtoile, ESGapPndMoins = 0, 0, 0, 0, 0, 0
     
    Response = getVectorContextOfSentence(Response)                    # S: Student answer
    ModelResponse = getVectorContextOfSentence(ModelResponse)          # R: Reference answer (model answer)
    Question = getVectorContextOfSentence(Question)                    # Q: Question
                                                                       # "⊙" : refers to the Hadamard product (component-to-component multiplication of vectors )

    ### Cosine(S⊙Q, R⊙Q)
    ESExpectExprimEtoile = cosine_similarity(np.multiply(np.array(Response), np.array(Question)),
                                         np.multiply(np.array(ModelResponse), np.array(Question)))[0][0]
    ### Cosine(S-Q, R-Q)
    ESExpectExprimMoins = \
    cosine_similarity(abs(np.add(np.dot(Response, -1), Question)), abs(np.add(np.dot(ModelResponse, -1), Question)))[0][0]

    ### Cosine(S⊙(S⊙R), R⊙(S⊙R))
    gap = np.multiply(np.array(Response), np.array(ModelResponse))
    ESEcartInfoEtoile = \
    cosine_similarity(np.multiply(np.array(Response), gap), np.multiply(np.array(ModelResponse), gap))[0][0]

    ### Cosine(S⊙(S-R), R⊙(S-R))
    gap = abs(np.add(np.dot(Response, -1), ModelResponse))
    ESEcartInfoMoins = \
    cosine_similarity(np.multiply(np.array(Response), gap), np.multiply(np.array(ModelResponse), gap))[0][0]

    ### Cosine(S⊙(R⊙Q), R)
    gap = np.multiply(np.array(ModelResponse), np.array(Question))
    ESGapPndEtoile = cosine_similarity(np.multiply(np.array(Response), gap), ModelResponse)[0][0]

    ### Cosine(S⊙(R-Q), R)
    gap = abs(np.add(np.dot(ModelResponse, -1), Question))
    ESGapPndMoins = cosine_similarity(np.multiply(np.array(Response), gap), ModelResponse)[0][0]

    return ESExpectExprimEtoile, ESExpectExprimMoins, ESEcartInfoEtoile, ESEcartInfoMoins, ESGapPndEtoile, ESGapPndMoins

#this function uses (GapCalcul) function to return all gap for set of answers 
def GapForAllResponses(AllQuestionCorpus,ModelResponses, Questions):
    All1 = []

    one1 = []

    All2 = []
    one2 = []

    All3 = []
    one3 = []

    All4 = []
    one4 = []

    All5 = []
    one5 = []

    All6 = []
    one6 = []

    for i, responses in enumerate(AllQuestionCorpus):
        for response in responses:
            ESExpectExprimEtoile, ESExpectExprimMoins, ESEcartInfoEtoile, ESEcartInfoMoins, ESGapPndEtoile, ESGapPndMoins = GapCalcul(
                response, ModelResponses[i], Questions[i])
            one1.append(ESExpectExprimEtoile)
            one2.append(ESExpectExprimMoins)
            one3.append(ESEcartInfoEtoile)
            one4.append(ESEcartInfoMoins)
            one5.append(ESGapPndEtoile)
            one6.append(ESGapPndMoins)

        All1.append(one1)
        All2.append(one2)
        All3.append(one3)
        All4.append(one4)
        All5.append(one5)
        All6.append(one6)
        one1 = []
        one2 = []
        one3 = []
        one4 = []
        one5 = []
        one6 = []

    return All1, All2, All3, All4, All5, All6

### this function returns all gap features using WE
def GapCalculWE(Response, ModelResponse, Question, dictWE):
    # 'WEExpectExprim*','WEExpectExprim-','WEEcartInfo*','WEEcartInfo-','WEGapPnd-','WEGapPnd*'
    WEExpectExprimEtoile, WEExpectExprimMoins, WEEcartInfoEtoile, WEEcartInfoMoins, WEGapPndEtoile, WEGapPndMoins = 0, 0, 0, 0, 0, 0
    Response = getVecteurSentenceWE(Response, dictWE)
    ModelResponse = getVecteurSentenceWE(ModelResponse, dictWE)
    Question = getVecteurSentenceWE(Question, dictWE)

    ### Cosine(S⊙Q, R⊙Q)
    WEExpectExprimEtoile = \
    cosine_similarity(np.multiply(np.array(Response).reshape(1, -1), np.array(Question).reshape(1, -1)),
                      np.multiply(np.array(ModelResponse).reshape(1, -1), np.array(Question).reshape(1, -1)))[0][0]

    ### Cosine(S-Q, R-Q)
    WEExpectExprimMoins = cosine_similarity(abs(np.add(np.dot(Response.reshape(1, -1), -1), Question.reshape(1, -1))),
                                            abs(np.add(np.dot(ModelResponse.reshape(1, -1), -1),
                                                       Question.reshape(1, -1))))[0][0]
    ### Cosine(S⊙(S⊙R), R⊙(S⊙R))
    gap = np.multiply(np.array(Response.reshape(1, -1)), np.array(ModelResponse.reshape(1, -1)))
    WEEcartInfoEtoile = cosine_similarity(np.multiply(np.array(Response.reshape(1, -1)), gap),
                                          np.multiply(np.array(ModelResponse.reshape(1, -1)), gap))[0][0]
    ### Cosine(S⊙(S-R), R⊙(S-R))
    gap = abs(np.add(np.dot(Response.reshape(1, -1), -1), ModelResponse.reshape(1, -1)))
    WEEcartInfoMoins = cosine_similarity(np.multiply(np.array(Response.reshape(1, -1)), gap),
                                         np.multiply(np.array(ModelResponse.reshape(1, -1)), gap))[0][0]
    ### Cosine(S⊙(R⊙Q), R)
    gap = np.multiply(np.array(ModelResponse.reshape(1, -1)), np.array(Question.reshape(1, -1)))
    WEGapPndEtoile = \
    cosine_similarity(np.multiply(np.array(Response.reshape(1, -1)), gap), ModelResponse.reshape(1, -1))[0][
        0]  # np.multiply(np.array(ModelResponse.reshape(1,-1)), gap)
    ### Cosine(S⊙(R-Q), R)
    gap = abs(np.add(np.dot(ModelResponse.reshape(1, -1), -1), Question.reshape(1, -1)))
    WEGapPndMoins = \
    cosine_similarity(np.multiply(np.array(Response.reshape(1, -1)), gap), ModelResponse.reshape(1, -1))[0][0]
    return WEExpectExprimEtoile, WEExpectExprimMoins, WEEcartInfoEtoile, WEEcartInfoMoins, WEGapPndEtoile, WEGapPndMoins