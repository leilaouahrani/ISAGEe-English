##### Text Similarities
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
 
import codecs as c
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from math import *
import linecache as cc
from collections import Counter
from nltk.stem import SnowballStemmer, PorterStemmer
import string
from nltk.corpus import stopwords
import sys

####### 1. STS: String Textual Similarity (STS) using NLCS, NMCLSn, NMCLCS1 and Common Word Order similarity weighting  ################
def NMCLCSn(S, T):     # Normalized Maximal Consecutive Longest Common Subsequence starting at character n, NMCLSn
    m = len(S)
    n = len(T)
    matrice = [[0] * (n + 1) for x in range(m + 1)]             # matrice = Word to Word Similarity Matrix
    taille = 0
    MCLCSn_set = set()
    for i in range(m):
        for j in range(n):
            if S[i] == T[j]:
                a = matrice[i][j] + 1
                matrice[i + 1][j + 1] = a
                if a > taille:
                    MCLCSn_set = set()
                    taille = a
                    MCLCSn_set.add(S[i - a + 1:i + 1])

                elif a == taille:
                    MCLCSn_set.add(S[i - a + 1:i + 1])

    return pow(taille, 2) / (m * n)


def NMCLCS1(S, T):          # Normalized Maximal Consecutive Longest Common Subsequence starting at character 1, NMCLCS1
    a = len(S)
    b = len(T)
    A = list(S)
    B = list(T)
    M = []
    c = 0
    if a >= b:
        for i in range(a):
            for j in range(b):
                if A[i] == B[j]:
                    m = A[i]
                    M.append(m)
                    c = c + 1
                    i = i + 1
                else:
                    return pow(c, 2) / (a * b)
    elif b > a:
        for i in range(b):
            for j in range(a):
                if B[i] == A[j]:
                    m = B[i]
                    M.append(m)
                    c = c + 1
                    i = i + 1
                else:
                    return pow(c, 2) / (a * b)
    return pow(c, 2) / (a * b)


def Matrice(l1, l2):                    # matrice = Word to Word Similarity Matrix
    p1 = l1.split()                     # sentence 1 with matching tokens
    p2 = l2.split()                     # sentence 2 with matching tokens
    m = len(p1)
    n = len(p2)
    k = list(p1)
    g = list(p2)
    listeMax = []
    a1 = []
    a2 = []
    s = 0
    for i in range(len(p1)):
        if (len(p1) < i):
            break
        for j in range(len(p2)):
            if p1[i] == p2[j]:
                a1.append(p1[i])
                s = s + 1
                del p2[j]
                break

    for i in range(len(g)):
        if (len(g) < i):
            break
        for j in range(len(k)):
            if g[i] == k[j]:
                a2.append(g[i])
                del k[j]
                break

    score = np.zeros(shape=(len(k), len(p2)))  # remplir matrice avec zÃ©ros
    for i in range(len(k)):
        for j in range(len(p2)):
            score[i][j] = ((1 / 3) * NLCS(k[i], p2[j])) + ((1 / 3) * NMCLCS1(k[i], p2[j])) + (
                    (1 / 3) * NMCLCSn(k[i], p2[j]))

    if m - s == 0:

        d = list(a1)
        for i in range(len(a1)):
            a1[i] = i                   # sentence 1 transformed into numbers (positions)
        for i in range(len(d)):
            for j in range(len(a2)):
                if d[i] == a2[j]:
                    a2[j] = i
        if s > 0:                    # If there are no words that match, we don't need Common Order Similarity calculation
            f = [abs(a2_elt - a1_elt) for a2_elt, a1_elt in zip(a2,
                                                                a1)]
            # Commun Word Order Similarity
            somme = sum(f)
            if s > 1 and (s % 2 != 0):
                so = 1 - ((2 * somme) / (pow(s, 2) - 1))
            elif s == 1 and (s % 2 != 0):
                so = 1
            elif s % 2 == 0 and s > 0:
                so = 1 - ((2 * somme) / pow(s, 2))
        elif s == 0:
            so = 0

        wf = 0.1                   #  Common Word Order similarity weighting (can be modified)
        sommeMax = 0

        sim = (((s * (1 - wf + wf * so)) + sommeMax) * (m + n)) / (2 * (m * n))

    elif m - s != 0:

        mm = m - s
        nn = n - s
        while (((m - s - len(listeMax)) > 0) and np.sum(score) != 0):
            MaxValue = np.max(score)
            ligne, colonne = 0, 0
            find = False
            for xx in range(mm):
                for k in range(nn):
                    if score[xx][k] == MaxValue:
                        ligne = xx          # matrix maximum row
                        colonne = k         # matrix maximum column
                        find = True
                        break
                if find == True:
                    break
            score = np.delete(score, ligne, axis=0)
            score = np.delete(score, colonne, axis=1)
            mm -= 1
            nn -= 1
            listeMax.append(MaxValue)

            m = m - 1
        sommeMax = sum(listeMax)
        d = list(a1)

        for i in range(len(a1)):
            a1[i] = i
        for i in range(len(d)):
            for j in range(len(a2)):
                if d[i] == a2[j]:
                    a2[j] = i
        if s > 0:
            f = [abs(a2_elt - a1_elt) for a2_elt, a1_elt in zip(a2,
                                                                a1)]
            somme = sum(f)
            if s > 1 and (s % 2 != 0):
                so = 1 - ((2 * somme) / (pow(s, 2) - 1))
            elif s == 1 and (s % 2 != 0):
                so = 1
            elif s % 2 == 0 and s > 0:
                so = 1 - ((2 * somme) / pow(s, 2))
        elif s == 0:
            so = 0

        wf = 0.1
        m = len(p1)

        sim = (((s * (1 - wf + wf * so)) + sommeMax) * (m + n)) / (
                2 * (m * n))
    return sim


def sts(l1, l2):
    p1 = l1.split()
    p2 = l2.split()
    m = len(p1)
    n = len(p2)
    if (m <= n and m != 0):
        a = Matrice(l1, l2)


    elif (m > n and n != 0):
        a = Matrice(l2, l1)
    elif (m == 0 or n == 0):
        a = 0

    return a

def SimStsForQuestion(ResponsesCorpus,ModelResponse):
    sim = []
    for response in ResponsesCorpus:
        sim.append(sts(response, ModelResponse))
    return sim
######  STS  end


############## Calcul NLCS ################
def NLCS(s1, s2):
    if (len(s1) != 0 and len(s2) != 0):
        matrix = [[0 for x in range(len(s2))] for x in range(len(s1))]
        cs = ""
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    if i == 0 or j == 0:
                        matrix[i][j] = 1
                        cs += s1[i]
                    else:
                        matrix[i][j] = matrix[i - 1][j - 1] + 1
                        cs += s1[i]
                else:
                    if i == 0 or j == 0:
                        matrix[i][j] = 0
                    else:
                        matrix[i][j] = max(matrix[i - 1][j], matrix[i][j - 1])

        return pow(matrix[len(s1) - 1][len(s2) - 1], 2) / (len(s1) * len(s2))
    else:

        return 0

def SimNLCSForQuestion(ResponsesCorpus,ModelResponse):
    sim = []
    for response in ResponsesCorpus:
        sim.append(NLCS(response, ModelResponse))
    return sim


###############  LCS similarity  #############
def LCS(s1, s2):
    if (len(s1) != 0 and len(s2) != 0):
        matrix = [[0 for x in range(len(s2))] for x in range(len(s1))]
        cs = ""
        for i in range(len(s1)):
            for j in range(len(s2)):
                if s1[i] == s2[j]:
                    if i == 0 or j == 0:
                        matrix[i][j] = 1
                        cs += s1[i]
                    else:
                        matrix[i][j] = matrix[i - 1][j - 1] + 1
                        cs += s1[i]
                else:
                    if i == 0 or j == 0:
                        matrix[i][j] = 0
                    else:
                        matrix[i][j] = max(matrix[i - 1][j], matrix[i][j - 1])

        return matrix[len(s1) - 1][len(s2) - 1] / max(len(s1), len(s2))
    else:

        return 0

def SimLCSForQuestion(ResponsesCorpus,ModelResponse):
    sim = []
    for response in ResponsesCorpus:
        sim.append(LCS(response, ModelResponse))
    return sim




################ DICE Similarity #################
def dice(a, b):
    if not len(a) or not len(b): return 0.0
    if len(a) == 1:  a = a + u'.'
    if len(b) == 1:  b = b + u'.'

    a_bigram_list = []
    for i in range(len(a) - 1):
        a_bigram_list.append(a[i:i + 2])
    b_bigram_list = []
    for i in range(len(b) - 1):
        b_bigram_list.append(b[i:i + 2])

    a_bigrams = set(a_bigram_list)
    b_bigrams = set(b_bigram_list)
    overlap = len(a_bigrams & b_bigrams)
    dice_coeff = overlap * 2.0 / (len(a_bigrams) + len(b_bigrams))
    return dice_coeff


def SimDiceForQuestion(ResponsesCorpus,ModelResponse):
    sim = []
    for response in ResponsesCorpus:
        sim.append(dice(response, ModelResponse))
    return sim

######################################

##############  JARO Similarity ##########
def jaro(s, t):
    s_len = len(s)
    t_len = len(t)

    if s_len == 0 and t_len == 0:
        return 1

    match_distance = (max(s_len, t_len) // 2) - 1

    s_matches = [False] * s_len
    t_matches = [False] * t_len

    matches = 0
    transpositions = 0

    for i in range(s_len):
        start = max(0, i - match_distance)
        end = min(i + match_distance + 1, t_len)

        for j in range(start, end):
            if t_matches[j]:
                continue
            if s[i] != t[j]:
                continue
            s_matches[i] = True
            t_matches[j] = True
            matches += 1
            break

    if matches == 0:
        return 0

    k = 0
    for i in range(s_len):
        if not s_matches[i]:
            continue
        while not t_matches[k]:
            k += 1
        if s[i] != t[k]:
            transpositions += 1
        k += 1

    return ((matches / s_len) +
            (matches / t_len) +
            ((matches - transpositions / 2) / matches)) / 3

def SimJaroForQuestion(ResponsesCorpus,ModelResponse):
    sim = []
    for response in ResponsesCorpus:
        sim.append(jaro(response, ModelResponse))
    return sim

######################


####################### Jaccard Similarity ##############

def jaccard(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / float(union_cardinality)

def SimJaccardForQuestion(ResponsesCorpus,ModelResponse):
    sim = []
    for response in ResponsesCorpus:
        sim.append(jaccard(response, ModelResponse))
    return sim

##########  Order Similarity ########

def SimilarityOrder(l1, l2):
    p1 = l1.split()  # phrase 1 avec tockens qui matchent
    p2 = l2.split()  # phrase 2 avec tockens qui matchent
    m = len(p1)
    n = len(p2)
    k = list(p1)
    g = list(p2)
    listeMax = []
    a1 = []
    a2 = []
    s = 0

    for i in range(len(p1)):
        if (len(p1) < i):
            break
        for j in range(len(p2)):
            if p1[i] == p2[j]:
                a1.append(p1[i])
                s = s + 1
                del p2[j]
                break
    #print(a1)
    for i in range(len(g)):
        if (len(g) < i):
            break
        for j in range(len(k)):
            if g[i] == k[j]:
                a2.append(g[i])
                del k[j]
                break

    #print(a2)
    #print('nobmre de match', s)
    d = list(a1)
    for i in range(len(a1)):
        a1[i] = i  # phrase 1 transformée en numéros
    #print(a1)
    for i in range(len(d)):
        for j in range(len(a2)):
            if d[i] == a2[j]:
                a2[j] = i
    #print(a2)
    if s > 0:  # S'il n'ya pas des mots qui matchent, on n'a pas besoin d'algo de common order
        f = [abs(a2_elt - a1_elt) for a2_elt, a1_elt in
             zip(a2, a1)]  # liste de soustraction des elements des deux liste   abs(): fct qui retourne valeur absolue
        somme = sum(f)
        #print(somme)
        if s > 1 and (s % 2 != 0):
            so = 1 - ((2 * somme) / (pow(s, 2) - 1))
        elif s == 1 and (s % 2 != 0):
            so = 1
        elif s % 2 == 0 and s > 0:
            so = 1 - ((2 * somme) / pow(s, 2))
    elif s == 0:
        so = 0
    return so

def SimOrderForQuestion(ResponsesCorpus,ModelResponse):
    sim=[]
    for response in ResponsesCorpus:
        sim.append(SimilarityOrder(response,ModelResponse))
    return sim