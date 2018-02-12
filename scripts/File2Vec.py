#coding: utf-8
#author: Tingyan Xiang
#In this secton, each slip opinion file is converted to a vector of features (n-gram)
#Files are converted to feature matrix

import numpy as np
import pandas as pd
import string
import math
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import text
from string import punctuation
from collections import Counter
from nltk.corpus import stopwords

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import re

#tokenize each opinion file into a list; just keep useful words
def cleanData(data):
    '''tokenize each tweet into a list; just keep useful words;
    return a list in which each element denotes the list of each tweet'''
    table = str.maketrans('', '', punctuation)
    #add and delete words from stop_words
    stop_words = stopwords.words('english')[:]
    #stop_words.extend(["#ff", "ff", "rt"])
    stop_words.remove('not')
    stop_words.remove('no')
    lemma = nltk.wordnet.WordNetLemmatizer()
    contractions = {}
    
    tokens = data.split()
    cdata = []
    for o in tokens:
        # change words to lowercase
        o = o.lower()
        # remove punctuation from each token
        o = o.translate(table)  
        # filter out short tokens 
        if len(o) < 2: continue
        # remove remaining tokens that are not alphabetic
        if not o.isalpha(): continue
        # filter out stop words
        if o in stop_words: continue
        o = lemma.lemmatize(o)
        cdata.append(o)  
    return cdata

def f_ngram(data, mode='tfidf', binary=1, ngram=(1,1), min_c=1):
    '''exact n-gram feacturs
    return: feature array (fgram); feature vocabulary (vocab)
    input: data; ngram = (n,n) denote n_gram and ngram=(1,2) denote
    1_gram and 2_gram; tokens with count below min_c are cut off. 
    '''
    if mode == 'tfidf':
        if binary==1:
            gram = text.TfidfVectorizer(ngram_range=ngram, binary=True, min_df=min_c)
        else:
            gram = text.TfidfVectorizer(ngram_range=ngram, min_df=min_c)
        
    else: #mode=count
        if binary==1:
            gram = text.CountVectorizer(ngram_range=ngram, binary=True, min_df=min_c)
        else:
            gram = text.CountVectorizer(ngram_range=ngram, min_df=min_c)
    gram = gram.fit(data)
    vocab = gram.get_feature_names()
    fgram = gram.transform(data).toarray()
    return (fgram, vocab)

def dataFilter(data):
    '''clean data
    return a list, each element denote the string of each file'''
    lines = []
    for element in data:
        tk = ' '.join(element)
        lines.append(tk)
    return lines


path = 'NY-Appellate-Scraping/2017-09-10/courtdoc/txt/'
filelist = os.listdir(path)
clean_data = []
for i in filelist:
    with open(path + i, 'r') as f:
        data = f.readlines()
        data = (' ').join(data) 
        data = data.replace('\n', ' ')
        cdata = cleanData(data)
        clean_data.append(cdata)

clean_data1 = dataFilter(clean_data)

features_data, vocab = f_ngram(clean_data1, mode='tfidf', binary=0, ngram=(1,1), min_c=5)

features_data.shape

vocab = pd.Series(vocab)





