"""
compute different embedding matrices
"""

import io
import os
import nltk
import time
import sklearn
import zipfile
import requests
import numpy as np
import pandas as pd
import multiprocessing
from functools import partial
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from scipy.spatial.distance import euclidean, cosine
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.decomposition import TruncatedSVD

# tfidf

def buildVocab(X) -> object:
  vectorizer = CountVectorizer(min_df=0, lowercase=False)
  vectorizer.fit(X)
  return vectorizer.vocabulary_

def getTfIdfReprentation(data, vectorizer) -> object: 
  data_tfidf = vectorizer.fit_transform(data)
  features = vectorizer.get_feature_names()
  dense = data_tfidf.todense()
  return dense


def get_doc_embedded(X, vocab, embeddings) -> object:
  X_embedded = np.zeros((len(X), len(embeddings)), dtype=float)

  for i, doc in enumerate(X):
    vec = np.zeros((1, len(embeddings)), dtype=float)
    tokens = doc.split() #new_question_tfidf
    cpt = 0
    for word in tokens:
      if(word in vocab):
        cpt += 1
        vec += embeddings[word]
    vec /= cpt
    X_embedded[i] = vec
  return X_embedded

  def getMedian(corpus):
    total_lenght = sorted([len(doc) for doc in corpus])
    return total_lenght[int(len(total_lenght) * 2/3)]

  
def sklearn_svd(df, k):
    svd_model = TruncatedSVD(n_components=k)
    df_r = svd_model.fit_transform(df)
    return  df_r
    
# glove

def read_from_csv(path):
    """ 
    reads a matrix from a csv
    """
    data = pd.read_csv(path)
    data = data.dropna(axis=1,how='all')
    return (data.to_numpy().T).tolist()

def get_lines_gloves(line):
    """ 
    this function takes:
    line: a line from the glove text file (a string)
    returns a tuple (word, embeddings vector)
    """
    values = line.split()
    word = values[0]
    return word, np.asarray(values[1:], dtype=float)

def get_gloves_dict(path = "glove.6B.300d.txt"):
    """ 
    this function takes:
    path: to a  glove text file (a string)
    returns a dict {key=word:Value=embeddings vector}
    """
    with open(path, "r", encoding="UTF-8") as f:
            lines = f.readlines()
    p = multiprocessing.Pool()
    result = p.map(get_lines_gloves, lines)
    p.close()
    p.join()
    p.terminate()
    return dict(result)

def get_plong_doc(doc, embeddings_dict, len_vec_emb):
    """
    this functions takes in:
    doc: a string representing a doc in the corpus ex:'il est'
    embeddings_dict: a dict {key=word:Value=embeddings}
    len_vec_emb: the length of the embedding vector (d)
    return an embedding vector for the doc 
    this result is the mean of the vector embedding of each word
    """
    vectorizer = CountVectorizer()
    temp_ = vectorizer.fit([doc]).vocabulary_
    vec = np.zeros(len_vec_emb, dtype=float)
    for word in temp_.keys():
        vec += (embeddings_dict.get(word, 0) * temp_[word])
    return vec / sum(temp_.values())

def get_plong_corpus(corpus, embeddings_dict):
    """
    his functions takes in:
    corpus: ['je vais' 'il est']a list of strings representing the corpus. each string in the list is document in the corpus
    embeddings_dict: a dict {key=word:Value=embeddings}
    return a list of embedding vector [] each vector is the embedding vector for a doc
    """
    p = multiprocessing.Pool()
    result = p.map(partial(get_plong_doc, embeddings_dict=embeddings_dict, len_vec_emb=len(list(embeddings_dict.items())[0][1])), corpus)
    p.close()
    p.join()
    p.terminate()
    return result