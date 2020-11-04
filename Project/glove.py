import numpy as np
import pandas as pd
import random
import multiprocessing
import itertools
from functools import partial
import operator
import math
import functools
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_extraction.text import CountVectorizer


def read_from_csv(path):
    """ 
    reads a matrix from a csv
    """
    data = pd.read_csv(path)
    data = data.dropna(axis=1,how='all')
    return (data.to_numpy().T).tolist()


path = "corpus/corpus.csv"
datat = read_from_csv(path)

vectorizer = CountVectorizer()
X = vectorizer.fit(datat[1]).vocabulary_

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

glove_dict = get_gloves_dict()
key_set = set(X.keys()) & set(glove_dict.keys())
glove_dict_vocab_corpus = {key: glove_dict[key] for key in key_set}


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

plongement_doc = get_plong_corpus(datat[1], glove_dict_vocab_corpus)
print('')