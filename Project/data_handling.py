import io
import os
import nltk
import time
import sklearn
import zipfile
import requests
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
nltk.download('stopwords')
nltk.download('wordnet')


def read_data(path: str) -> Tuple[List[int], List[str]]:
    data = pd.read_csv(path)
    ids = data["id"].tolist()
    paragraphs = data["paragraph"].tolist()
    return ids, paragraphs

def read_questions(path: str) -> Tuple[List[int], List[str], List[int], List[str]]:
    data = pd.read_csv(path)
    ids = data["id"].tolist()
    questions = data["question"].tolist()
    if "test" not in path:
        paragraph_ids = data["paragraph_id"].tolist()
        answers = data["answer"].tolist()
        return ids, questions, paragraph_ids, answers
    else:
        return ids, questions

def save_to_csv(path: str, corpus):
    df = pd.DataFrame(corpus, columns= list(corpus.keys())).head()
    df.to_csv (os.path.join(output_path, path), index = False, header=True)
    
class Preprocess(object):
    def __init__(self, lemmatize=True):
        self.stopwords = set(nltk.corpus.stopwords.words("english"))
        self.lemmatize = lemmatize

    def preprocess_pipeline(self, data):
        clean_tokenized_data = self._clean_doc(data)
        if self.lemmatize:
            clean_tokenized_data = self._lemmatize(clean_tokenized_data)

        return clean_tokenized_data

    def _clean_doc(self, data):
        tokenizer = nltk.tokenize.RegexpTokenizer(r"\w+")
        return [
            [
                token.lower()
                for token in tokenizer.tokenize(review)
                if token.lower() not in self.stopwords
                and len(token) > 1
                and token.isalpha()
            ]
            for review in data
        ]

    def _lemmatize(self, data):
        lemmatizer = nltk.stem.WordNetLemmatizer()
        return [[lemmatizer.lemmatize(word) for word in review] for review in data]

    def convert_to_reviews(self, tokenized_reviews):
        reviews = []
        for tokens in tokenized_reviews:
            reviews.append(" ".join(tokens))

        return reviews
