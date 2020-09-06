import os
from typing import List, Literal, Tuple
import pandas as pd
import re

data_path = "data"
output_path = "output"

def read_data(path: str) -> Tuple[List[str], List[bool], List[Literal["M", "W"]]]:
    data = pd.read_csv(path)
    inputs = data["response_text"].tolist()
    labels = (data["sentiment"] == "Positive").tolist()
    gender = data["op_gender"].tolist()
    return inputs, labels, gender

def read_data_all(path: str) -> Tuple[List[str], List[str], List[bool], List[Literal["M", "W"]]]:
    data = pd.read_csv(path)
    source = data["source"].tolist()
    inputs = data["response_text"].tolist()
    labels = (data["sentiment"] == "Positive").tolist()
    gender = data["op_gender"].tolist()
    return source, inputs, labels, gender


def corpus_sentences_association(data):
    
    #check data is not empty and lists inside data have the same length
    if (not data) or [len(element) for element in data if len(element) != len(data[0])]:
        raise Exception("Sorry, no numbers below zero")
    vocabulary = {}
    for i, item in enumerate(data[1]):
        #if key is not in dict add it and map it empty list
        #and append value to list mapped to key
        val = vocabulary.setdefault(data[0][i], [])
        sentences = item.split(".")
        sentences_list = [sentence + "." for sentence in sentences if sentence != ""]
        if val:
            val.extend(sentences_list)
            vocabulary[data[0][i]] = val
        else:
            vocabulary[data[0][i]] = sentences_list
    
    return vocabulary
    #cre

def write_to_csv(sentences, corpus_name):
    dt = pd.DataFrame(sentences, columns =['sentences'])
    dt.to_csv(corpus_name + '.csv')

def corpus_to_csv(corpus_dict):
    for corpus in corpus_dict.keys():
        write_to_csv(corpus_dict[corpus], corpus)


def normalise_corpus(corpus):
    normalise_corpus = []
    for sentence in corpus:
        transformed = re.sub(r'\b(?:not|never|no)\b[\w\s]+[^\w\s]', lambda match: re.sub(r'(\s+)(\w+)', r'\1\2_NEG', match.group(0)), sentence, flags=re.IGNORECASE)
        normalise_corpus.append(transformed)
    
    return normalise_corpus

def segment_sentence(sentence):
    return re.split(r'[\s]+[\t]*|[\s]*[\t]+', sentence)
    #return re.split(' \n| \t|\n |\t |\t|\n| ', sentence)

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer() 
lemmatizer.lemmatize("rocks")

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")
word1 = ["Visitors", "from", "all", "over", "the", "world", "fishes", "during", "the", "summer","."]
[ stemmer.stem(w) for w in word1]

from nltk.stem.porter import *
stemmer = PorterStemmer()
plurals = ['caresses', 'flies', 'dies', 'mules', 'denied',
...            'died', 'agreed', 'owned', 'humbled', 'sized',
...            'meeting', 'stating', 'siezing', 'itemization',
...            'sensational', 'traditional', 'reference', 'colonizer',
...            'plotted']
singles = [stemmer.stem(plural) for plural in plurals]
print(' '.join(singles))

import re
string = "It was never going to work, he thought. He did not play so well, so he had to practice some more. Not foobar !"
transformed = re.sub(r'\b(?:not|never|no)\b[\w\s]+[^\w\s]', 
       lambda match: re.sub(r'(\s+)(\w+)', r'\1\2_NEG', match.group(0)), 
       string,
       flags=re.IGNORECASE)
print(transformed)




""" import re
string = "no one enjoys it."
string = "It was never going to work, he thought. He did not play so well, so he had to practice some more. Not foobar !"
transformed = re.sub(r'\b(?:not|never|no)\b[\w\s]+[^\w\s]', 
       lambda match: re.sub(r'(\s+)(\w+)', r'_NEG\1\2', match.group(0)), 
       string,
       flags=re.IGNORECASE)
print(transformed) """

re.sub(r'\b(?:not|never|no)\b[\w\s]+[^\w\s]')

train_data = read_data_all(os.path.join(data_path, "train.csv"))
dict_corpus = corpus_sentences_association(train_data)

train_data = read_data(os.path.join(data_path, "train.csv"))
test_data = read_data(os.path.join(data_path, "test.csv"))


train_data = ([text.lower() for text in train_data[0]], train_data[1], train_data[2])
test_data = ([text.lower() for text in test_data[0]], test_data[1], train_data[2])


""" def preprocess_corpus(input_file: str, output_file: str) -> None:
    pass

preprocess_corpus(
    os.path.join(data_path, "train.csv"), os.path.join(output_path, "train_norm.csv")
)
preprocess_corpus(
    os.path.join(data_path, "test.csv"), os.path.join(output_path, "test_norm.csv")
) """