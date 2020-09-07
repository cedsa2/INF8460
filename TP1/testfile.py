import os
from typing import List, Literal, Tuple
import pandas as pd
import re
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

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

def func_to_csv(func, corpus, corpus_name):
    result = func(corpus)
    write_to_csv(result, corpus_name)


def normalise_corpus(corpus):
    normalise_corpus = []
    for sentence in corpus:
        transformed = re.sub(r'\b(?:not|never|no)\b[\w\s]+[^\w\s]', lambda match: re.sub(r'(\s+)(\w+)', r'\1\2_NEG', match.group(0)), sentence, flags=re.IGNORECASE)
        normalise_corpus.append(transformed)
    
    return normalise_corpus

def normalise_corpus_to_csv(corpus):
    normalise_corpus(corpus)

def tokenize_sentence(sentence):
    #return re.split(r'[\s]+[\t]*|[\s]*[\t]+', sentence)
    return nltk.word_tokenize(sentence.lower())
    #return re.split(' \n| \t|\n |\t |\t|\n| ', sentence)

def tokenize_corpus(corpus):
    results = []
    for sentence in corpus:
        tokens = tokenize_sentence(sentence)
        results.append(" ".join(tokens))
    return results

def tokenize_corpus_to_csv(corpus, corpus_name):
    result = tokenize_corpus(corpus)
    write_to_csv(result, corpus_name)




def lemmatisation_sentence(wordList):
    lemmatizer = WordNetLemmatizer()
    word_tuples = nltk.pos_tag(wordList)
    results = []
    for word, tag in word_tuples:
        if tag[0].lower() in []:
            results.append(lemmatizer.lemmatize(word, wntag))
        else:
            results.append(word)
    return results

def lemma_corpus(corpus):
    results = []
    for line in corpus:
        lemmas = lemmatisation_sentence(re.split(r'\s]', line))
        results.append(" ".join(lemmas))
    return results

def lemma_corpus_to_csv(corpus, corpus_name):
    result = lemma_corpus(corpus)
    write_to_csv(result, corpus_name)


def stemming_line(wordList):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in wordList]

def stems_corpus(corpus):
    results = []
    for line in corpus:
        stems = stemming_line(line)
        results.append(" ".join(stems))
    return results

def stems_corpus_to_csv(corpus, corpus_name):
    result = stems_corpus(corpus)
    write_to_csv(result, corpus_name)

def remove_stopwords_wordList(wordList):
    stop_words = set(stopwords.words('english'))
    return [word for word in wordList if word not in stop_words]

def remove_stopwords_corpus(corpus):
    results = []
    for line in corpus:
        clean_line = remove_stopwords_wordList(line)
        results.append(" ".join(clean_line))
    return results

def remove_stopwords_corpus_to_csv(corpus, corpus_name):
    result = remove_stopwords_corpus(corpus)
    write_to_csv(result, corpus_name)


def preprocess_corpus(input_file: str, output_file: str)-> None:
    pass



#lemmatizer = WordNetLemmatizer() 
#lemmatizer.lemmatize("rocks")
#lemmatizer.lemmatize("rocks", pos="v")

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