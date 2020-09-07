import os
from typing import List, Literal, Tuple
import pandas as pd
import re
import nltk
from nltk import tokenize
from nltk.stem import PorterStemmer 
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

def make_sentence(line):
    return tokenize.sent_tokenize(line)


def corpus_to_sentences(data):  
    #check data is not empty and lists inside data have the same length
    if (not data) or [len(element) for element in data if len(element) != len(data[0])]:
        raise ValueError("Data is not valid.")
    vocabulary = {}
    for i, item in enumerate(data[1]):
        #if key is not in dict add it and map it empty list
        #and append value to list mapped to key
        val = vocabulary.setdefault(data[0][i], [])
        sentences = make_sentence(item)
        if val:
            val.extend(sentences)
            vocabulary[data[0][i]] = val
        else:
            vocabulary[data[0][i]] = sentences
    
    return vocabulary

def process_dict_corpus(func, corpus_dict):
    dict_corpus = {}
    for corpus_name in corpus_dict.keys():
        dict_corpus.setdefault(corpus_name, func(corpus_dict[corpus_name]))
    return dict_corpus

def write_to_csv(sentences, corpus_name):
    dt = pd.DataFrame(sentences, columns =['sentences'])
    dt.to_csv(corpus_name + '.csv')

def dict_corpus_to_csv(corpus_dict, name=None):
    if name is None:
        for corpus_name in corpus_dict.keys():
            write_to_csv(corpus_dict[corpus_name], corpus_name)
    else:
        for corpus_name in corpus_dict.keys():
            write_to_csv(corpus_dict[corpus_name], corpus_name + name)

def func_to_csv(func, corpus, corpus_name):
    result = func(corpus)
    write_to_csv(result, corpus_name)


def normalise_corpus(corpus):
    normalised_corpus = []
    for sentence in corpus:
        normalised = re.sub(r'\b(?:not|never|no)\b[\w\s]+[^\w\s]', lambda match: re.sub(r'(\s+)(\w+)', r'\1\2_NEG', match.group(0)), sentence, flags=re.IGNORECASE)
        normalised_corpus.append(normalised)
    return normalised_corpus

def normalise_dict_corpus(corpus_dict):
    dict_corpus = {}
    for corpus_name in corpus_dict.keys():
        dict_corpus.setdefault(corpus_name, normalise_corpus(corpus_dict[corpus_name]))
    return dict_corpus

def normalise_corpus_to_csv(corpus, corpus_name):
    result = normalise_corpus(corpus)
    write_to_csv(result, corpus_name)

def tokenize_sentence(sentence):
    return nltk.word_tokenize(sentence.lower())

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
        if tag[0].lower() in ['a', 'r', 'n', 'v']:
            results.append(lemmatizer.lemmatize(word, tag[0].lower()))
        else:
            results.append(word)
    return results

def lemmatize_corpus(corpus):
    results = []
    for line in corpus:
        lemmas = lemmatisation_sentence(re.split(r'\s]', line))
        results.append(" ".join(lemmas))
    return results

def lemma_corpus_to_csv(corpus, corpus_name):
    result = lemmatize_corpus(corpus)
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
    train_data = read_data_all(os.path.join(data_path, "train.csv"))
    dict_corpus = corpus_to_sentences(train_data)
    dict_corpus_to_csv(dict_corpus, '_phrases')

    normalised_dict_corpus = process_dict_corpus(normalise_corpus, dict_corpus)
    dict_corpus_to_csv(normalised_dict_corpus, '_normalised')

    #tokenized_dict_corpus = process_dict_corpus(tokenize_corpus, normalised_dict_corpus)
    tokenized_dict_corpus = process_dict_corpus(tokenize_corpus, dict_corpus)
    dict_corpus_to_csv(tokenized_dict_corpus, '_mots')

    lemmatized_dict_corpus = process_dict_corpus(lemmatize_corpus, tokenized_dict_corpus)
    dict_corpus_to_csv(lemmatized_dict_corpus, '_lemmes')

    stemmed_dict_corpus = process_dict_corpus(stems_corpus, lemmatized_dict_corpus)
    dict_corpus_to_csv(stemmed_dict_corpus, '_stems')

    removed_stopwords_dict_corpus = process_dict_corpus(remove_stopwords_corpus, stemmed_dict_corpus)
    dict_corpus_to_csv(removed_stopwords_dict_corpus, '_norm')