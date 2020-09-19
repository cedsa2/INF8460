import os
from typing import List, Literal, Tuple
import pandas as pd
import re
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk import tokenize
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import regexp_tokenize
from itertools import combinations

data_path = "data"
output_path = "output"

def read_data(path: str) -> Tuple[List[str], List[bool], List[Literal["M", "W"]]]:
    data = pd.read_csv(path)
    inputs = data["response_text"].tolist()
    labels = (data["sentiment"] == "Positive").tolist()
    gender = data["op_gender"].tolist()
    return inputs, labels, gender


def write_to_csv(sentences, corpus_name):
    dt = pd.DataFrame(sentences, columns =['sentences'])
    dt.to_csv(corpus_name + '.csv')

def write_corpus_to_csv(corpus, corpus_name):
    sentences = []
    for doc in corpus:
        sentences.extend(doc[1])
    write_to_csv(sentences, corpus_name)

def process_list_corpus_tup(func, corpus_tup_list):
    process_corpus = []
    for doc in corpus_tup_list:
        result = (doc[0], func(doc[1]), doc[2])
        process_corpus.append(result)
    return process_corpus



def make_sentence(line):
    sentences = tokenize.sent_tokenize(line)
    return [sentence for sentence in sentences if re.findall(r"[\w]+", sentence)]


def corpus_to_sentences(data):  
    #check data is not empty and lists inside data have the same length
    if (not data) or [len(element) for element in data if len(element) != len(data[0])]:
        raise ValueError("Data is not valid.")
    vocabulary = []
    for i, item in enumerate(data[0]):
        document = (data[2][i], make_sentence(item), data[1][i])
        vocabulary.append(document)
    
    return vocabulary

def normalise_doc(doc):
    normalised_doc = []
    for sentence in doc:
        normalised = re.sub(r'\b(?:not|never|no)\b[\w\s]+[^\w\s]', lambda match: re.sub(r'(\s+)(\w+)', r'\1\2_NEG', match.group(0)), sentence, flags=re.IGNORECASE)
        normalised_doc.append(normalised)
    return normalised_doc


def tokenize_sentence(sentence):
    #return nltk.word_tokenize(sentence.lower())
    return regexp_tokenize(sentence.lower(), "[\w']+") 

def tokenize_doc(doc):
    results = []
    for sentence in doc:
        tokens = tokenize_sentence(sentence)
        results.append(" ".join(tokens))
    return results


def lemmatisation_sentence(wordList):
    lemmatizer = WordNetLemmatizer()
    word_tuples = nltk.pos_tag(wordList)
    results = []
    for word, tag in word_tuples:
        if tag[0].lower() in ['a', 'r', 'n', 'v']:
            if '_NEG' in word:
                word.replace('_NEG', '')
                lemme = lemmatizer.lemmatize(word, tag[0].lower())
                lemme = lemme + '_NEG'
                results.append(lemme)
            else:
                results.append(lemmatizer.lemmatize(word, tag[0].lower()))
        else:
            results.append(word)
    return results

def lemmatize_doc(doc):
    results = []
    for line in doc:
        lemmas = lemmatisation_sentence(re.split(r'\s', line))
        results.append(" ".join(lemmas))
    return results

def stemming_sentence(wordList):
    stemmer = PorterStemmer()
    result = []
    for word in wordList:
        if '_NEG' in word:
            word.replace('_NEG', '')
            stem = stemmer.stem(word)
            stem = stem + '_NEG'
            result.append(stem)
        else:
            result.append(stemmer.stem(word))
    return result

def stems_doc(doc):
    results = []
    for line in doc:
        stems = stemming_sentence(re.split(r'\s', line))
        results.append(" ".join(stems))
    return results

def remove_stopwords_wordList(wordList):
    stop_words = set(stopwords.words('english'))
    return [word for word in wordList if word not in stop_words]

def remove_stopwords_doc(doc):
    results = []
    for line in doc:
        clean_line = remove_stopwords_wordList(re.split(r'\s', line))
        results.append(" ".join(clean_line))
    return results


def preprocess_corpus(input_file: str, output_file: str)-> None:
    matches  = re.findall(r"\w+_norm.csv$", output_file)
    if matches:
        output_file = matches[0].split("_norm.csv")[0]
    
    train_data = read_data(input_file)
    corpus = corpus_to_sentences(train_data)
    write_corpus_to_csv(corpus, output_file + '_phrases')

    normalised_corpus = process_list_corpus_tup(normalise_doc, corpus)
    write_corpus_to_csv(normalised_corpus, output_file + '_normalised')

    #tokenized_corpus = process_dict_corpus(tokenize_doc, normalised_corpus)
    tokenized_corpus = process_list_corpus_tup(tokenize_doc, corpus)
    write_corpus_to_csv(tokenized_corpus, output_file + '_mots')

    lemmatized_corpus = process_list_corpus_tup(lemmatize_doc, tokenized_corpus)
    write_corpus_to_csv(lemmatized_corpus, output_file + '_lemmes')

    stemmed_corpus = process_list_corpus_tup(stems_doc, lemmatized_corpus)
    write_corpus_to_csv(stemmed_corpus, output_file + '_stems')

    removed_stopwords_corpus = process_list_corpus_tup(remove_stopwords_doc, stemmed_corpus)
    write_corpus_to_csv(removed_stopwords_corpus, output_file + '_norm')




#preprocess_corpus(os.path.join(data_path, "train.csv"), "test_norm.csv")

def count_words(doc, vocabulary_dict):
    #vocabulary_dict = {}
    for line in doc:
        words_list = re.split(r'\s', line)
        for word in words_list:
            if word in vocabulary_dict:
                vocabulary_dict[word] += 1
            else:
                vocabulary_dict.setdefault(word, 1)
    return vocabulary_dict

def count_words_corpus(corpus):
    vocabulary_dict = {}
    for doc in corpus:
        vocabulary_dict = count_words(doc[1], vocabulary_dict) 
    return vocabulary_dict


def frequence_table_corpus(tokenized_corpus, N):
    dict_word = count_words_corpus(tokenized_corpus)
    result = sorted(dict_word.items(), key=lambda x: x[1], reverse=True)
    if len(result) > N:
        result = result[0:N-1]
    return result

def write_frequence_to_csv(tup_list):
    result = list(zip(*tup_list))
    rang = range(1, len(result[1]))
    dict_result = {"word" : result[0], "frequence" : result[1], "rang":  rang}
    dt = pd.DataFrame(dict_result)
    dt.to_csv('table_freq.csv')


def combinations_maker(process_list):
    tuple_list = []
    for i, _ in enumerate(process_list):
        if i == 2:
            break
        tuple_list.extend(list(combinations(process_list, i+1)))
    return [item for item in tuple_list if item != (lemmatize_doc, stems_doc) and item !=  (stems_doc, lemmatize_doc)]


def combinations_process_corpus(process_list, tokenized_corpus):
    result = []
    comb_process_list = combinations_maker(process_list)
    for comb in comb_process_list:
        corpus = tokenized_corpus
        for func in comb:
            corpus = process_list_corpus_tup(func, corpus)
        result.append((corpus, tuple(i.__name__ for i in comb)))
    return result


train_data = read_data(os.path.join(data_path, "train.csv"))
corpus = corpus_to_sentences(train_data)
tokenized_corpus = process_list_corpus_tup(tokenize_doc, corpus)
#corpus = tokenized_corpus
#dict_word = count_words_corpus(tokenized_corpus)
#result = sorted(dict_word.items(), key=lambda x: x[1], reverse=True)

#result = frequence_table_corpus(tokenized_corpus, 10)
#write_frequence_to_csv(result)
#print(result)
process_list = [lemmatize_doc, stems_doc, remove_stopwords_doc]
#process_list = ['lemmatize_doc', 'stems_doc', 'remove_stopwords_doc']

results = combinations_process_corpus(process_list, tokenized_corpus)
print(results)
