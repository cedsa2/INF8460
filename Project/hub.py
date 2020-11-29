import io
import os
import data_handling
from data_handling import read_questions
from data_handling import read_data
from data_handling import Preprocess
from helper import voisins
from typing import Dict, List, Tuple
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from create_embeddings import getTfIdfReprentation, get_gloves_dict, get_plong_corpus
from scipy import spatial
from sent2vec.vectorizer import Vectorizer
from bert import getParagraph
from bert import answer_question

def load_data(name="corpus", force_refresh = 0)-> object:
    data_path = "data"
    output_path = "/content/drive/My Drive/Colab Notebooks/INF8460/Project/output"

    result = ()
    if name == "corpus" :
        result = read_data(os.path.join(data_path, "/content/drive/My Drive/Colab Notebooks/INF8460/Project/data/corpus.csv"))

    elif name == "train":
        result = read_questions(os.path.join(data_path, "/content/drive/My Drive/Colab Notebooks/INF8460/Project/data/train_ids.csv"))

    elif name == "validation":
        result = read_questions(os.path.join(data_path, "/content/drive/My Drive/Colab Notebooks/INF8460/Project/data/val_ids.csv"))

    else:
        print("error name")
    
    return result


def load_data(data, force_refresh = 0)-> object:
    pre = Preprocess()

    data_tokenized = pre.preprocess_pipeline(data)
    data_text = [" ".join(sentence) for sentence in data_tokenized]

    return data_text


def get_ranking_list(paragraphs_id, paragraphs_vectors, questions_id, questions_vectors, metrique, top)-> object:

    dic_paragraphs = {}
    for i, ids in enumerate(paragraphs_id) :
        dic_paragraphs[paragraphs_ids[i]] = paragraphs_vectors[i]


    ranking_list = {}
    for i in range(len(questions_vectors)):

        topk_ids, topk_questions = voisins(questions_vectors[i], dic_paragraphs, top, distfunc=metrique)
        print(topk_ids, topk_questions)

        ranking_list[i] = topk_ids

    return ranking_list



def get_emmbending(paragraph, question, representation_name)-> object:
    paragraphs_vectors = []
    questions_vectors = []

    if representation_name == "tfidf" :
        vectorizer = TfidfVectorizer(max_features=15000) # 
        paragraphs_vectors = getTfIdfReprentation(paragraph, vectorizer)
        questions_vectors = vectorizer.transform(question).todense()

    elif representation_name == "glove":
        vectorizer = CountVectorizer()
        X = vectorizer.fit(paragraph).vocabulary_

        glove_dict = get_gloves_dict()
        key_set = set(X.keys()) & set(glove_dict.keys())
        glove_dict_vocab_corpus = {key: glove_dict[key] for key in key_set}

        paragraphs_vectors = get_plong_corpus(paragraph, glove_dict_vocab_corpus)
        questions_vectors = get_plong_corpus(paragraph, glove_dict_vocab_corpus)


    elif representation_name == "bert":
        vectorizer = Vectorizer()

        vectorizer.bert(paragraph)
        paragraphs_vectors = vectorizer.vectors

        vectorizer.bert(question)
        questions_vectors = vectorizer.vectors

    else:
        print("error representation_name")
    
    return paragraphs_vectors, questions_vectors


def get_answers(model, tokenizer, paragraphs_id, paragraphs, questions_id, questions, ranking_list):

    questions_answers = {}
    
    for question_id in ranking_list:
        question = questions[question_id]
        top_paragraphs_ids = ranking_list[question_id]
        top_paragraphs_text = getParagraph(top_paragraphs_ids, paragraphs_id, paragraphs)

        questions_answers[question] = []
        for j, paragraph in enumerate(top_paragraphs_text):
            answers = answer_question(model, tokenizer, question, paragraph)
            questions_answers[question].append( answers )


    print( question )
    print( questions_answers[question] )

    return questions_answers

    
def voisins(word, df, n, distfunc=cosine):
    assert distfunc.__name__ == 'cosine' or distfunc.__name__ == 'euclidean', "distance metric not supported"
    order = True if distfunc.__name__ == 'euclidean' else False

    closest = {}
    for w in df:
        distance = distfunc(word, df[w])
        closest[w] = distance

    closest = {k: v for k, v in sorted(closest.items(), key=lambda item: item[1], reverse=order)}

    return list(closest.keys())[:n], list(closest.values())[:n]