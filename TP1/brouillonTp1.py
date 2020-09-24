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


import nltk
from nltk import tokenize
string = "It was never going to work, he thought. He did not play so well, so he had to practice some more. Not foobar !"
transformed = tokenize.sent_tokenize(string)
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

def read_data_all(path: str) -> Tuple[List[str], List[str], List[bool], List[Literal["M", "W"]]]:
    data = pd.read_csv(path)
    source = data["source"].tolist()
    inputs = data["response_text"].tolist()
    labels = (data["sentiment"] == "Positive").tolist()
    gender = data["op_gender"].tolist()
    return source, inputs, labels, gender


def corpus_to_sentences_O(data):  
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


def normalise_dict_corpus(corpus_dict):
    dict_corpus = {}
    for corpus_name in corpus_dict.keys():
        dict_corpus.setdefault(corpus_name, normalise_corpus(corpus_dict[corpus_name]))
    return dict_corpus

def normalise_corpus_to_csv(corpus, corpus_name):
    result = normalise_corpus(corpus)
    write_to_csv(result, corpus_name)



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
        lemmas = lemmatisation_sentence(re.split(r'\s', line))
        results.append(" ".join(lemmas))
    return results

def lemma_corpus_to_csv(corpus, corpus_name):
    result = lemmatize_corpus(corpus)
    write_to_csv(result, corpus_name)


def stemming_sentence(wordList):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in wordList]

def stems_corpus(corpus):
    results = []
    for line in corpus:
        stems = stemming_sentence(re.split(r'\s', line))
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
        clean_line = remove_stopwords_wordList(re.split(r'\s', line))
        results.append(" ".join(clean_line))
    return results

def remove_stopwords_corpus_to_csv(corpus, corpus_name):
    result = remove_stopwords_corpus(corpus)
    write_to_csv(result, corpus_name)

def normalise_corpus(corpus):
    normalised_corpus = []
    for doc in corpus:
        result = (doc[0], normalise_doc(doc[1]), doc[2])
        normalised_corpus.append(result)
    return normalised_corpus


process_step = [('normalise_doc', 'tokenize_doc', 'lemmatize_doc', 'stems_doc'), ('normalise_doc', 'tokenize_doc', 'lemmatize_doc', 'remove_stopwords_doc'), ('normalise_doc', 'tokenize_doc', 'stems_doc', 'remove_stopwords_doc')]

for step in process_step:
    corpus = process_list_corpus_tup(tokenize_doc, corpus)


normalised_corpus = process_list_corpus_tup(normalise_doc, corpus)
tokenized_corpus = process_list_corpus_tup(tokenize_doc, corpus)
lemmatized_corpus = process_list_corpus_tup(lemmatize_doc, tokenized_corpus)
stemmed_corpus = process_list_corpus_tup(stems_doc, lemmatized_corpus)
removed_stopwords_corpus = process_list_corpus_tup(remove_stopwords_doc, stemmed_corpus)

result = frequence_table_corpus(tokenized_corpus, 10)
write_frequence_to_csv(result)
print(result)


from itertools import combinations
#process_list = ["normalise_doc", "tokenize_doc", "lemmatize_doc", "stems_doc", "remove_stopwords_doc"]
process_list = ["lemmatize_doc", "stems_doc", "remove_stopwords_doc"]
tuple_list = []
for i, _ in enumerate(process_list):
    tuple_list.extend(list(combinations(process_list, i+1)))

print(tuple_list)


print(list(combinations(process_list, 4)))
process_step = [('normalise_doc', 'tokenize_doc', 'lemmatize_doc', 'stems_doc'), ('normalise_doc', 'tokenize_doc', 'lemmatize_doc', 'remove_stopwords_doc'), ('normalise_doc', 'tokenize_doc', 'stems_doc', 'remove_stopwords_doc')]

def combinations_maker(process_list):
    tuple_list = []
    for i, _ in enumerate(process_list):
        if i == 3:
            break
        tuple_list.extend(list(combinations(process_list, i+1)))
    return [item for item in tuple_list if item != (lemmatize_doc, stems_doc) and item !=  (stems_doc, lemmatize_doc)]


def combinations_process_corpus(process_list, tokenized_corpus):
    result = []
    comb_process_list = combinations_maker(process_list)
    for comb in comb_process_list:
        corpus = tokenized_corpus
        if normalise_doc in comb:
            corpus = process_list_corpus_tup(normalise_doc, corpus)
        corpus = process_list_corpus_tup(tokenize_doc, corpus)
        for func in comb:
            if func == normalise_doc:
                continue
            corpus = process_list_corpus_tup(func, corpus)
        result.append((corpus, tuple(i.__name__ for i in comb)))
    return result