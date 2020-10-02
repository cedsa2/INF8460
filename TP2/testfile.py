import math
import numpy as np
from collections import Counter
import re
#Ecrire une fonction qui retourne les +/- 5 voisins du mot en parametres

x = ["vous", "ecrire", "une", "fonction", "qui", "retourne", "les", "voisins", "du", "mot"]


def get_neigbours_old(list_words, index, N):
    result = []
    try:
        word = list_words[index]
    except:
        return result
    max_index =  index + N
    if max_index >= len(list_words):
        max_index = len(list_words) - 1
    min_index = index - N
    if min_index < 0:
        min_index = 0
    while(min_index <= max_index):
        if list_words[min_index] != word:
            result.append((word, list_words[min_index]))
        min_index += 1
    return result

def get_neighbours(list_words, index, N, is_distance):
    result = []
    try:
        word = list_words[index]
    except:
        return result
    max_index =  index + N
    if max_index >= len(list_words):
        max_index = len(list_words) - 1
    min_index = index - N
    if min_index < 0:
        min_index = 0
    if is_distance:
        while(min_index <= max_index):
            if list_words[min_index] != word:
                result.append(((word, list_words[min_index]), 1 / (min_index - index)))
            min_index += 1
    else:
        while(min_index <= max_index):
            if list_words[min_index] != word:
                result.append((word, list_words[min_index]))
            min_index += 1
    return result


print(get_neighbours(x, 7, 5, True))
#print(get_neigbours_d(x, 7, 5))


def neighbours_grapper(list_words, word, N, is_distance):
    #if list_words is empty return
    if not list_words:
        return []
    #find all indexes of word
    indexes = [i for i, w in enumerate(list_words) if w == word]
    #if indexes is empty return
    if not indexes:
        return []
    result = []
    for i in indexes:
        result.extend(get_neighbours(list_words, i, N, is_distance))
    return result


def neighbours_corpus(corpus, N, frame_panda, is_distance):
    result = []
    for word in frame_panda:
        for sentence in corpus:
            re.split(r'\s', sentence)
            result.extend(neighbours_grapper(sentence, word, N, is_distance))
    return result


def dict_tuple_creator(tuple_list):
    dict_tuple = Counter(tuple_list)
    return dict(dict_tuple)

test = [(('voisins', 'une'), -0.2), (('voisins', 'fonction'), -0.25), (('voisins', 'qui'), -0.3333333333333333), (('voisins', 'retourne'), -0.5), (('voisins', 'une'), -0.4)]
#Que faire des distances negatives?

def dict_tuple_creator_d(tuple_list):
    res = list(zip(*tuple_list))
    tup = list(res[0])
    dict_tup = Counter(tup)
    print(dict_tup)
    for key in dict_tup.keys():
        temp = [item[1] for item in tuple_list if item[0] == key]
        total = sum(temp)
        dict_tup[key] = total / dict_tup[key]
    return dict(dict_tup)

print(dict_tuple_creator_d(test))

def matrix_creator(dict_tuple, unigramme_list, N):
    # avoir un dict avec key (mot1 , mot2) et frequence comme value
    # creer une matrice 5000 * 5000 et la remplir de zero
    # parcourir le dictionnaire:
    #   pour chaque key (mot1 , mot2): find la position du mot1 et du mot2 dans la matrice
    #                           pour avoir ces coordonnes (pos1, pos2) dans la matrice
    # mettre la value correspondante a cette key a (pos1, pos2) dans la matrice
    list_zero = [0] * N
    mat_result = [list_zero] * N
    for key in dict_tuple.keys():
        #if key[0] == key[1]:
        try:
            x = unigramme_list.index(key[0])
            y = unigramme_list.index(key[1])
        except ValueError:
            continue
        mat_result[x][y] = dict_tuple[key]
    return mat_result

def probability_matrix(mat_Result):
    total = 0
    for line in mat_Result:
        total += sum(line)
    arr = np.array(mat_Result)
    arr_prob = np.divide(arr, total)
    return arr_prob.tolist()

def ppmi_pmi(mat_prob, is_ppmi):
    arr = np.array(mat_prob)
    sum_arr_column = arr.sum(axis=0)
    sum_arr_line = arr.sum(axis=1)
    for i, line in enumerate(mat_prob):
        for j, item in enumerate(line):
            if item == 0:
                mat_prob[i][j] = -100
            elif is_ppmi:         
                mat_prob[i][j] = max(math.log(item / (sum_arr_column[j]*sum_arr_line[i]), 2), 0)
            else:
                mat_prob[i][j] = math.log(item / (sum_arr_column[j]*sum_arr_line[i]), 2)
    return mat_prob


mat_prob = [[0.0, 0.0, 0.05, 0.0, 0.05], [0, 0, 0.05, 0, 0.05], [0.11, 0.05, 0, 0.05, 0.0], [0.05, 0.32, 0, 0.21, 0]]

print(ppmi_pmi(mat_prob, False))

    