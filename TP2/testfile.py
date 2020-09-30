

#Ecrire une fonction qui retourne les +/- 5 voisins du mot en parametres

x = ["vous", "ecrire", "une", "fonction", "qui", "retourne", "les", "voisins", "du", "mot"]

def get_neigbours(list_words, index, N):
    max_index =  index + N
    if max_index >= len(list_words):
        max_index = len(list_words) - 1
    min_index = index - N
    if min_index < 0:
        min_index = 0
    result = []
    while(min_index <= max_index):
        if min_index != index:
            result.append(list_words[min_index])
        min_index += 1
    return result

def get_neigbours_d(list_words, index, N):
    max_index =  index + N
    if max_index >= len(list_words):
        max_index = len(list_words) - 1
    min_index = index - N
    if min_index < 0:
        min_index = 0
    result_dist = []
    while(min_index <= max_index):
        if min_index != index:
            result_dist.append((list_words[min_index], 1 / (min_index - index)))
        min_index += 1
    return result_dist


print(get_neigbours(x, 7, 5))


def neighbours_grapper(list_words, word, N):
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
        result.append(get_neigbours(list_words, i, N))
    #retourner le counter de chaque mot?
    return result

    
    
    