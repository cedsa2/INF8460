import torch
from transformers import pipeline

def getParagraph(paragraphIds, corpus_id, corpus_paragraph):
    result = []
    for i, id in enumerate(paragraphIds):
        index = corpus_id.index(id)
        result.append(corpus_paragraph[index])
    return result


def answer_question(nlp, question, context):
    result = nlp(question=question, context=context)
    return result