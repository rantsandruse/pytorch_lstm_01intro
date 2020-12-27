'''
Util functions to preprocess sequences
'''
import torch
import pandas as pd


def seq_to_embedding(seq, to_ix):
    '''
    This is a good entry point for passing in different kinds of embeddings and
    :param seq: sequence of words
    :param to_ix: embedding lib
    :return:
    '''
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def seqs_to_dictionary(training_data: list):
    '''
    Parameters
    ----------
    training_data: training data as a list of tuples.

    Returns
    -------
    word_to_ix: a dictionary mapping words to indices
    tag_to_ix: a dictionary mapping tags to indices
    '''
    word_to_ix = {}
    tag_to_ix = {}
    count1 = count2 = 0

    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = count1
                count1 += 1
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = count2
                count2 += 1
    return word_to_ix, tag_to_ix

