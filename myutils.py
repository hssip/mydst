# -*-coding:utf-8 -*-

import numpy as np 
import json, pickle
from gensim import models
from io import BytesIO
from zipfile import ZipFile
from collections import OrderedDict

slots = {
    'taxi':['name', 'reference', 'leaveAt', 'destination', 'departure', 'arrivaBy'],
    # 'police':['name', 'reference'],
    'hotel':['name', 'reference', 'type', 'parking', 'pricerange', 'internet', 
                'day', 'stay', 'people', 'area', 'stars'],
    'attraction':['name', 'reference', 'type', 'area'],
    'train':['name', 'reference', 'people', 'leaveAt', 'destination', 'day',
                'arriveBy', 'departure', ''],
    'hospital':['name', 'reference', 'daparture']
}

def get_embedding_dict(EMBEDDIND_FILE_NAME):
    w = models.KeyedVectors.load_word2vec_format(EMBEDDIND_FILE_NAME, binary=True)
    return w

def load_histr_dia():
    return []

def load_histr_slot():
    return []

def load_all_slot():
    return np.array([])

def sen2vec(sentence):
    return []

# def 


def dialogs2embedding(dialogs, max_sentence_length, w, WORD_EMBEDDING_LENGTH):
    user_diag_list = dialogs['user_turns']
    sys_diag_list = dialogs['sys_turns']
    diag_embedding = []
    for user_diag, sys_diag in zip(user_diag_list, sys_diag_list):
        user_tokens = user_diag.strip('\n').split()
        sys_tokens = sys_diag.strip('\n').split()
        tokens_embedding = []
        i=0
        for token in user_tokens:
            i+=1
            try:
                token_embedding = w[token]
            except KeyError:
                token_embedding = [0 for i in range(WORD_EMBEDDING_LENGTH)]
            tokens_embedding.append(token_embedding)
        while i < max_sentence_length:
            tokens_embedding.append([0 for i in range(WORD_EMBEDDING_LENGTH)])
        diag_embedding.append(tokens_embedding)
        tokens_embeding = []
        for token in sys_tokens:
            try:
                token_embedding = w[token]
            except KeyError:
                token_embedding = [0 for i in range(WORD_EMBEDDING_LENGTH)]
            tokens_embedding.append(token_embedding)
        while i < max_sentence_length:
            tokens_embedding.append([0 for i in range(WORD_EMBEDDING_LENGTH)])
        diag_embedding.append(tokens_embedding)
    return diag_embedding 

def slots2embed(slots, w):
    slots_embedding = []


def load_pub_domin():
    return []

