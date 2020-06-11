# -*-coding:utf-8 -*-

import numpy as np 
import json, pickle
from gensim import models
from io import BytesIO
from zipfile import ZipFile
from collections import OrderedDict

# slots = {
#     'taxi':['leaveAt', 'destination', 'departure', 'arrivaBy'],
#     'hotel':['name','type', 'parking', 'pricerange', 'internet', 
#                 'day', 'stay', 'people', 'area', 'stars'],
#     'attraction':['name', 'type', 'area'],
#     'train':['people', 'leaveAt', 'destination', 'day','arriveBy', 'departure'],
#     'restaurant' :['food', 'price', 'area', 'name', 'time', 'day', 'people']
# }
done_slots = {
    'taxi':['leave', 'destination', 'departure', 'arriva'],
    'hotel':['name','type', 'parking', 'price', 'internet', 
                'day', 'stay', 'people', 'area', 'stars'],
    'attraction':['name', 'type', 'area'],
    'train':['people', 'leave', 'destination', 'day','arrive', 'departure'],
    'restaurant' :['food', 'price', 'area', 'name', 'time', 'day', 'people']
}

def get_embedding_dict(EMBEDDIND_FILE_NAME):
    w = models.KeyedVectors.load_word2vec_format(EMBEDDIND_FILE_NAME, binary=True)
    w = models.KeyedVectors.load()
    return w

def load_histr_dia():
    return []

def load_histr_slot():
    return []

def load_all_slot():
    return np.array([])

def sen2vec(sentence):
    return []


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
    for domin, value in slots.items():
        try:
            slots_embedding.append(w[domin] + w[value])
        except KeyError:
            raise 'domin-slot pairs:' + domin + '-'+ value + 'not find in embed_matric'
    return np.array(slots_embedding)

def slots2gates(slots1, slots2):
    gates = []
    for domin, domin_value in slots1.item():
        for slot, slot_value in domin_value:
            pass
    return 

def load_pub_domin():
    return []

