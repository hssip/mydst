# -*-coding:utf-8 -*-

import numpy as np 
import json, pickle
from gensim import models
from io import BytesIO
from zipfile import ZipFile
from collections import OrderedDict 

# done_slots = {
#     'taxi':['leaveAt', 'destination', 'departure', 'arriveBy'],
#     'hotel':['name','type', 'parking', 'pricerange', 'internet', 
#                 'day', 'stay', 'people', 'area', 'stars'],
#     'attraction':['name', 'type', 'area'],
#     'train':['people', 'leaveAt', 'destination', 'day','arriveBy', 'departure'],
#     'restaurant' :['food', 'pricerange', 'area', 'name', 'time', 'day', 'people']
# }
domin_list = ['hotel', 'restaurant', 'taxi', 'attraction', 'train']
slots_list = [['name','type', 'parking', 'pricerange', 'internet', 'day', 'stay', 'people', 'area', 'stars'],
                ['food', 'pricerange', 'area', 'name', 'time', 'day', 'people'],
                ['leaveAt', 'destination', 'departure', 'arriveBy'], 
                ['name', 'type', 'area'],
                ['people', 'leaveAt', 'destination', 'day','arriveBy', 'departure']
                ]


GATE_INDEX = ['UPDATE', 'DONTCARE', 'NONE', 'DELETE']

# gate2index = {
#     'UPDATE':   [1,0,0,0],
#     'DONTCARE': [0,1,0,0],
#     'NONE':     [0,0,1,0],
#     'DELETE':   [0,0,0,1]
# }

gate2index = {
    'UPDATE':   [1],
    'DONTCARE': [2],
    'NONE':     [3],
    'DELETE':   [4]
}

special_slot_value={
    'dontcare':['any', 'does not care'],
    'none':['not men', 'not mentioned', 'fun', 'art', 'not', 
            'not mendtioned']
}

def dialogs2tokens(dialogs, max_sentence_length):
    user_diag_list = dialogs['user_turns']
    sys_diag_list = dialogs['sys_turns']
    diag_tokens = []
    for user_diag, sys_diag in zip(user_diag_list, sys_diag_list):
        user_tokens = user_diag.strip('\n').split()
        sys_tokens = sys_diag.strip('\n').split()
        i=0
        turn_tokens = []
        for token in user_tokens:
            i+=1
            turn_tokens.append(token)
            if i > max_sentence_length:
                break
        for j in range(i, max_sentence_length):
            turn_tokens.append('None')
        diag_tokens.append(turn_tokens)

        turn_tokens = []
        i=0
        for token in sys_tokens:
            turn_tokens.append(token)
            i+=1
            if i > max_sentence_length:
                break
        for j in range(i, max_sentence_length):
            turn_tokens.append('None')
        diag_tokens.append(turn_tokens)

    return diag_tokens 

def get_turn_tokens(turn_number,
                hist_turn_length, 
                max_sentence_length,  
                dia_token_list,
                if_complete_turns = True):

    all_tokens = []

    #assert legal
    if turn_number < 0:
        raise RuntimeError('turn_nunmber connot be negtive')
    elif len(dia_token_list) % 2 != 0:
        raise RuntimeError('dia_token_list length wrong')
    
    if turn_number < hist_turn_length:
        for i in range(hist_turn_length - turn_number):
            for j in range(max_sentence_length):
                all_tokens.append(0)
        for i in range(turn_number + 1):
            all_tokens.extend(dia_token_list[i])
            all_tokens.extend(dia_token_list[i + 1])
    else:
        for i in range(turn_number - hist_turn_length, turn_number):
            all_tokens.extend(dia_token_list[i])
            all_tokens.extend(dia_token_list[i + 1])
    
    all_tokens.extend(dia_token_list[turn_number])
    all_tokens.extend(dia_token_list[turn_number + 1])

    return all_tokens




def uttr_token2index(tokens, word_dict):
    tokindx = []
    for token in tokens:
        if token in word_dict:
            tokindx.append(word_dict[token])
        else:
            tokindx.append(0)
    
    return np.array(tokindx).astype('int64')

def slots_attr2index(word_dict):
    slotsindex = []
    for d_index, domin in enumerate(domin_list):
        for slot in slots_list[d_index]:
            if slot in word_dict:
                slotsindex.append(word_dict[slot])
            else:
                slotsindex.append(0)
    
    return np.array(slotsindex).astype('int64')



# def slots2embed(slots, w):
#     slots_embedding = []
#     for domin, value in slots.items():
#         try:
#             slots_embedding.append(w[domin] + w[value])
#         except KeyError:
#             raise 'domin-slot pairs:' + domin + '-'+ value + 'not find in embed_matric'
#     return np.array(slots_embedding)

def slots2gates(slots1, slots2):
    gates = []
    # print(slots1)
    # print(slots2)
    for domin, domin_value in slots1.items():
        for slot, slot_value in domin_value.items():
            #none
            if slots1[domin][slot] == slots2[domin][slot]:
                gates.append(gate2index['NONE'])
            #update
            elif slots2[domin][slot] not in special_slot_value['dontcare'] and \
                slots2[domin][slot] not in special_slot_value['none']:
                gates.append(gate2index['UPDATE'])
            #dontcare
            elif slots2[domin][slot] in special_slot_value['dontcare']:
                gates.append(gate2index['DONTCARE'])
            #delete
            elif slots2[domin][slot] in special_slot_value['none']:
                gates.append(gate2index['DELETE'])
            else:
                gates.append(gate2index['NONE'])

    return np.array(gates).astype('int32')

def get_initial_slots():
    initial_slots = OrderedDict()
    for d_index, domin in enumerate(domin_list):
        temp = OrderedDict()
        for slot in slots_list[d_index]:
            temp[slot] = ''
        initial_slots[domin] = temp

    return initial_slots

def load_all_slot():
    result = []
    for d_index, domin in enumerate(domin_list):
        for slot in slots_list[d_index]:
            result.append(slot)
    
    return result