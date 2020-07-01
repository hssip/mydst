# -*-coding:utf-8 -*-

import numpy as np 
import json, pickle
from gensim import models
from io import BytesIO
from zipfile import ZipFile
from collections import OrderedDict 
import copy

from predata_2_1 import *

domain_list = ['hotel', 'restaurant', 'taxi', 'attraction', 'train']
slots_list = [['name','type', 'parking', 'pricerange', 'internet', 'day', 'stay', 'people', 'area', 'stars'],
                ['food', 'pricerange', 'area', 'name', 'time', 'day', 'people'],
                ['leaveat', 'destination', 'departure', 'arriveby'], 
                ['name', 'type', 'area'],
                ['people', 'leaveat', 'destination', 'day','arriveby', 'departure']
                ]

GATE_INDEX = ['UPDATE', 'DONTCARE', 'NONE', 'DELETE']

gate2index = {
    'UPDATE':   0,
    'DONTCARE': 1,
    'NONE':     2,
    'DELETE':   3
}

special_slot_value={
    'dontcare':['any', 'does not care', 'dont care'],
    'none':['not men', 'not mentioned', 'fun', 'art', 'not', 
            'not mendtioned', '']
}

def load_slot_value_list():
    slot_value_list = set()
    slot_value_list.add('')
    with open('pub_dataset/MultiWOZ_1.0/ontology.json', 'r') as json_file:
        data = json.load(json_file)
        for index, domain_slot in enumerate(data):
            for value in data[domain_slot]:
                slot_value_list.add(value)
    
    slot_value_list = list(slot_value_list)
    a = slot_value_list.index('')
    slot_value_list[0], slot_value_list[a] = slot_value_list[a], slot_value_list[0]

    return slot_value_list

def uttr_token2index(tokens, word_dict):
    tokindx = [[]]
    leng = len(word_dict)
    for token in tokens:
        a = bytes(token, encoding='utf8')
        if  a in word_dict:
            tokindx[0].append(word_dict[a])
        else:
            word_dict[a] = leng
            tokindx[0].append(leng)

    return tokindx

def slots_attr2index():
    slotsindex = []
    i = 0
    for d_index, domain in enumerate(domain_list):
        for slot in slots_list[d_index]:
            slotsindex.append(i)
            i+=1
    
    return slotsindex

def slots2gates(slots1, slots2): 
    gates = []
    for domain, domain_value in slots1.items():
        for slot, slot_value in domain_value.items():
            #none
            if slots1[domain][slot]== '' and slots2[domain][slot] == '':
                gates.append('NONE')
            #update
            elif slots2[domain][slot] not in special_slot_value['dontcare'] and \
                slots2[domain][slot] not in special_slot_value['none']:
                gates.append('UPDATE')
            #dontcare
            elif slots2[domain][slot] in special_slot_value['dontcare']:
                gates.append('DONTCARE')
            #delete
            elif slots2[domain][slot] in special_slot_value['none'] and \
                slots1[domain][slot] not in special_slot_value['dontcare'] and \
                slots1[domain][slot] not in special_slot_value['none']:
                gates.append('DELETE')
            else:
                gates.append('NONE')

    return gates

def getes2index(gates_tokens):
    result = []
    for gate in gates_tokens:
        result.append(gate2index[gate])
    return result

def get_initial_slots():
    initial_slots = OrderedDict()
    for d_index, domain in enumerate(domain_list):
        temp = OrderedDict()
        for slot in slots_list[d_index]:
            temp[slot] = ''
        initial_slots[domain] = temp

    return initial_slots

def load_all_slot():
    result = []
    for d_index, domain in enumerate(domain_list):
        for slot in slots_list[d_index]:
            result.append(domain + '-' +slot)
    return result

def slot2state(gates, slots2):
    state_list = []
    for i, gate in enumerate(gates):
        if gate != gate2index['UPDATE'] and gate != gate2index['DONTCARE']:
            state_list.append('[None]')
            continue
        else:
            flag = False
            j=0
            for domain in slots2:
                for attr in slots2[domain]:
                    if j == i:
                        state_list.append(slots2[domain][attr])
                        flag = True
                        break
                    j += 1
                if flag:
                    break    
    
    return state_list

def state2index(state_tokens, value_list):
    result = []
    out_index = len(value_list)
    for state in state_tokens:
        if state == '[None]':
            result.append(0)
        else:
            try:
                a = value_list.index(state)
                result.append(a)
            except ValueError:
                result.append(out_index)
    return result

def get_feed_data(word_dict, data_kind='train', samples_num=300):

    tokens_file = open(data_kind + '_tokens.txt', mode='w', encoding='utf-8')
    index_file = open(data_kind + '_index.txt', mode='w', encoding='utf-8')
    token_str = ''
    index_str = ''

    value_list = load_slot_value_list()
    all_slot = load_all_slot()

    dias_data = []
    in_dias = load_diag_data(data_kind=data_kind,samples_num=samples_num)
    slots1 = get_initial_slots()
    for turn_num, turn_dia in enumerate(in_dias):
        turn_tokens = turn_dia['histr_context']
        sentences_feed_data = uttr_token2index(turn_tokens, word_dict)

        slots_feed_data = slots_attr2index()

        slots2 = turn_dia['belief_state']
        gates_tokens = slots2gates(slots1, slots2)
        gates_feed_data = getes2index(gates_tokens)
        slots1 = copy.deepcopy(slots2)

        state_tokens = slot2state(gates = gates_feed_data,
                        slots2=slots2)
        states_feed_data = state2index(state_tokens=state_tokens,
                                        value_list=value_list)

        turn_domain = turn_dia['domain']
        domin_feed_data = [domain_list.index(turn_domain)]
        ###############################################
        # token_str += str()
        # print(turn_tokens)
        token_str += ' tokens:' + str(turn_tokens) + ' '
        # print(sentences_feed_data)
        index_str += ' tokens:' + str(sentences_feed_data) + ' '
        # print slots
        token_str += ' slot:' + str(all_slot) + ' '
        # print slot index
        index_str += ' slot:' + str(slots_feed_data) + ' '
        #print gates
        token_str += ' gate:' + str(gates_tokens) + ' '
        # print gata index
        index_str += ' gate:' + str(gates_feed_data) + ' '
        #print domin
        token_str += ' domin:' + str(turn_domain) + ' '
        # print domin index
        index_str += ' domin:' + str(domin_feed_data) + ' '

        token_str += ' state:' + str(state_tokens) + '\n'
        #print state index
        index_str += ' state:' + str(states_feed_data) + '\n'
        ######################################################
        dias_data.append([sentences_feed_data, slots_feed_data, gates_feed_data, states_feed_data, domin_feed_data])
    
    tokens_file.write(token_str)
    index_file.write(index_str)
    tokens_file.close()
    index_file.close()
    print(data_kind + ' cases is %d'%(turn_num))
 
    return dias_data

def save_predict(gates_predict, gates_label, kind = 'train'):
    token_file = open(kind + '_tokens_predict.txt', mode='w')
    index_file = open(kind + '_indexs_predict.txt', mode='w')

    leng = len(gates_predict)
    # print(leng)
    token_str = ''
    index_str = ''
    for gate_predict, gate_label in zip(gates_predict, gates_label):
        pre = gate_predict.tolist()
        lab = gate_label.tolist()
        index_str += 'predict:' + str(pre) +';'
        index_str += 'label:' + str(lab) + '\n'

        tok_pre = [GATE_INDEX[i] for i in np.argmax(pre, axis=1)]
        tok_lab = [GATE_INDEX[i] for i in lab]
        token_str += 'predict:' + str(tok_pre) + ';'
        token_str += 'label:' + str(tok_lab) + '\n'

    token_file.write(token_str)
    index_file.write(index_str)
    token_file.close()
    index_file.close()

# def save_input(input, kind='train'):
#     outfile = open('input_' + kind, mode='w')
#     outfile.write(str(input))
#     outfile.close()

def show_f1(gates_predict, gates_label):
    
    for i in range(4):
        tp=0
        fp=0
        fn=0
        tn=0
        for gate_predict, gate_label in zip(gates_predict, gates_label):
            pre = np.argmax(gate_predict, axis=1).tolist()
            lab = gate_label.tolist()
            for a,b in zip(pre, lab):
                if  b== i and a == i:
                    tp +=1
                elif b!=i and a == i:
                    fp+=1
                elif b==i and a!=i:
                    fn+=1
                else:
                    tn+=1
        
        precsion = tp/(tp+fp) if (tp+fp) > 0 else 0
        recall = tp/(tp+fn) if (tp+fn) >0 else 0
        f1 = 2 * precsion * recall/(precsion + recall) if precsion + recall >0 else 0
        print(GATE_INDEX[i] + ' f1 score:  %f' % (f1))

    return



    
