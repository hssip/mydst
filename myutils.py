# -*-coding:utf-8 -*-

import numpy as np 
import json, pickle
from gensim import models
from io import BytesIO
from zipfile import ZipFile
from collections import OrderedDict 
import copy

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

def dialogs2tokens(dialogs):
    user_diag_list = dialogs['user_turns']
    sys_diag_list = dialogs['sys_turns']
    diag_tokens = []
    for user_diag, sys_diag in zip(user_diag_list, sys_diag_list):
        user_tokens = user_diag.strip('\n').split(' ')
        sys_tokens = sys_diag.strip('\n').split(' ')
        # i=0
        turn_tokens = []
        for token in user_tokens:
            turn_tokens.append(token)
            # i+=1
            # if i >= max_sentence_length:
                # break
        # for j in range(i, max_sentence_length):
            # turn_tokens.append('None')
        diag_tokens.append(turn_tokens)

        turn_tokens = []
        # i=0
        for token in sys_tokens:
            turn_tokens.append(token)
            # i+=1
            # if i >= max_sentence_length:
                # break
        # for j in range(i, max_sentence_length):
            # turn_tokens.append('None')
        diag_tokens.append(turn_tokens)

    return diag_tokens 

def get_turn_tokens(turn_number,
                hist_turn_length,
                dia_token_list,
                uttr_token_length,
                if_complete_turns = True):

    all_tokens = ['[START]']

    #assert legal
    if turn_number < 0:
        raise RuntimeError('turn_nunmber connot be negtive')
    elif len(dia_token_list) % 2 != 0:
        raise RuntimeError('dia_token_list length wrong')
    
    # if turn_number < hist_turn_length:
    #     for i in range(hist_turn_length - turn_number):
    #         for j in range(2 * max_sentence_length):
    #             all_tokens.append('None')
    #     for i in range(turn_number):
    #         all_tokens.extend(dia_token_list[2 * i])
    #         all_tokens.extend(dia_token_list[2 * i + 1])
    # else:
    #     for i in range(turn_number - hist_turn_length, turn_number):
    #         all_tokens.extend(dia_token_list[2 * i])
    #         all_tokens.extend(dia_token_list[2 * i + 1])
    if turn_number < hist_turn_length:
        for i in range(turn_number):
            all_tokens.extend(dia_token_list[2 * i])
            all_tokens.extend(dia_token_list[2 * i + 1])
            all_tokens.append('[STEP]')
    else:
        for i in range(turn_number - hist_turn_length, turn_number):
            all_tokens.extend(dia_token_list[2 * i])
            all_tokens.extend(dia_token_list[2 * i + 1])
            all_tokens.append('[STEP]')

    all_tokens.extend(dia_token_list[2 * turn_number])
    # all_tokens.extend(dia_token_list[2 * turn_number + 1])

    all_tokens.append('[END]')
    leng = len(all_tokens)

    for i in range(leng, uttr_token_length):
        all_tokens.append('[NONE]')

    leng = len(all_tokens)
    new_all_tokens = []
    for i in range(leng - uttr_token_length, leng):
        new_all_tokens.append(all_tokens[i])

    return new_all_tokens

def uttr_token2index(tokens, word_dict):
    tokindx = []
    for token in tokens:
        a = bytes(token, encoding='utf8')
        if  a in word_dict:
            tokindx.append(word_dict[a])
        else:
            tokindx.append(0)
    
    return np.array(tokindx).astype('int64')

def slots_attr2index():
    slotsindex = []
    i = 0
    for d_index, domin in enumerate(domin_list):
        for slot in slots_list[d_index]:
            # a = bytes(slot, encoding='utf8')
            # if a in word_dict:
            #     slotsindex.append(word_dict[a])
            # else:
            #     slotsindex.append(0)
            slotsindex.append(i)
            i+=1
    
    return np.array(slotsindex).astype('int64')

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
            elif slots2[domin][slot] in special_slot_value['none'] and \
                slots1[domin][slot] not in special_slot_value['dontcare'] and \
                slots1[domin][slot] not in special_slot_value['none']:
                gates.append(gate2index['DELETE'])
            else:
                gates.append(gate2index['NONE'])

    return np.array(gates).astype('int64')

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
            result.append(domin + '-' +slot)
    
    return result

def slot2state(gates, slots2, value_list):
    state_list = []
    value_num = len(value_list)
    
    for i, gate in enumerate(gates):
        # state = [0 for i in range(value_num + 1)]
        if gate != gate2index['UPDATE'] and gate != gate2index['DONTCARE']:
            state_list.append(0)
            continue
        else:
            flag = False
            j=0
            for domin in slots2:
                for attr in slots2[domin]:
                    if j == i:
                        try:
                            # state[value_list.index(slots2[domin][attr])] = 1
                            state_list.append(j)
                        except ValueError:
                            # print('EXCEPT: domin: %s attr: %s value: %s '%(domin, attr, slots2[domin][attr]))
                            state_list.append(0)
                        flag = True
                        break
                    j += 1
                if flag:
                    break    
    
    return np.array(state_list).astype('int64')

def get_feed_data(in_dias, 
                    hist_turn_length, 
                    uttr_token_length, 
                    word_dict, 
                    values_list, 
                    all_slot,
                    slots_feed_data, 
                    kind='train'):
    dias_data = {}

    tokens_file = open(kind + '_tokens.txt', mode='w+', encoding='utf-8')
    index_file = open(kind + '_index.txt', mode='w+', encoding='utf-8')

    for dia_name, dia in in_dias.items():
        dia_tokens = dialogs2tokens(dialogs=dia)
        turns = int(len(dia_tokens)/2)
        slots1 = get_initial_slots()

        dia_sentence_data = []
        dia_gate_data = []
        dia_state_data = []
        for i in range(turns):
            turn_tokens = get_turn_tokens(turn_number=i,
                                            hist_turn_length=hist_turn_length,
                                            dia_token_list=dia_tokens,
                                            uttr_token_length=uttr_token_length,
                                            if_complete_turns=True)

            slots2 = dia['turns_status'][i]
            gates_feed_data = slots2gates(slots1, slots2)
            slots1 = copy.deepcopy(slots2)

            sentences_feed_data = uttr_token2index(turn_tokens, word_dict)

            state_feed_data = slot2state(gates = gates_feed_data,
                            slots2=slots2,
                            value_list=values_list)
            ###############################################
            dia_sentence_data.append(sentences_feed_data)
            dia_gate_data.append(gates_feed_data)
            dia_state_data.append(state_feed_data)

            #save_data
            #print sentence
            token_str = ''
            index_str = ''
            # print(turn_tokens)
            token_str += 'tokens:['
            for token in turn_tokens:
                token_str += token + ', '
            token_str += '] \n'

            # print(sentences_feed_data)
            index_str += 'tokens:['
            for index in sentences_feed_data:
                index_str += str(index) + ', '
            index_str += '] \n'

            # print slots
            token_str += 'slot:['
            for slot in all_slot:
                token_str += slot + ', '
            token_str += '] \n'

            # print slot index
            index_str += 'slot:['
            for slot in slots_feed_data:
                index_str += str(slot) + ', '
            index_str += '] \n'

            #print gates
            token_str += 'gate:['
            for gate in gates_feed_data:
                token_str += GATE_INDEX[gate] + ', '
            token_str += '] \n'

            # print gata index
            index_str += 'gate:['
            for gate in gates_feed_data:
                index_str += str(gate) + ', '
            index_str += '] \n'

            token_str += 'state:['
            for index in state_feed_data:
                if index == 0:
                    token_str += '[NONE], '
                else:
                    token_str += values_list[index] + ', '
            token_str += '] \n'
            token_str += '*********************************\n'

            #print state index
            index_str += 'state:['
            for index in state_feed_data:
                index_str += str(index) + ', '
            index_str += '] \n'
            index_str += '*********************************\n'
            tokens_file.write(token_str)
            index_file.write(index_str)

        
        dias_data[dia_name] = {'dia_sentence_data':dia_sentence_data,
                        'dia_gate_data': dia_gate_data,
                        'dia_state_data': dia_state_data}

    tokens_file.close()
    index_file.close()


    return dias_data

# def save_feed_data(dias_data, 
                    # slots_feed_data,
                    # kind='train'):



    # for dia_name, dia_data in dias_data.items():
    #     dia_sentence_data = dia_data['dia_sentence_data']
    #     dia_gate_data = dia_data['dia_gate_data']
    #     dia_state_data = dia_data['dia_state_data']
    #     turns = 
    return

# def save_token_data()