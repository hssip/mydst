# -*-coding:utf-8 -*-

import numpy as np 
import paddle as pd
import paddle.fluid as fluid
from collections import OrderedDict
import math, copy

from creative_data import *
from myutils import *
import time
# import tensorflow as tf


HISTR_TURNS_LENGTH = 2
SENTENCE_LENGTH = 32
VOCAB_EMBEDDING_LENGTH = 128
SLOT_EMBEDDING_LENGTH = 128

# UTTR_TOKEN_LENGTH = (HISTR_TURNS_LENGTH * (2 + 1)) * SENTENCE_LENGTH
UTTR_TOKEN_LENGTH = 128

ENCODER_HIDDEN_SIZE = 64
ENCODER_LAYERS_NUM = 1
ATTENTION_HEAD_NUM = 1


all_slot = load_all_slot()
ALL_SLOT_NUM = len(all_slot)
values_list = load_slot_value_list()
SLOT_VALUE_NUM = len(values_list) + 1
GATE_KIND = 4
GATE_INDEX = ['UPDATE', 'DONTCARE', 'NONE', 'DELETE']


# EMBEDDIND_FILE_NAME = 'pub_dataset/ignore_GoogleNews-vectors-negative300.bin'
# EMBEDDIND_FILE_NAME = 'pub_dataset/ignore_GoogleNews-vectors-negative300-SLIM.bin'

def utterance_encoder(sentences, dict_size):

    emb = fluid.embedding(input=sentences,
                            size=[dict_size, VOCAB_EMBEDDING_LENGTH])

    # init_h = fluid.layers.fill_constant(shape=[ENCODER_LAYERS_NUM, UTTR_TOKEN_LENGTH, ENCODER_HIDDEN_SIZE], dtype='float32',value=0.0)
    # init_c = fluid.layers.fill_constant(shape=[ENCODER_LAYERS_NUM, UTTR_TOKEN_LENGTH, ENCODER_HIDDEN_SIZE], dtype='float32',value=0.0)
    
    aemb = fluid.layers.reshape(x = emb,
                                shape=[1, UTTR_TOKEN_LENGTH, VOCAB_EMBEDDING_LENGTH])

    # encode_out, encode_last_h, encode_last_c = fluid.layers.lstm(input=emb, 
    #                                                             init_h=init_h,
    #                                                             init_c=init_c,
    #                                                             max_len=VOCAB_EMBEDDING_LENGTH,
    #                                                             hidden_size=ENCODER_HIDDEN_SIZE,
    #                                                             num_layers=ENCODER_HIDDEN_SIZE,
    #                                                             is_bidirec=False)
    cell = fluid.layers.GRUCell(hidden_size=ENCODER_HIDDEN_SIZE)
    encode_out, encode_last_h = fluid.layers.rnn(cell=cell,
                                                inputs=aemb)
    
    return encode_out, emb

def state_generator(encoder_result, slots_embedding):

    encoder_result = fluid.layers.reshape(encoder_result, shape=[UTTR_TOKEN_LENGTH, ENCODER_HIDDEN_SIZE])

    Q_1 = fluid.layers.create_parameter(shape=[SLOT_EMBEDDING_LENGTH, int(SLOT_VALUE_NUM/ATTENTION_HEAD_NUM)], 
                                                dtype='float32', 
                                                name='Q_1',
                                                default_initializer=None)
    K_1 = fluid.layers.create_parameter(shape=[ENCODER_HIDDEN_SIZE, int(SLOT_VALUE_NUM/ATTENTION_HEAD_NUM)], 
                                                dtype='float32', 
                                                name='K_1',
                                                default_initializer=None)
    V_1 = fluid.layers.create_parameter(shape=[ENCODER_HIDDEN_SIZE, int(SLOT_VALUE_NUM/ATTENTION_HEAD_NUM)], 
                                                dtype='float32', 
                                                name='V_1',
                                                default_initializer=None)
    q1 = fluid.layers.mul(slots_embedding, Q_1)
    k1 = fluid.layers.mul(encoder_result, K_1)
    v1 = fluid.layers.mul(encoder_result, V_1)

    qk = fluid.layers.mul(q1, fluid.layers.t(k1))# /math.sqrt(SLOT_VALUE_NUM)
    head1 = fluid.layers.mul(fluid.layers.softmax(qk), v1)

    #calc
    states = fluid.layers.fc(head1, size = SLOT_VALUE_NUM, act='relu')

    return states

def slot_gate(encoder_result, slots_embedding):

    encoder_result = fluid.layers.reshape(encoder_result, shape=[UTTR_TOKEN_LENGTH, ENCODER_HIDDEN_SIZE])

    slots_to_hidden = fluid.layers.create_parameter(shape=[SLOT_EMBEDDING_LENGTH, UTTR_TOKEN_LENGTH], 
                                                    dtype='float32', 
                                                    name='slots_to_hidden',
                                                    default_initializer=None)
    hidden_to_gate = fluid.layers.create_parameter(shape=[ENCODER_HIDDEN_SIZE, GATE_KIND], 
                                                dtype='float32', 
                                                name='hidden_to_gate',
                                                default_initializer=None)
    slots = fluid.layers.tanh(fluid.layers.mul(slots_embedding, slots_to_hidden))
    hiddens = fluid.layers.tanh(fluid.layers.mul(encoder_result ,hidden_to_gate))
    gates = fluid.layers.softmax(fluid.layers.mul(slots, hiddens))
    return gates

def optimizer_program():
    return fluid.optimizer.Adagrad(learning_rate=0.01)

def calcu_cost(gates, gates_label, state, state_label):
    loss1 = fluid.layers.mean(fluid.layers.cross_entropy(gates, gates_label))
    # loss2 = fluid.layers.mean(fluid.layers.cross_entropy(state, state_label))
    # temp_state = np.zeros(shape=[state_label.shape[0], ALL_SLOT_NUM])
    # for state in state_label:
    #     temp_state[state[0]] = 1
    # loss2 = fluid.layers.mean(fluid.layers.mul(state, state_label))
    # loss2 = fluid.layers.reduce_mean(fluid.layers.square(states - states_label))
    # loss2 = 0.0
    # loss = 0.0
    # for gate_label in gates_label:
    #     if fluid.layers.argmax(gates_label).numpy()[0] == 1:
    #         loss += loss1 + loss2
    #     else:
    #         loss += loss1
    return loss1
    # return loss1 + loss2

def calcu_acc(gates, gates_label):

    return fluid.layers.accuracy(input=gates, label=gates_label)


def mymodel(dict_size):
    sentences_index_holder = fluid.data(name='sentences_index_holder',
                                shape=[UTTR_TOKEN_LENGTH],
                                dtype='int64')

    slots_index_holder = fluid.data(name='slots_index_holder',
                                    shape=[ALL_SLOT_NUM],
                                    dtype='int64')
    slots_emb = fluid.embedding(input=slots_index_holder,
                                size=[ALL_SLOT_NUM, SLOT_EMBEDDING_LENGTH],
                                padding_idx=0)

    encoder_result, a = utterance_encoder(sentences_index_holder, dict_size)
    states= state_generator(encoder_result, slots_emb)
    gates = slot_gate(encoder_result, slots_emb)

    return gates, states, a

def train_program(dict_size):

    gates_label = fluid.data('gates_label', shape=[ALL_SLOT_NUM, 1], dtype='int64')
    state_label= fluid.data('state_label', shape=[ALL_SLOT_NUM, 1], dtype='int64')

    gates_predict, state_predict, a = mymodel(dict_size)
    cost = calcu_cost(gates=gates_predict, 
                    gates_label=gates_label, 
                    state=state_predict, 
                    state_label=state_label)
    acc = calcu_acc(gates=gates_predict, gates_label=gates_label)
    
    return cost, acc, a, state_predict

word_dict = pd.dataset.imdb.word_dict()

cost, acc, a, b = train_program(len(word_dict))
optimizer = optimizer_program()
optimizer.minimize(cost)

# place = fluid.CUDAPlace(0)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
main_program = fluid.default_main_program()
# feed_order = ['sentence_index_holder', 'slots_index_holder',
                # 'gates_label', 'state_label']
# feed_var_list_loop = [main_program.global_block().var(var_name) for var_name in feed_order]
# feeder = fluid.DataFeeder(feed_list = feed_var_list_loop,
                            # place=place)

tokens_file = open('tokens.txt', mode='w+', encoding='utf-8')
index_file = open('index.txt', mode='w+', encoding='utf-8')

dias = load_diag_data()
slots_feed_data = slots_attr2index()
dia_num = 0
for dia_name, dia in dias.items():
    print(dia_name)
    dia_num += 1
    # dia = process_dialog(dia, SENTENCE_LENGTH)
    dia_tokens = dialogs2tokens(dialogs=dia)
    turns = int(len(dia_tokens)/2)
    slots1 = get_initial_slots()
    # print(dia_tokens)
    for i in range(turns):
        turn_tokens = get_turn_tokens(turn_number=i,
                                        hist_turn_length=HISTR_TURNS_LENGTH,
                                        dia_token_list=dia_tokens,
                                        uttr_token_length=UTTR_TOKEN_LENGTH,
                                        if_complete_turns=True)
        token_str = ''
        index_str = ''
        # print(turn_tokens)
        token_str += 'tokens:['
        for token in turn_tokens:
            token_str += token + ', '
        token_str += '] \n'

        sentences_feed_data = uttr_token2index(turn_tokens, word_dict)

        # print(sentences_feed_data)
        index_str += 'tokens:['
        for index in sentences_feed_data:
            index_str += str(index) + ', '
        index_str += '] \n'

        #print slots
        token_str += 'slot:['
        for slot in all_slot:
            token_str += slot + ', '
        token_str += '] \n'

        #print slot index
        index_str += 'slot:['
        for slot in slots_feed_data:
            index_str += str(slot) + ', '
        index_str += '] \n'

        slots2 = dia['turns_status'][i]
        gates_feed_data = slots2gates(slots1, slots2)
        slots1 = copy.deepcopy(slots2)

        #print gates
        token_str += 'gate:['
        for gate in gates_feed_data:
            token_str += GATE_INDEX[gate[0]] + ', '
        token_str += '] \n'

        #print gata index
        index_str += 'gate:['
        for gate in gates_feed_data:
            index_str += str(gate[0]) + ', '
        index_str += '] \n'

        state_feed_data = slot2state(gates = gates_feed_data,
                                    slots2=slots2,
                                    value_list=values_list)
        # print(sentences_feed_data)
        # print(slots_feed_data)
        # print(gates_feed_data)

        # print(state_feed_data)
        #print state

        # get_value = []
        token_str += 'state:['
        for index in state_feed_data:
            if index[0] == 0:
                token_str += '[NONE], '
            else:
                token_str += values_list[index[0]] + ', '
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

        myfeed = {
            'sentences_index_holder':sentences_feed_data,
            'slots_index_holder':slots_feed_data,
            'gates_label':gates_feed_data,
            'state_label':state_feed_data
        }

        cost1, acc1, a1, b1= exe.run(main_program,
                        feed = myfeed,
                        fetch_list=[cost, acc, a, b],
                        )
        # print(a1)
        # print(b1)
        if i == turns - 1:
            print('cost is : %f, acc is: %f'%(cost1, acc1))
        # time.sleep(0.5)
            # print(sentences_feed_data.dtype)
            # print(a1)

tokens_file.close()
index_file.close()
