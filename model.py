# -*-coding:utf-8 -*-

import numpy as np 
import paddle as pd
import paddle.fluid as fluid
from collections import OrderedDict
import math, copy

from creative_data import *
from myutils import *
# import tensorflow as tf


HISTR_TURNS_LENGTH = 2
SENTENCE_LENGTH = 32
VOCAB_EMBEDDING_LENGTH = 128
SLOT_EMBEDDING_LENGTH = 128

UTTR_TOKEN_LENGTH = (HISTR_TURNS_LENGTH * (2 + 1)) * SENTENCE_LENGTH

ENCODER_HIDDEN_SIZE = 64
ENCODER_LAYERS_NUM = 1
ATTENTION_HEAD_NUM = 1


all_slot = load_all_slot()
ALL_SLOT_NUM = len(all_slot)
GATE_KIND = 4
GATE_INDEX = ['UPDATE', 'DONTCARE', 'NONE', 'DELETE']


# EMBEDDIND_FILE_NAME = 'pub_dataset/ignore_GoogleNews-vectors-negative300.bin'
# EMBEDDIND_FILE_NAME = 'pub_dataset/ignore_GoogleNews-vectors-negative300-SLIM.bin'

def utterance_encoder(sentences, dict_size):

    # sentence_holder = fluid.data(name='sentence_holder', 
    #                             shape=[None, UTTR_TOKEN_LENGTH, VOCAB_EMBEDDING_LENGTH],
    #                             dtype='float32')
    emb = fluid.embedding(input=sentences,
                            size=[dict_size, VOCAB_EMBEDDING_LENGTH],
                            padding_idx=0)

    init_h = fluid.layers.fill_constant(shape=[ENCODER_LAYERS_NUM, UTTR_TOKEN_LENGTH, ENCODER_HIDDEN_SIZE], dtype='float32',value=0.0)
    init_c = fluid.layers.fill_constant(shape=[ENCODER_LAYERS_NUM, UTTR_TOKEN_LENGTH, ENCODER_HIDDEN_SIZE], dtype='float32',value=0.0)
    
    emb = fluid.layers.reshape(x = emb,
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
                                                inputs=emb)
    
    return encode_out

def state_generator(encoder_result, slots_embedding):

    encoder_result = fluid.layers.reshape(encoder_result, shape=[UTTR_TOKEN_LENGTH, ENCODER_HIDDEN_SIZE])

    Q_1 = fluid.layers.create_parameter(shape=[VOCAB_EMBEDDING_LENGTH, int(VOCAB_EMBEDDING_LENGTH/ATTENTION_HEAD_NUM)], 
                                                dtype='float32', 
                                                name='Q_1',
                                                default_initializer=None)
    K_1 = fluid.layers.create_parameter(shape=[ENCODER_HIDDEN_SIZE, int(VOCAB_EMBEDDING_LENGTH/ATTENTION_HEAD_NUM)], 
                                                dtype='float32', 
                                                name='K_1',
                                                default_initializer=None)
    V_1 = fluid.layers.create_parameter(shape=[ENCODER_HIDDEN_SIZE, int(VOCAB_EMBEDDING_LENGTH/ATTENTION_HEAD_NUM)], 
                                                dtype='float32', 
                                                name='V_1',
                                                default_initializer=None)
    q1 = fluid.layers.mul(slots_embedding, Q_1)
    k1 = fluid.layers.mul(encoder_result, K_1)
    v1 = fluid.layers.mul(encoder_result, V_1)
    qk = fluid.layers.mul(q1, fluid.layers.t(k1))/math.sqrt(VOCAB_EMBEDDING_LENGTH)
    head1 = fluid.layers.mul(fluid.layers.softmax(qk), v1)

    #calc
    states = fluid.layers.fc(head1, size = VOCAB_EMBEDDING_LENGTH, act='relu')
    # vocab = np.sum(np.array(state), axis=0)
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
    slots = fluid.layers.mul(slots_embedding, slots_to_hidden)
    hiddens = fluid.layers.mul(encoder_result ,hidden_to_gate)
    gates = fluid.layers.softmax(fluid.layers.mul(slots, hiddens))
    return gates

######### will be used in practice###################
# def update_slots(slot, vocab, gate):
#     gates = np.array(fluid.layers.argmax(gates,axis=-1))
#     for i in range(ALL_SLOT_NUM):
#         if GATE_INDEX[gates[i]] == 'UPDATE':
#             pass
#         elif GATE_INDEX[gates[i]] == 'DONTCARE':
#             pass
#         elif GATE_INDEX[gates[i]] == 'NONE':
#             pass
#         elif GATE_INDEX[gates[i]] == 'DELETE':
#             pass
#     return slot

def optimizer_program():
    return fluid.optimizer.Adam(learning_rate=0.01)

def calcu_cost(gates, gates_label):
    loss1 = fluid.layers.mean(fluid.layers.cross_entropy(gates, gates_label))
    # loss2 = fluid.layers.reduce_mean(fluid.layers.square(states - states_label))
    loss2 = 0.0
    loss = 0.0
    # for gate_label in gates_label:
    #     if fluid.layers.argmax(gates_label).numpy()[0] == 1:
    #         loss += loss1 + loss2
    #     else:
    #         loss += loss1
    return loss1

def calcu_acc(gates, gates_label):

    return fluid.layers.accuracy(input=gates, label=gates_label)


def mymodel(dict_size):
    sentence_index_holder = fluid.data(name='sentence_index_holder',
                                shape=[None, UTTR_TOKEN_LENGTH],
                                dtype='int64')

    slots_index_holder = fluid.data(name='slots_index_holder',
                                    shape=[ALL_SLOT_NUM],
                                    dtype='int64')
    slots_emb = fluid.embedding(input=slots_index_holder,
                                size=[ALL_SLOT_NUM, SLOT_EMBEDDING_LENGTH],
                                padding_idx=0)

    encoder_result = utterance_encoder(sentence_index_holder, dict_size)
    states = state_generator(encoder_result, slots_emb)
    gates = slot_gate(encoder_result, slots_emb)

    return gates, states

def train_program(dict_size):

    gates_label = fluid.data('gates_label', shape=[ALL_SLOT_NUM, 1], dtype='int64')
    state_label= fluid.data('state_label', shape=[ALL_SLOT_NUM, VOCAB_EMBEDDING_LENGTH], dtype='float32')

    gates_predict, state_predict = mymodel(dict_size)
    cost = calcu_cost(gates_predict, gates_label)
    acc = calcu_acc(gates=gates_predict, gates_label=gates_label)
    
    return cost, acc, gates_predict

word_dict = pd.dataset.imdb.word_dict()

cost, acc, fetch_gates_label = train_program(len(word_dict))
optimizer = optimizer_program()
optimizer.minimize(cost)

# place = fluid.CUDAPlace(0)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
main_program = fluid.default_main_program()
feed_order = ['sentence_index_holder', 'slots_index_holder',
                'gates_label', 'state_label']
# feed_var_list_loop = [main_program.global_block().var(var_name) for var_name in feed_order]
# feeder = fluid.DataFeeder(feed_list = feed_var_list_loop,
                            # place=place)

dias = load_diag_data(max_length=SENTENCE_LENGTH)
slots_feed_data = slots_attr2index(word_dict)
dia_num = 0
for dia_name, dia in dias.items():
    dia_num += 1
    dia_tokens = dialogs2tokens(dialogs=dia,
                                max_sentence_length=SENTENCE_LENGTH)
    turns = int(len(dia_tokens)/2)
    slots1 = get_initial_slots()
    for i in range(turns):
        turn_tokens = get_turn_tokens(turn_number=i,
                                        hist_turn_length=HISTR_TURNS_LENGTH,
                                        max_sentence_length=SENTENCE_LENGTH,
                                        dia_token_list=dia_tokens,
                                        if_complete_turns=True)
        sentences_feed_data = uttr_token2index(turn_tokens, word_dict)

        slots2 = dia['turns_status'][i]
        gates_feed_data = slots2gates(slots1, slots2)
        slots1 = copy.deepcopy(slots2)

        state_feed_data = np.zeros(shape=(ALL_SLOT_NUM, VOCAB_EMBEDDING_LENGTH),
                                dtype='float32')
        # print(feeder)
        # print(sentences_feed_data.shape)
        # print(slots_feed_data.shape)
        # print(gates_feed_data.shape)
        # print(state_feed_data.shape)
        myfeed = {
            'sentence_index_holder':sentences_feed_data,
            'slots_index_holder':slots_feed_data,
            'gates_label':gates_feed_data,
            'state_label':state_feed_data
        }

        cost1, acc1, a = exe.run(main_program,
                        # feed=feeder.feed([sentences_feed_data,
                        #     slots_feed_data,
                        #     gates_feed_data,
                        #     state_feed_data]),
                        feed = myfeed,
                        fetch_list=[cost, acc, fetch_gates_label],
                        )
        if i == turns - 1:
            print('cost is : %f, acc is: %f'%(cost1, acc1))
            print(a)
