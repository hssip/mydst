# -*-coding:utf-8 -*-

import numpy as np 
import paddle as pd
import paddle.fluid as fluid
from collections import OrderedDict
import math

from creative_data import *
from myutils import *
# import tensorflow as tf

# SENTENCE_VEC_LENGTH = 256
# SLOT_VEC_LENGTH = 128
# HISTR_HIDDEN_NUM = 64
# HISTR_VEC_LENGTH = SLOT_VEC_LENGTH + SENTENCE_VEC_LENGTH
# HISTR_LENGTH = 128
# LAYERS_NUM = 1
# # CURR_SENTENCE_MAX_LENGTH = 128
# SENTENCE_MAX_LENGTH = 128
# VOCAB_VEC_LENGTH = 256

HISTR_TURNS_LENGTH = 2
SENTENCE_LENGTH = 64
# VOCAB_EMBEDDING_LENGTH = 256
# SLOT_EMBEDDING_LENGTH = 128
VOCAB_EMBEDDING_LENGTH = 300
SLOT_EMBEDDING_LENGTH = 300

UTTR_TOKEN_LENGTH = (HISTR_TURNS_LENGTH * 2 + 1) * SENTENCE_LENGTH

ENCODER_HIDDEN_SIZE = 64
ENCODER_LAYERS_NUM = 1
ATTENTION_HEAD_NUM = 1


all_slot = load_all_slot()
ALL_SLOT_NUM = len(all_slot)
SLOT_EMBEDDING_LENGTH
GATE_KIND = 4
GATE_INDEX = ['UPDATE', 'DONTCARE', 'NONE', 'DELETE']


# EMBEDDIND_FILE_NAME = 'pub_dataset/ignore_GoogleNews-vectors-negative300.bin'
EMBEDDIND_FILE_NAME = 'pub_dataset/ignore_GoogleNews-vectors-negative300-SLIM.bin'



# def utterance_encoder():
#     slot_holder = fluid.data(name='slot_holder', shape=[None, HISTR_LENGTH, SENTENCE_MAX_LENGTH, SLOT_VEC_LENGTH])
#     sentence_holder = fluid.data(name='sentence_holder',shape=[None, HISTR_LENGTH, SENTENCE_VEC_LENGTH])
#     histr_input = fluid.layers.concat([slot_holder, sentence_holder], axis=-1)
#     init_h = fluid.layers.fill_constant(shape=[LAYERS_NUM, HISTR_LENGTH, HISTR_HIDDEN_NUM], dtype='float32',value=0.0)
#     init_c = fluid.layers.fill_constant(shape=[LAYERS_NUM, HISTR_LENGTH, HISTR_HIDDEN_NUM], dtype='float32',value=0.0)
#     rnn_out, last_h, last_c = fluid.layers.lstm(histr_input, 
#                                                 init_h=init_h,
#                                                 init_c=init_c,
#                                                 hidden_size=HISTR_HIDDEN_NUM,
#                                                 num_layers=LAYERS_NUM,
#                                                 is_bidirec=True,
#                                                 max_len=HISTR_VEC_LENGTH)                                        
# def curr_encoder():
    # curr_sentence = fluid.data(name='curr_sentence', shape=[None, SENTENCE_MAX_LENGTH, VOCAB_VEC_LENGTH])

def utterance_encoder(sentence):

    # histr_slot_holder = fluid.data(name='histr_slot_holder', shape=[None, ALL_SLOT_NUM, SLOT_EMBEDDING_LENGTH])

    init_h = fluid.layers.fill_constant(shape=[ENCODER_LAYERS_NUM, UTTR_TOKEN_LENGTH, ENCODER_HIDDEN_SIZE], dtype='float32',value=0.0)
    init_c = fluid.layers.fill_constant(shape=[ENCODER_LAYERS_NUM, UTTR_TOKEN_LENGTH, ENCODER_HIDDEN_SIZE], dtype='float32',value=0.0)

    encode_out, encode_last_h, encode_last_c = fluid.layers.lstm(input=sentence, 
                                                                init_h=init_h,
                                                                init_c=init_c,
                                                                max_len=VOCAB_EMBEDDING_LENGTH,
                                                                hidden_size=ENCODER_HIDDEN_SIZE,
                                                                num_layers=ENCODER_HIDDEN_SIZE,
                                                                is_bidirec=True)
    
    # encoder_result = fluid.layers.concat(encode_out[:][0][:], axis=[])
    return encode_out

def state_generator(encoder_result, slots_embedding):

    encoder_result = fluid.layers.reshape(encoder_result, shape=[UTTR_TOKEN_LENGTH, ENCODER_HIDDEN_SIZE])
    # slots_to_hidden = fluid.layers.create_parameter(shape=[SLOT_EMBEDDING_LENGTH, UTTR_TOKEN_LENGTH], 
    #                                                 dtype='float32', 
    #                                                 name='slots_to_hidden',
    #                                                 default_initializer=None)
    # hidden_to_vocab = fluid.layers.create_parameter(shape=[ENCODER_HIDDEN_SIZE, VOCAB_EMBEDDING_LENGTH], 
    #                                             dtype='float32', 
    #                                             name='hidden_to_vocab',
    #                                             default_initializer=None)
    # slots = fluid.layers.mul(slots_embedding, slots_to_hidden)
    # hiddens = fluid.layers.mul(encoder_result, hidden_to_vocab)
    # vocabs = fluid.layers.mul(slots, hiddens)

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

# def calcu_cost(gates, gates_label, states, states_label):
#     loss1 = fluid.layers.reduce_mean(fluid.layers.cross_entropy(gates, gates_label))
#     loss2 = fluid.layers.reduce_mean(fluid.layers.square(states - states_label))

#     loss = 0.0
#     for gate_label in gates_label:
#         if fluid.layers.argmax(gates_label).numpy()[0] == 1:
#             loss += loss1 + loss2
#         else:
#             loss += loss1

def calcu_cost(gates, gates_label):
    loss1 = fluid.layers.reduce_mean(fluid.layers.cross_entropy(gates, gates_label))
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


def mymodel():
    slots_holder = fluid.data('slots_holder', shape=[ALL_SLOT_NUM, SLOT_EMBEDDING_LENGTH], dtype='float32')
    sentence_holder = fluid.data(name='sentence_holder', shape=[None, UTTR_TOKEN_LENGTH, VOCAB_EMBEDDING_LENGTH])
    encoder_result = utterance_encoder(sentence_holder)
    states = state_generator(encoder_result, slots_holder)
    gates = slot_gate(encoder_result, slots_holder)

    return gates, states


def train_program():
    # slots = np.array()
    # utterances = np.array()

    gates_label = fluid.data('gates_label', shape=[ALL_SLOT_NUM, 1], dtype='int32')
    state_label= fluid.data('state_label', shape=[ALL_SLOT_NUM, VOCAB_EMBEDDING_LENGTH], dtype='float32')

    gates_predict, state_predict = mymodel()
    cost = calcu_cost(gates_predict, gates_label)
    acc = calcu_acc(gates=gates_predict, gates_label=gates_label)
    
    return cost, acc
    

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
main_program = fluid.default_main_program()
# feeder.feed()
cost, acc = train_program()
optimizer = optimizer_program()
optimizer.minimize(cost)
feeder = fluid.DataFeeder(feed_list = ['sentence_holder', 'slots_holder',
                                        'gates_label', 'state_label'],
                            place=place)

w = get_embedding_dict(EMBEDDIND_FILE_NAME) #cost lot of time

padding_embed = []
for i in range(SENTENCE_LENGTH):
    padding_embed.append([0 for j in range(VOCAB_EMBEDDING_LENGTH)])

dias = load_diag_data(max_length=SENTENCE_LENGTH)
for dia_name, dia in dias.items():
    embed_dia = dialogs2embedding(dia, SENTENCE_LENGTH, w, WORD_EMBEDDING_LENGTH=300)
    turns = int(len(embed_dia)/2)
    slots1 = get_initial_slots()
    for i in range(turns):
        sentence_feed_data = []
        if i < HISTR_TURNS_LENGTH:
            for j in range(i, HISTR_TURNS_LENGTH):
                sentence_feed_data.extend(padding_embed)
                sentence_feed_data.extend(padding_embed)
            for j in range(i):
                sentence_feed_data.extend(embed_dia[2 * i])
                sentence_feed_data.extend(embed_dia[2 * i + 1])
            sentence_feed_data.extend(embed_dia[2 * i])
        else:
            for j in range(i-HISTR_TURNS_LENGTH, i):
                sentence_feed_data.extend(embed_dia[2 * j])
                sentence_feed_data.extend(embed_dia[2 * i + 1])
            sentence_feed_data.extend(embed_dia[2 * i])

        sentence_feeder = fluid.Tensor()
        sentence_feeder.set(np.array(sentence_feed_data))

        slots2 = dia['turns_status'][i]

        slot_feeder = fluid.Tensor()
        slot_feeder.set(np.array(slots2embed(slots2 ,w)))

        gates_feeder = np.array(slots2gates(slots1, slots2))

        state_feeder = np.array()

        # feed_data = {'sentence_holder':sentence_feeder,
        #             'slots_holder':'',
        #             'gates_label':'', 
        #             'state_label':''}
        cost, acc = exe.run(main_program,
                        feed=feeder.feed([sentence_feeder,
                            slot_feeder,
                            gates_feeder,
                            state_feeder]),
                        fetch_list=[cost, acc],
                        )
    
        print('cost is : %f, acc is: %f'%(cost, acc))





