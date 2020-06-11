# -*-coding:utf-8 -*-

import numpy as np 
import paddle as pd
import paddle.fluid as fluid
from myutils import *
from collections import OrderedDict
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

HISTR_LENGTH = 32
SENTENCE_LENGTH = 300
# VOCAB_EMBEDDING_LENGTH = 256
# SLOT_EMBEDDING_LENGTH = 128
VOCAB_EMBEDDING_LENGTH = 300
SLOT_EMBEDDING_LENGTH = 300

UTTR_TOKEN_LENGTH = HISTR_LENGTH * SENTENCE_LENGTH

ENCODER_HIDDEN_SIZE = 64
ENCODER_LAYERS_NUM = 1
ATTENTION_HEAD_NUM = 1


all_slot = load_all_slot()
ALL_SLOT_NUM = len(all_slot)
SLOT_EMBEDDING_LENGTH
GATE_KIND = 4
GATE_INDEX = ['UPDATE', 'DONTCARE', 'NONE', 'DELETE']


EMBEDDIND_FILE_NAME = 'pub_dataset/ignore_GoogleNews-vectors-negative300.bin'



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

def utterance_encoder():

    sentence_holder = fluid.data(name='sentence_holder', shape=[None, UTTR_TOKEN_LENGTH, VOCAB_EMBEDDING_LENGTH])
    # histr_slot_holder = fluid.data(name='histr_slot_holder', shape=[None, ALL_SLOT_NUM, SLOT_EMBEDDING_LENGTH])

    init_h = fluid.layers.fill_constant(shape=[ENCODER_LAYERS_NUM, UTTR_TOKEN_LENGTH, ENCODER_HIDDEN_SIZE], dtype='float32',value=0.0)
    init_c = fluid.layers.fill_constant(shape=[ENCODER_LAYERS_NUM, UTTR_TOKEN_LENGTH, ENCODER_HIDDEN_SIZE], dtype='float32',value=0.0)

    encode_out, encode_last_h, encode_last_c = fluid.layers.lstm(input=sentence_holder, 
                                                                init_h=init_h,
                                                                init_c=init_c,
                                                                max_len=VOCAB_EMBEDDING_LENGTH,
                                                                hidden_size=ENCODER_HIDDEN_SIZE,
                                                                num_layers=ENCODER_HIDDEN_SIZE,
                                                                is_bidirec=True)
    
    # encoder_result = fluid.layers.concat(encode_out[:][0][:], axis=[])
    return encode_out

def state_generator(encoder_result, slots_embedding):

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
    K_1 = fluid.layers.create_parameter(shape=[VOCAB_EMBEDDING_LENGTH, int(VOCAB_EMBEDDING_LENGTH/ATTENTION_HEAD_NUM)], 
                                                dtype='float32', 
                                                name='K_1',
                                                default_initializer=None)
    V_1 = fluid.layers.create_parameter(shape=[VOCAB_EMBEDDING_LENGTH, int(VOCAB_EMBEDDING_LENGTH/ATTENTION_HEAD_NUM)], 
                                                dtype='float32', 
                                                name='V_1',
                                                default_initializer=None)
    q1 = fluid.layers.mul(slots_embedding, Q_1)
    k1 = fluid.layers.mul(encoder_result, K_1)
    v1 = fluid.layers.mul(encoder_result, V_1)
    qk = fluid.layers.mul(q1, fluid.layers.t(k1))/fluid.layers.sqrt(VOCAB_EMBEDDING_LENGTH)
    head1 = fluid.layers.mul(fluid.layers.softmax(qk), v1)

    #calc
    states = fluid.layers.fc(head1, size = VOCAB_EMBEDDING_LENGTH, act='relu')
    # vocab = np.sum(np.array(state), axis=0)
    return states

def slot_gate(encoder_result, slots_embedding):

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

def calcu_cost(gates, gates_label, states, states_label):
    loss1 = fluid.layers.reduce_mean(fluid.layers.cross_entropy(gates, gates_label))
    loss2 = fluid.layers.reduce_mean(fluid.layers.square(states - states_label))

    loss = 0.0
    for gate_label in gates_label:
        if fluid.layers.argmax(gates_label).numpy()[0] == 1:
            loss += loss1 + loss2
        else:
            loss += loss1
    
    # value_tensor = fluid.Tensor()
    # value_tensor.set([1,0,0,0])

    return loss


def mymodel():
    slots_holder = fluid.data('slots_holder', shape=[ALL_SLOT_NUM, SLOT_EMBEDDING_LENGTH], dtype='float32')
    
    gates_holder_y = fluid.data('gates_holder_y', shape=[ALL_SLOT_NUM, GATE_KIND], dtype='int32')
    state_holder_y = fluid.data('state_holder_y', shape=[ALL_SLOT_NUM, VOCAB_EMBEDDING_LENGTH], dtype='float32')

    encoder_result = utterance_encoder()
    states = state_generator(encoder_result, slots_holder)
    gates = slot_gate(encoder_result, slots_holder)

    return gates, states


def train_program(data):
    slots = np.array()
    utterances = np.array()


    
    

place = fluid.CUDAPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program)
feeder = fluid.DataFeeder(feed_list = ['sentence_holder', 'slots_holder'],
                            place=place)

w = get_embedding_dict(EMBEDDIND_FILE_NAME) #cost lot of time

