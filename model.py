# -*-coding:utf-8 -*-

import numpy as np 
import paddle as pd
import paddle.fluid as fluid
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

HISTR_LENGTH = 32
SENTENCE_LENGTH = 128
VOCAB_EMBEDDING_LENGTH = 256
SLOT_EMBEDDING_LENGTH = 128

UTTR_TOKEN_LENGTH = HISTR_LENGTH * SENTENCE_LENGTH

ENCODER_HIDDEN_SIZE = 64
ENCODER_LAYERS_NUM = 1


all_slot = load_all_slot()
ALL_SLOT_NUM = len(all_slot)
SLOT_DO_KINDS_NUM = 4
GATE_KIND = 4
GATE_INDEX = ['UPDATE', 'DONTCARE', 'NONE', 'DELETE']


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

    # gen_out, gen_last_h, gen_last_c = fluid.layers.lstm(input=encoder_result,
    #                                                     )
    slots_to_hidden = fluid.layers.create_parameter(shape=[SLOT_EMBEDDING_LENGTH, UTTR_TOKEN_LENGTH], 
                                                    dtype='float32', 
                                                    name='slots_to_hidden',
                                                    default_initializer=None)
    hidden_to_vocab = fluid.layers.create_parameter(shape=[ENCODER_HIDDEN_SIZE, VOCAB_EMBEDDING_LENGTH], 
                                                dtype='float32', 
                                                name='hidden_to_vocab',
                                                default_initializer=None)
    slots = fluid.layers.mul(slots_embedding, slots_to_hidden)
    hiddens = fluid.layers.mul(encoder_result, hidden_to_vocab)
    vocabs = fluid.layers.mul(slots, hiddens)
    return []

def slot_gate(encoder_result, slots_embedding):

    # slots_to_hidden = fluid.layers.fc(slots_embedding, size=HISTR_LENGTH * SENTENCE_LENGTH,act='relu')
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

def update_slots(vocabs, gates):
    gates = np.array(fluid.layers.argmax(gates,axis=-1))
    vocabs = np.array(vocabs)
    for i in range(ALL_SLOT_NUM):
        if GATE_INDEX[gates[i]] == 'UPDATE':
            pass
        elif GATE_INDEX[gates[i]] == 'DONTCARE':
            pass
        elif GATE_INDEX[gates[i]] == 'NONE':
            pass
        else:
            pass
