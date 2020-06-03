import numpy as np 
import paddle as pd
import paddle.fluid as fluid
from myutils import *

SENTENCE_VEC_LENGTH = 256
SLOT_VEC_LENGTH = 128
HISTR_HIDDEN_NUM = 64
HISTR_VEC_LENGTH = SLOT_VEC_LENGTH + SENTENCE_VEC_LENGTH
HISTR_LENGTH = 128
LAYERS_NUM = 1

# CURR_SENTENCE_MAX_LENGTH = 128
SENTENCE_MAX_LENGTH = 128

VOCAB_VEC_LENGTH = 256


def utterance_encoder():
    slot_holder = fluid.data(name='slot_holder', shape=[None, HISTR_LENGTH, SENTENCE_MAX_LENGTH, SLOT_VEC_LENGTH])
    sentence_holder = fluid.data(name='sentence_holder',shape=[None, HISTR_LENGTH, SENTENCE_VEC_LENGTH])
    histr_input = fluid.layers.concat([slot_holder, sentence_holder], axis=-1)
    init_h = fluid.layers.fill_constant(shape=[LAYERS_NUM, HISTR_LENGTH, HISTR_HIDDEN_NUM], dtype='float32',value=0.0)
    init_c = fluid.layers.fill_constant(shape=[LAYERS_NUM, HISTR_LENGTH, HISTR_HIDDEN_NUM], dtype='float32',value=0.0)

    rnn_out, last_h, last_c = fluid.layers.lstm(histr_input, 
                                                init_h=init_h,
                                                init_c=init_c,
                                                hidden_size=HISTR_HIDDEN_NUM,
                                                num_layers=LAYERS_NUM,
                                                is_bidirec=True,
                                                max_len=HISTR_VEC_LENGTH)                                        
# def curr_encoder():
    # curr_sentence = fluid.data(name='curr_sentence', shape=[None, SENTENCE_MAX_LENGTH, VOCAB_VEC_LENGTH])



