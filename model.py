# -*-coding:utf-8 -*-

import numpy as np 
import paddle as pd
import paddle.fluid as fluid
from collections import OrderedDict
import math

from creative_data import *
from myutils import *

PASS_NUM = 5


HISTR_TURNS_LENGTH = 2
SENTENCE_LENGTH = 32
VOCAB_EMBEDDING_LENGTH = 128
SLOT_EMBEDDING_LENGTH = 128

UTTR_TOKEN_LENGTH = 128

ENCODER_HIDDEN_SIZE = 64
ENCODER_LAYERS_NUM = 1
ATTENTION_HEAD_NUM = 1
SLOTGATE_HEAD_NUM = 1


all_slot = load_all_slot()
ALL_SLOT_NUM = len(all_slot)
values_list = load_slot_value_list()
SLOT_VALUE_NUM = len(values_list) + 1
GATE_KIND = 4
GATE_INDEX = ['UPDATE', 'DONTCARE', 'NONE', 'DELETE']
SLOT_GATE_HIDDEN_SIZE = 64
save_model_name = 'first_model.mdl'

def utterance_encoder(sentences, dict_size):

    emb = fluid.embedding(input=sentences,
                            size=[dict_size, VOCAB_EMBEDDING_LENGTH])

    # init_h = fluid.layers.fill_constant(shape=[ENCODER_LAYERS_NUM, UTTR_TOKEN_LENGTH, ENCODER_HIDDEN_SIZE], dtype='float32',value=0.0)
    # init_c = fluid.layers.fill_constant(shape=[ENCODER_LAYERS_NUM, UTTR_TOKEN_LENGTH, ENCODER_HIDDEN_SIZE], dtype='float32',value=0.0)
    
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

    G_Q_1 = fluid.layers.create_parameter(shape=[SLOT_EMBEDDING_LENGTH, int(SLOT_VALUE_NUM/ATTENTION_HEAD_NUM)], 
                                                dtype='float32', 
                                                name='G_Q_1',
                                                default_initializer=None)
    G_K_1 = fluid.layers.create_parameter(shape=[ENCODER_HIDDEN_SIZE, int(SLOT_VALUE_NUM/ATTENTION_HEAD_NUM)], 
                                                dtype='float32', 
                                                name='G_K_1',
                                                default_initializer=None)
    G_V_1 = fluid.layers.create_parameter(shape=[ENCODER_HIDDEN_SIZE, int(SLOT_VALUE_NUM/ATTENTION_HEAD_NUM)], 
                                                dtype='float32', 
                                                name='G_V_1',
                                                default_initializer=None)
    q1 = fluid.layers.mul(slots_embedding, G_Q_1)
    k1 = fluid.layers.mul(encoder_result, G_K_1)
    v1 = fluid.layers.mul(encoder_result, G_V_1)

    qk = fluid.layers.mul(q1, fluid.layers.t(k1))# /math.sqrt(SLOT_VALUE_NUM)
    head1 = fluid.layers.mul(fluid.layers.softmax(qk), v1)

    #calc
    states = fluid.layers.fc(head1, size = SLOT_VALUE_NUM, act='softmax')

    return states

# def slot_gate(encoder_result, slots_embedding):

#     encoder_result = fluid.layers.reshape(encoder_result, shape=[UTTR_TOKEN_LENGTH, ENCODER_HIDDEN_SIZE])

#     slots_to_hidden = fluid.layers.create_parameter(shape=[SLOT_EMBEDDING_LENGTH, UTTR_TOKEN_LENGTH], 
#                                                     dtype='float32', 
#                                                     name='slots_to_hidden',
#                                                     default_initializer=None)
#     hidden_to_gate = fluid.layers.create_parameter(shape=[ENCODER_HIDDEN_SIZE, GATE_KIND], 
#                                                 dtype='float32', 
#                                                 name='hidden_to_gate',
#                                                 default_initializer=None)
#     slots = fluid.layers.tanh(fluid.layers.mul(slots_embedding, slots_to_hidden))
#     hiddens = fluid.layers.tanh(fluid.layers.mul(encoder_result ,hidden_to_gate))
#     gates = fluid.layers.softmax(fluid.layers.mul(slots, hiddens))
#     return gates

def slot_gate(encoder_result, slots_embedding):

    encoder_result = fluid.layers.reshape(encoder_result, shape=[UTTR_TOKEN_LENGTH, ENCODER_HIDDEN_SIZE])

    S_Q_1 = fluid.layers.create_parameter(shape=[SLOT_EMBEDDING_LENGTH, int(SLOT_VALUE_NUM/SLOTGATE_HEAD_NUM)], 
                                                dtype='float32', 
                                                name='S_Q_1',
                                                default_initializer=None)
    S_K_1 = fluid.layers.create_parameter(shape=[SLOT_GATE_HIDDEN_SIZE, int(SLOT_VALUE_NUM/SLOTGATE_HEAD_NUM)], 
                                                dtype='float32', 
                                                name='S_K_1',
                                                default_initializer=None)
    S_V_1 = fluid.layers.create_parameter(shape=[SLOT_GATE_HIDDEN_SIZE, int(SLOT_VALUE_NUM/SLOTGATE_HEAD_NUM)], 
                                                dtype='float32', 
                                                name='S_V_1',
                                                default_initializer=None)
    
    q1 = fluid.layers.mul(slots_embedding, S_Q_1)
    k1 = fluid.layers.mul(encoder_result, S_K_1)
    v1 = fluid.layers.mul(encoder_result, S_V_1)

    qk = fluid.layers.mul(q1, fluid.layers.t(k1))# /math.sqrt(SLOT_VALUE_NUM)
    head1 = fluid.layers.mul(fluid.layers.softmax(qk), v1)

    #calc
    gates = fluid.layers.fc(head1, size = ALL_SLOT_NUM, act='softmax')

    return gates

def optimizer_program():
    return fluid.optimizer.Adagrad(learning_rate=0.001)

def get_single_turn_cost(gates, gates_label, state, state_label):
    loss1 = fluid.layers.mean(fluid.layers.cross_entropy(gates, gates_label))
    # loss2 = fluid.layers.reduce_mean(fluid.layers.reduce_sum(fluid.layers.elementwise_mul(fluid.layers.log(-state), state_label), dim = 1))
    loss2 = fluid.layers.mean(fluid.layers.cross_entropy(state, state_label))
    return loss1 + loss2

# def get_single_turn_acc(gates, gates_label):

#     return fluid.layers.accuracy(input=gates, label=gates_label)

def get_ok_slot_num(gates, gates_label):

    # return fluid.layers.accuracy(input=gates, label=gates_label)
    # gates_label = fluid.layers.reshape(gates_label, shape=[ALL_SLOT_NUM])
    ok_slot_num = fluid.layers.reduce_sum(fluid.layers.cast(
            fluid.layers.equal(fluid.layers.argmax(gates, axis=1), gates_label), 
            dtype='int64'))
        
    return ok_slot_num



def mymodel(dict_size):
    sentences_index_holder = fluid.data(name='sentences_index_holder',
                                shape=[UTTR_TOKEN_LENGTH],
                                dtype='int64')

    slots_index_holder = fluid.data(name='slots_index_holder',
                                    shape=[ALL_SLOT_NUM],
                                    dtype='int64')
    slots_emb = fluid.embedding(input=slots_index_holder,
                                size=[ALL_SLOT_NUM, SLOT_EMBEDDING_LENGTH]
                                )

    encoder_result= utterance_encoder(sentences_index_holder, dict_size)
    states= state_generator(encoder_result, slots_emb)
    gates = slot_gate(encoder_result, slots_emb)

    return gates, states

def single_turn_train_program(dict_size):

    gates_label = fluid.data('gates_label', shape=[ALL_SLOT_NUM], dtype='int64')
    state_label= fluid.data('state_label', shape=[ALL_SLOT_NUM], dtype='int64')

    gates_predict, state_predict = mymodel(dict_size)
    single_turn_cost = get_single_turn_cost(gates=gates_predict, 
                    gates_label=gates_label, 
                    state=state_predict, 
                    state_label=state_label)
    # single_turn_acc = get_single_turn_acc(gates=gates_predict, gates_label=gates_label)
    ok_slot_num = get_ok_slot_num(gates=gates_predict, gates_label=gates_label)
    
    return gates_predict, single_turn_cost, ok_slot_num

word_dict = pd.dataset.imdb.word_dict()

gates_predict, single_turn_cost, ok_slot_num = single_turn_train_program(len(word_dict))
optimizer = optimizer_program()
optimizer.minimize(single_turn_cost)

place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
main_program = fluid.default_main_program()

# tokens_file = open('tokens.txt', mode='w+', encoding='utf-8')
# index_file = open('index.txt', mode='w+', encoding='utf-8')

train_dias, test_dias = load_diag_data(train_samples_num=300, 
                                        test_saples_num=20,
                                        SNG=True)
slots_feed_data = slots_attr2index()

def train_test(train_test_program, test_data):
    acc_set = []
    avg_loss_set = []
    all_turns = 0
    dia_acc = 0.0
    for dia_name, dia_data in test_data.items():
        dia_cost = 0.0

        dia_sentence_data = dia_data['dia_sentence_data']
        dia_gate_data = dia_data['dia_gate_data']
        dia_state_data = dia_data['dia_state_data']

        turns = len(dia_sentence_data)
        for i in range(turns):
            sentences_feed_data = dia_sentence_data[i]
            gates_feed_data = dia_gate_data[i]
            state_feed_data = dia_state_data[i]
            myfeed = {
                'sentences_index_holder':sentences_feed_data,
                'slots_index_holder':slots_feed_data,
                'gates_label':gates_feed_data,
                'state_label':state_feed_data
            }

            cost1, ok_slot_num1= exe.run(train_test_program,
                            feed = myfeed,
                            fetch_list=[single_turn_cost, ok_slot_num],
                            )
            # dia_cost += cost1
            if ok_slot_num1  == ALL_SLOT_NUM:
                dia_acc += 1
        all_turns += turns
    print('test acc: %f'%(dia_acc/all_turns))
    return dia_acc/all_turns

train_dias_data = get_feed_data(train_dias,
                            hist_turn_length=HISTR_TURNS_LENGTH,
                            uttr_token_length= UTTR_TOKEN_LENGTH,
                            word_dict=word_dict,
                            values_list=values_list)
#train
for epoch in range(PASS_NUM):

    all_turns = 0
    dia_acc = 0.0
    dia_cost = 0.0
    for dia_name, dia_data in train_dias_data.items():

        dia_sentence_data = dia_data['dia_sentence_data']
        dia_gate_data = dia_data['dia_gate_data']
        dia_state_data = dia_data['dia_state_data']

        turns = len(dia_sentence_data)
        for i in range(turns):
            sentences_feed_data = dia_sentence_data[i]
            gates_feed_data = dia_gate_data[i]
            state_feed_data = dia_state_data[i]
            myfeed = {
                'sentences_index_holder':sentences_feed_data,
                'slots_index_holder':slots_feed_data,
                'gates_label':gates_feed_data,
                'state_label':state_feed_data
            }

            cost1, ok_slot_num1 = exe.run(main_program,
                            feed = myfeed,
                            fetch_list=[single_turn_cost, ok_slot_num],
                            )
            # print(ok_slot_num1)
            # print(abc1)
            dia_cost += cost1
            # if 1.0 - acc1 < 1e-1:
            if ok_slot_num1 == ALL_SLOT_NUM:
                dia_acc += 1
        all_turns += turns 
    # print('cost: %f, acc: %f'%(dia_cost/turns, dia_acc/turns))
        # print(dia_name)
    print('epoch: %d, avg_cost: %f, avg_acc: %f' %(epoch,dia_cost/all_turns,dia_acc/all_turns))

    # test_data = get_feed_data(test_dias, )
    test_data = get_feed_data(test_dias,
                            hist_turn_length=HISTR_TURNS_LENGTH,
                            uttr_token_length= UTTR_TOKEN_LENGTH,
                            word_dict=word_dict,
                            values_list=values_list)
    train_test(main_program, test_data)