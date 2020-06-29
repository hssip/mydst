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
# SENTENCE_LENGTH = 32
VOCAB_EMBEDDING_LENGTH = 256
SLOT_EMBEDDING_LENGTH = 128

UTTR_TOKEN_LENGTH = 32

ENCODER_HIDDEN_SIZE = 256
ENCODER_LAYERS_NUM = 1
ATTENTION_HEAD_NUM = 1
SLOTGATE_HEAD_NUM = 1

lr = 0.001


all_slot = load_all_slot()
ALL_SLOT_NUM = len(all_slot)
values_list = load_slot_value_list()
SLOT_VALUE_NUM = len(values_list) + 3
GATE_KIND = 4
GATE_INDEX = ['UPDATE', 'DONTCARE', 'NONE', 'DELETE']
SLOT_GATE_HIDDEN_SIZE = 64
save_model_name = 'first_model.mdl'

def utterance_encoder(sentences, dict_size):

    emb = fluid.embedding(input=sentences,
                            size=[dict_size, VOCAB_EMBEDDING_LENGTH],
                            param_attr=fluid.ParamAttr(
                                name='word_embs',
                                initializer=fluid.initializer.Normal(0., VOCAB_EMBEDDING_LENGTH**-0.5)
                            ))
    # emb = fluid.layers.so

    # init_h = fluid.layers.fill_constant(shape=[ENCODER_LAYERS_NUM, UTTR_TOKEN_LENGTH, ENCODER_HIDDEN_SIZE], dtype='float32',value=0.0)
    # init_c = fluid.layers.fill_constant(shape=[ENCODER_LAYERS_NUM, UTTR_TOKEN_LENGTH, ENCODER_HIDDEN_SIZE], dtype='float32',value=0.0)
    
    # emb = fluid.layers.reshape(x = emb,
                                # shape=[None, None, VOCAB_EMBEDDING_LENGTH])

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
    encode_out = fluid.layers.reshape(encode_out, shape=[-1, ENCODER_HIDDEN_SIZE])
    encode_out = fluid.layers.fc(encode_out, size=ENCODER_HIDDEN_SIZE, act='tanh')
    
    return encode_out

def state_generator(encoder_result, slots_embedding):

    # encoder_result = fluid.layers.reshape(encoder_result, shape=[-1, ENCODER_HIDDEN_SIZE])

    G_Q_1 = fluid.layers.create_parameter(shape=[SLOT_EMBEDDING_LENGTH, int(SLOT_VALUE_NUM/ATTENTION_HEAD_NUM)], 
                                                dtype='float32', 
                                                name='G_Q_1',
                                                attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(0., 2.)))
    G_K_1 = fluid.layers.create_parameter(shape=[ENCODER_HIDDEN_SIZE, int(SLOT_VALUE_NUM/ATTENTION_HEAD_NUM)], 
                                                dtype='float32', 
                                                name='G_K_1',
                                                attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(0., 2.)))
    G_V_1 = fluid.layers.create_parameter(shape=[ENCODER_HIDDEN_SIZE, int(SLOT_VALUE_NUM/ATTENTION_HEAD_NUM)], 
                                                dtype='float32', 
                                                name='G_V_1',
                                                attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(0., 2.)))
    q1 = fluid.layers.mul(slots_embedding, G_Q_1)
    k1 = fluid.layers.mul(encoder_result, G_K_1)
    v1 = fluid.layers.mul(encoder_result, G_V_1)

    qk = fluid.layers.mul(q1, fluid.layers.t(k1))# /math.sqrt(SLOT_VALUE_NUM)
    sqk = fluid.layers.softmax(qk)
    head1 = fluid.layers.mul(sqk, v1)

    #calc
    states = fluid.layers.fc(head1, size = SLOT_VALUE_NUM, act='softmax')

    return states

def slot_gate(encoder_result, slots_embedding):

    # encoder_result = fluid.layers.reshape(encoder_result, shape=[-1, ENCODER_HIDDEN_SIZE])

    # S_Q_1 = fluid.layers.create_parameter(shape=[SLOT_EMBEDDING_LENGTH, SLOT_GATE_HIDDEN_SIZE], 
    #                                             dtype='float32', 
    #                                             name='S_Q_1',
    #                                             attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(0., 2.)))
    # S_K_1 = fluid.layers.create_parameter(shape=[ENCODER_HIDDEN_SIZE, SLOT_GATE_HIDDEN_SIZE], 
    #                                             dtype='float32', 
    #                                             name='S_K_1',
    #                                             attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(0., 2.)))
    # S_V_1 = fluid.layers.create_parameter(shape=[ENCODER_HIDDEN_SIZE, SLOT_GATE_HIDDEN_SIZE], 
    #                                             dtype='float32', 
    #                                             name='S_V_1',
    #                                             attr=fluid.ParamAttr(initializer=fluid.initializer.Normal(0., 2.)))   
    # q1 = fluid.layers.mul(slots_embedding, S_Q_1)
    # k1 = fluid.layers.mul(encoder_result, S_K_1)
    # v1 = fluid.layers.mul(encoder_result, S_V_1)
    # qk = fluid.layers.mul(q1, fluid.layers.t(k1))# /math.sqrt(SLOT_VALUE_NUM)
    # sqk = fluid.layers.softmax(qk)
    # head1 = fluid.layers.mul(sqk, v1)
    # gates = fluid.layers.fc(head1, size = GATE_KIND, act='softmax')
    contex_list = []
    for i in range(ALL_SLOT_NUM):
        slot_embedding = slots_embedding[i]
        slot_embedding = fluid.layers.expand_as(fluid.layers.unsqueeze(slot_embedding, axes=[0]), encoder_result)
        score_ = score_ = fluid.layers.elementwise_mul(slot_embedding, encoder_result)
        score = fluid.layers.softmax(score_)
        contex_vec = fluid.layers.unsqueeze(fluid.layers.reduce_sum(fluid.layers.elementwise_mul(score, encoder_result), dim=0),
                                            axes=[0])
        contex_list.append(contex_vec)
    contex_ = fluid.layers.concat(contex_list,axis=0)
    gates = fluid.layers.fc(contex_, size=GATE_KIND, act='softmax')

    return gates #, qk, sqk

def optimizer_program():
    return fluid.optimizer.SGD(learning_rate=lr)

def get_single_turn_cost(gates, gates_label, state, state_label):
    loss1 = fluid.layers.reduce_max(fluid.layers.cross_entropy(gates, gates_label))
    loss2 = fluid.layers.reduce_max(fluid.layers.cross_entropy(state, state_label))
    return loss1 + loss2

def get_ok_slot_num(gates, gates_label, states, states_label):
    ok_slot = fluid.layers.cast(
            fluid.layers.equal(fluid.layers.argmax(gates, axis=1), gates_label), 
            dtype='int64')
    ok_value = fluid.layers.cast(fluid.layers.equal(fluid.layers.argmax(states, axis=1), states_label), 
            dtype='int64')
    ok_slot_num = fluid.layers.reduce_sum(fluid.layers.elementwise_mul(ok_slot, ok_value))
        
    return ok_slot_num

def get_slot_acc(gates, gates_label, states, states_label):
    
    leng = len(gates)
    ok_num = 0
    arg_gate = np.argmax(gates, axis=1)
    arg_states = np.argmax(states, axis=1)
    for i in range(leng):
        if arg_gate[i] == gates_label[i] and arg_states[i] == states_label[i]:
            ok_num +=1
    
    return ok_num/leng



def mymodel(dict_size):
    sentences_index_holder = fluid.data(name='sentences_index_holder',
                                shape=[1, None],
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

    gates_predict, states_predict = mymodel(dict_size)
    single_turn_cost = get_single_turn_cost(gates=gates_predict, 
                    gates_label=gates_label, 
                    state=states_predict, 
                    state_label=state_label)
    # single_turn_acc = get_single_turn_acc(gates=gates_predict, gates_label=gates_label)
    ok_slot_num = get_ok_slot_num(gates=gates_predict, 
                                    gates_label=gates_label,
                                    states=states_predict,
                                    states_label=state_label)
    # slot_acc = get_slot_acc(gates=gates_predict, gates_label=gates_label)
    
    return gates_predict, single_turn_cost, ok_slot_num, states_predict

word_dict = pd.dataset.imdb.word_dict()

gates_predict, single_turn_cost, ok_slot_num, state_predict = single_turn_train_program(len(word_dict))
optimizer = optimizer_program()
optimizer.minimize(single_turn_cost)

# place = fluid.CUDAPlace(0)
place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
main_program = fluid.default_main_program()

train_dias, test_dias = load_diag_data(train_samples_num=300, 
                                        test_saples_num=20,
                                        SNG=True)
train_dias_data = get_feed_data(train_dias,
                            hist_turn_length=HISTR_TURNS_LENGTH,
                            uttr_token_length= UTTR_TOKEN_LENGTH,
                            word_dict=word_dict,
                            values_list=values_list,
                            kind='train')
test_dias_data = get_feed_data(test_dias,
                        hist_turn_length=HISTR_TURNS_LENGTH,
                        uttr_token_length= UTTR_TOKEN_LENGTH,
                        word_dict=word_dict,
                        values_list=values_list,
                        kind='test')
print('load data and save data ok, begin to train')

def train_test(train_test_program, test_data):
    print('begin test!')
    avg_loss_set = []
    dia_acc = 0.0
    all_slot_acc = 0.0
    temp1 = []
    temp2 = []
    np.random.shuffle(test_data)
    for dia_data in test_data:
        dia_cost = 0.0

        sentences_feed_data = np.array(dia_data[0])
        slots_feed_data = np.array(dia_data[1])
        gates_feed_data = np.array(dia_data[2])
        state_feed_data = np.array(dia_data[3])
        myfeed = {
            'sentences_index_holder':sentences_feed_data,
            'slots_index_holder':slots_feed_data,
            'gates_label':gates_feed_data,
            'state_label':state_feed_data
        }

        gates_predict1, cost1, ok_slot_num1, state_predict1,= exe.run(train_test_program,
                        feed = myfeed,
                        fetch_list=[gates_predict,single_turn_cost, ok_slot_num, state_predict]
                        )
        # dia_cost += cost1
        temp1.append(gates_predict1)
        temp2.append(gates_feed_data)
            # save_predict(gates_predict1, gates_feed_data, 'test')
        if ok_slot_num1  == ALL_SLOT_NUM:
            dia_acc += 1
        slot_acc1 = get_slot_acc(gates_predict1, gates_feed_data, state_predict1, state_feed_data)
        all_slot_acc+=slot_acc1
        # all_turns += turns
    save_predict(temp1, temp2, kind='test')
    print('test joint_acc: %f, slot_acc: %f'%(dia_acc/dia_num, all_slot_acc/dia_num))
    show_f1(temp1, temp2)
    return dia_acc/dia_num

feeder = fluid.DataFeeder(feed_list=['sentences_index_holder','slots_index_holder',
                                        'gates_label','state_label'],
                        place=place)

#train
for epoch in range(PASS_NUM):

    dia_acc = 0.0
    dia_cost = 0.0
    all_slot_acc = 0.0
    np.random.shuffle(train_dias_data)
    np.random.shuffle(test_dias_data)
    temp1 = []
    temp2 = []
    for dia_num, dia_data in enumerate(train_dias_data):
        sentences_feed_data = np.array(dia_data[0])
        slots_feed_data = np.array(dia_data[1])
        gates_feed_data = np.array(dia_data[2])
        state_feed_data = np.array(dia_data[3])
        myfeed = {
            'sentences_index_holder':np.array(dia_data[0]),
            'slots_index_holder':np.array(dia_data[1]),
            'gates_label':np.array(dia_data[2]),
            'state_label':np.array(dia_data[3])
        }

        gates_predict1, cost1, ok_slot_num1, state_predict1 = exe.run(main_program,
                        feed = myfeed,
                        fetch_list=[gates_predict, single_turn_cost, ok_slot_num, state_predict]
                        )
        temp1.append(gates_predict1)
        temp2.append(gates_feed_data)
        dia_cost += cost1
        if ok_slot_num1 == ALL_SLOT_NUM:
            dia_acc += 1
        slot_acc1 = get_slot_acc(gates_predict1, gates_feed_data, state_predict1, state_feed_data)
        all_slot_acc += slot_acc1
        
        # print(encoder_result1.tolist())
        # show_f1(temp1, temp2)
            # if dia_num == 0 and i < 4:
            # t_file = open('temp.txt', mode='a+')
            # t_file.write('qk :' + str(qk1.tolist()) + '\n')
            # t_file.write('sqk:'+ str(sqk1.tolist()) + '\n')
            # t_file.close()
        # all_turns += turns 
        if dia_num % 100 == 0 and dia_num != 0:
            print('%d turn, avg_cost: %f, avg_joint_acc: %f, slot_acc: %f' %(dia_num, dia_cost/dia_num,dia_acc/dia_num, all_slot_acc/dia_num))
            
            # show_f1(temp1, temp2)
    # print('etetetetet-----------' + str(all_turns))
    save_predict(temp1, temp2, kind='train')
    print('epoch: %d, avg_cost: %f, avg_acc: %f, slot_acc: %f' %(epoch,dia_cost/dia_num,dia_acc/dia_num, all_slot_acc/dia_num))
    show_f1(temp1, temp2)
    train_test(main_program, test_dias_data)