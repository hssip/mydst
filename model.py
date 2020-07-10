# -*-coding:utf-8 -*-

from collections import OrderedDict
import math, os

import numpy as np 
import paddle as pd
import paddle.fluid as fluid

from myutils import *


PASS_NUM = 5
BATCH_SIZE = 128
VOCAB_EMBEDDING_LENGTH = 400
SLOT_EMBEDDING_LENGTH = 400
ENCODER_HIDDEN_SIZE = 400
ENCODER_LAYERS_NUM = 1

lr = 0.001
all_slot = load_all_slot()
ALL_SLOT_NUM = len(all_slot)
values_list = load_slot_value_list()
SLOT_VALUE_NUM = len(values_list) + 3

DOMAIN_KIND = len(domain_list)
GATE_INDEX = ['UPDATE', 'DONTCARE', 'NONE']
GATE_KIND = len(GATE_INDEX)

# save_model_name = 'first_model.mdl'

word_dict = {'':0}
train_dias_data = get_feed_data(word_dict=word_dict, data_kind='train', batch_size=BATCH_SIZE)
test_dias_data = get_feed_data(word_dict=word_dict, data_kind='test', batch_size=BATCH_SIZE)

# save_input(train_dias_data, kind='train')
# save_input(test_dias_data, kind='test')
print('load data and save data ok, begin to train')


def utterance_encoder(sentences, dict_size):

    emb = fluid.embedding(input=sentences,
                        size=[dict_size, VOCAB_EMBEDDING_LENGTH],
                        param_attr=fluid.ParamAttr(
                        name='word_embs_new',
                        initializer=fluid.initializer.Normal(0., VOCAB_EMBEDDING_LENGTH**-0.5)))
    cell = fluid.layers.GRUCell(hidden_size=ENCODER_HIDDEN_SIZE)
    # cell = fluid.layers.LSTMCell(hidden_size=ENCODER_HIDDEN_SIZE)
    r_cell = fluid.layers.GRUCell(hidden_size=ENCODER_HIDDEN_SIZE)

    encode_out, encode_last_h = fluid.layers.rnn(cell=cell,
                                                inputs=emb)
    # print('encoder_out111_shape: %s' %(str(encode_out.shape)))
    r_encode_out, r_encode_last_h = fluid.layers.rnn(cell=r_cell,
                                                    inputs=emb,
                                                    is_reverse=True)
    # encode_out = fluid.layers.reshape(encode_out, shape=[-1, ENCODER_HIDDEN_SIZE])
    # encode_out = fluid.layers.fc(encode_out, size=ENCODER_HIDDEN_SIZE, act='tanh')
    encode_out = encode_out + r_encode_out
    encode_last_h = encode_last_h + r_encode_last_h
    
    
    return encode_out, encode_last_h

def domain_gate(encode_out):
    # encode_out1 = fluid.layers.reshape(x=encode_out, shape=[-1, ENCODER_HIDDEN_SIZE])
    socre = fluid.layers.reduce_mean(encode_out, dim=1)
    domain_value = fluid.layers.fc(socre, size=DOMAIN_KIND, act='softmax')
    # domain_value = fluid.layers.fc(encode_last_h, size=DOMAIN_KIND, act='softmax')
    return domain_value

def state_generator(encoder_result, slots_embedding):

    encoder_result = fluid.layers.reshape(encoder_result, shape=[-1, ENCODER_HIDDEN_SIZE])

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

def slot_gate(encoder_outs, encoder_last_h, slots_embedding):
    slots_embedding = fluid.layers.reshape(x=slots_embedding, shape=[ALL_SLOT_NUM, SLOT_EMBEDDING_LENGTH])
    # print ('encoder_outs: %s,  encoder_last_h: %s,  slots_embedding: %s' % (str(encoder_outs.shape), str(encoder_last_h.shape), str(slots_embedding.shape)))
    slot_emb_cat = slots_embedding[0]
    for i in range(1, ALL_SLOT_NUM):
        # print ('slot_emb: %s' % str(slot_emb.shape))
        slot_emb = slots_embedding[i]
        slot_emb_cat = fluid.layers.concat(input=[slot_emb_cat, slot_emb], axis=0)
    # print ('slot_emb_cat: %s' % str(slot_emb_cat.shape))
    slot_emb_cat = fluid.layers.reshape(x=slot_emb_cat, shape=[-1, SLOT_EMBEDDING_LENGTH])    #(batch_size*|slots|, SLOT_EMBEDDING_LENGTH)
    slot_emb_cat = fluid.layers.dropout(slot_emb_cat, dropout_prob=0.15)
    dec_h = fluid.layers.expand(x=encoder_last_h, expand_times=[ALL_SLOT_NUM, 1])
    cell = fluid.layers.GRUCell(hidden_size=ENCODER_HIDDEN_SIZE)
    dec_outs, dec_last_h = cell(slot_emb_cat, dec_h)
    dec_outs = fluid.layers.unsqueeze(dec_outs, axes=[1])
    dec_outs = fluid.layers.expand(dec_outs, expand_times=[1, fluid.layers.shape(encoder_outs)[1], 1])
    # print ('dec_outs: %s,  dec_last_h: %s' % (str(dec_outs.shape), str(dec_last_h.shape)))

    enc_outs = fluid.layers.unsqueeze(encoder_outs, 1)
    enc_outs = fluid.layers.expand(enc_outs, expand_times=[1, ALL_SLOT_NUM, 1, 1])
    scores = fluid.layers.softmax(
                fluid.layers.reduce_sum(
                fluid.layers.elementwise_mul(enc_outs, dec_outs, axis=1),
                dim=-1))
    print ('scores: %s' % str(scores.shape))

    context = fluid.layers.reduce_sum(fluid.layers.elementwise_mul(enc_outs, scores, axis=0), dim=-2)
    print ('context: %s' % str(context.shape))

    # gate_probs = fluid.layers.reshape(x=context, shape=[ALL_SLOT_NUM, -1])
    # gate_probs = fluid.layers.fc(input=gate_probs, size=GATE_KIND, act='softmax')
    
    gate_probs = fluid.layers.fc(input=context, size=GATE_KIND, num_flatten_dims=2, act='softmax',
        param_attr=fluid.ParamAttr(
        learning_rate=0.1,
        trainable=True,
        name="cls_out_w",
        initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
        bias_attr=fluid.ParamAttr(
        name="cls_out_b", initializer=fluid.initializer.Constant(0.)))
    # print('gate_probs: %s' % str(gate_probs.shape))
    # gate_probs = fluid.layers.fc(input=gate_probs, size=GATE_KIND, act='softmax')
    gate_probs = fluid.layers.reshape(x=gate_probs, shape=[ALL_SLOT_NUM, GATE_KIND])
    # temp = fluid.layers.squeeze(input=encoder_outs, axes=[0])
    # slot2hidden = fluid.layers.fc(input=slots_embedding, size=ENCODER_HIDDEN_SIZE, act='sigmoid')
    # a_h = fluid.layers.expand(encoder_last_h,expand_times=[ALL_SLOT_NUM, 1])
    # lasth2hidden = fluid.layers.fc(input=a_h, size=ENCODER_HIDDEN_SIZE, act='tanh')
    # # print('encoder_out_shape: %s' %(str(encoder_outs.shape)))
    # re_out2hidden = fluid.layers.fc(input=fluid.layers.squeeze(encoder_outs, axes=[0]), size=ENCODER_HIDDEN_SIZE, act='tanh')

    # out2hidden = fluid.layers.t(fluid.layers.fc(re_out2hidden, size=ALL_SLOT_NUM, act='tanh'))
    # gate_probs = fluid.layers.elementwise_mul(slot2hidden, fluid.layers.elementwise_add(lasth2hidden, out2hidden))

    # encoder_sum = fluid.layers.reduce_sum(input = encoder_outs, dim = 1)
    # gate_probs =fluid.layers.fc(input = encoder_sum,size=GATE_KIND, act='softmax')
    # gate_probs = fluid.layers.softmax(gate_probs, axis=-1)
    
    # gate_probs
    return gate_probs #, qk, sqk

def optimizer_program():
    return fluid.optimizer.SGD(learning_rate=lr)

# def get_single_turn_cost(gates, gates_label, state, state_label, domain, domain_label):
#     # loss1 = fluid.layers.reduce_mean(fluid.layers.cross_entropy(gates, gates_label))
#     # loss2 = fluid.layers.reduce_max(fluid.layers.cross_entropy(state, state_label))
#     # domain = fluid.layers.squeeze(domain, axes=[0])
#     domain_label = fluid.layers.unsqueeze(domain_label, axes=[1])
#     loss3 = fluid.layers.reduce_sum(fluid.layers.cross_entropy(domain, domain_label))
#     return loss3
def get_single_turn_cost(domain, domain_label):
    # loss1 = fluid.layers.reduce_mean(fluid.layers.cross_entropy(gates, gates_label))
    # loss2 = fluid.layers.reduce_max(fluid.layers.cross_entropy(state, state_label))
    # domain = fluid.layers.squeeze(domain, axes=[0])
    domain_label = fluid.layers.unsqueeze(domain_label, axes=[1])
    loss3 = fluid.layers.reduce_sum(fluid.layers.cross_entropy(domain, domain_label))
    return loss3

def get_ok_slot_num(gates, gates_label, states, states_label):
    ok_slot = fluid.layers.cast(
            fluid.layers.equal(fluid.layers.argmax(gates, axis=1), gates_label), 
            dtype='int64')
    # ok_value = fluid.layers.cast(fluid.layers.equal(fluid.layers.argmax(states, axis=1), states_label), 
            # dtype='int64')
    # ok_slot_num = fluid.layers.reduce_sum(fluid.layers.elementwise_mul(ok_slot, ok_value))  
    ok_slot_num = fluid.layers.reduce_sum(ok_slot)
    return ok_slot_num

def get_domain_acc(domain, domain_label):
    # domain = fluid.layers.unsqueeze()
    domain_arg = fluid.layers.argmax(domain, axis=1)
    # domain_label = fluid.layers.unsqueeze(domain_label,axes=[1])
    # print('domain_arg shape is: %s'%(str(domain_arg.shape)))
    domain_num = fluid.layers.reduce_mean(fluid.layers.cast(
        fluid.layers.equal(domain_arg, domain_label),
        dtype='int64'))
    
    # lenglen()
    # for i in range()
    # domain_arg = np.argmax(domain, axis=0)

    return domain_num

def get_slot_acc(gates, gates_label):#, states, states_label, domain, domain_label):
    # leng = len(gates)
    # ok_num = 0
    # arg_gate = np.argmax(gates, axis=1)
    # arg_states = np.argmax(states, axis=1)
    # for i in range(leng):
        # if arg_gate[i] == gates_label[i]:# and arg_states[i] == states_label[i]:
            # ok_num +=1
    # label = fluid.layers.unsqueeze(gates_label, axes=1)
    # acc = fluid.layers.accuracy(gates, label)
    gates_label = fluid.layers.unsqueeze(input=gates_label, axes=[1])
    acc = fluid.layers.accuracy(input=gates, label = gates_label)
    return acc

def mymodel(dict_size):
    sentences_index_holder = fluid.data(name='sentences_index_holder',
                                shape=[1, None],
                                dtype='int64')
    slots_index_holder = fluid.data(name='slots_index_holder',
                                    shape=[ALL_SLOT_NUM],
                                    dtype='int64')
    slots_emb = fluid.embedding(input=slots_index_holder,
                                size=[ALL_SLOT_NUM, SLOT_EMBEDDING_LENGTH],
                                param_attr=fluid.ParamAttr(
                                    name='slots_embs',
                                    learning_rate=0.1,
                                    initializer=fluid.initializer.Normal(0., SLOT_EMBEDDING_LENGTH**-0.5)))
    # slots_emb = fluid.layers.scale(x=slots_emb, scale=SLOT_EMBEDDING_LENGTH**-0.5)
    # slots_emb = fluid.layers.Print(input = slots_emb, message='slots_embedding', summarize=-1)
    
    encoder_result, encoder_last_h = utterance_encoder(sentences_index_holder, dict_size)
    # states= state_generator(encoder_result, slots_emb)
    # gates = slot_gate(encoder_result, encoder_last_h, slots_emb)
    domain = domain_gate(encode_out=encoder_result)

    # return gates, states, slots_emb, domain
    return domain

def single_turn_train_program(dict_size):

    gates_label = fluid.data('gates_label', shape=[ALL_SLOT_NUM], dtype='int64')
    state_label= fluid.data('state_label', shape=[ALL_SLOT_NUM], dtype='int64')
    domain_label = fluid.data('domain_label', shape=[None],dtype='int64')

    # gates_predict, states_predict, slots_emb, domain_predict = mymodel(dict_size)
    domain_predict = mymodel(dict_size)
    # single_turn_cost = get_single_turn_cost(gates=gates_predict, 
    #                 gates_label=gates_label, 
    #                 state=states_predict, 
    #                 state_label=state_label,
    #                 domain=domain_predict,
    #                 domain_label=domain_label)
    single_turn_cost = get_single_turn_cost(domain=domain_predict, domain_label=domain_label)
    # single_turn_acc = get_single_turn_acc(gates=gates_predict, gates_label=gates_label)
    # ok_slot_num = get_ok_slot_num(gates=gates_predict, 
    #                                 gates_label=gates_label,
    #                                 states=states_predict,
    #                                 states_label=state_label)
    # slot_acc = get_slot_acc(gates=gates_predict, gates_label=gates_label)
    domain_acc = get_domain_acc(domain=domain_predict, domain_label=domain_label)
    
    return single_turn_cost, domain_acc

single_turn_cost, domain_acc = single_turn_train_program(len(word_dict))
optimizer = optimizer_program()
optimizer.minimize(single_turn_cost)

place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())
main_program = fluid.default_main_program()

# place = fluid.CUDAPlace(int(os.getenv('FLAGS_selected_gpus', '0')))
# dev_count = fluid.core.get_cuda_device_count()
# print ('cuda used, dev_count=%d' % dev_count)
# exec_strategy = fluid.ExecutionStrategy()
# exec_strategy.use_experimental_executor = True
# exec_strategy.num_threads = 1
# train_exe = fluid.ParallelExecutor(
#     use_cuda=True,
#     loss_name=single_turn_cost.name,
#     exec_strategy=exec_strategy,
#     main_program=main_program)

# fetch_list = [gates_predict, single_turn_cost, ok_slot_num, state_predict, slot_acc]
fetch_list = [single_turn_cost, domain_acc]


def train_test(train_test_program, test_data):
    print('begin test!')
    # avg_loss_set = []
    # dia_acc = 0.0
    all_slot_acc = 0.0
    # temp1 = []
    # temp2 = []
    # np.random.shuffle(test_data)
    all_domain_num = 0
    for dia_num, dia_data in enumerate(test_data):
        # dia_cost = 0.0
        sentences_feed_data = np.array(dia_data[0])
        slots_feed_data = np.array(dia_data[1])
        gates_feed_data = np.array(dia_data[2])
        state_feed_data = np.array(dia_data[3])
        domain_feed_data = np.array(dia_data[4])
        myfeed = {
            'sentences_index_holder':sentences_feed_data,
            'slots_index_holder':slots_feed_data,
            'gates_label':gates_feed_data,
            'state_label':state_feed_data,
            'domain_label':domain_feed_data
        }
        cost1, domain_acc1 = exe.run(train_test_program,
                        feed = myfeed,
                        fetch_list=fetch_list
                        )
        
        # all_domain_num += domain_acc1
        # dia_cost += cost1
        # temp1.append(gates_predict1)
        # temp2.append(gates_feed_data)
            # save_predict(gates_predict1, gates_feed_data, 'test')
        # if ok_slot_num1  == ALL_SLOT_NUM:
            # dia_acc += 1
        # slot_acc1 = get_slot_acc(gates_predict1, gates_feed_data, state_predict1, state_feed_data)
        all_slot_acc+=domain_acc1
        # all_turns += turns
    # save_predict(temp1, temp2, kind='test')
    # print('test joint_acc: %f, slot_acc: %f'%(dia_acc/dia_num, all_slot_acc/dia_num))
    # show_f1(temp1, temp2)
    print('test domain acc: %f'%(all_slot_acc/dia_num))
    return dia_acc/dia_num

# feeder = fluid.DataFeeder(feed_list=['sentences_index_holder','slots_index_holder',
#                                         'gates_label','state_label'],
#                         place=place)

#train
for epoch in range(PASS_NUM):

    dia_acc = 0.0
    dia_cost = 0.0
    all_slot_acc = 0.0
    # all_domain_num = 0
    temp1 = []
    temp2 = []
    np.random.shuffle(train_dias_data)
    for dia_num, dia_data in enumerate(train_dias_data):
        sentences_feed_data = np.array(dia_data[0])
        slots_feed_data = np.array(dia_data[1])
        gates_feed_data = np.array(dia_data[2])
        state_feed_data = np.array(dia_data[3])
        domain_feed_data = np.array(dia_data[4])
        myfeed = {
            'sentences_index_holder':sentences_feed_data,
            'slots_index_holder':slots_feed_data,
            'gates_label':gates_feed_data,
            'state_label':state_feed_data,
            'domain_label':domain_feed_data
        }
        cost1, domain_acc1 = exe.run(main_program,
                        feed = myfeed,
                        fetch_list=fetch_list
                        )
        # fout = open('temp.txt', mode='a+', encoding='utf8')
        # stra = str(np.argmax(gates_predict1, 1))+'\n'
        # fout.write(stra)
        # fout.close()
        # print(domain_predict1)
        # print(domain_feed_data)
        # temp1.append(gates_predict1)
        # temp2.append(gates_feed_data)
        dia_cost += cost1
        # if ok_slot_num1 == ALL_SLOT_NUM:
        #     dia_acc += 1
        # slot_acc1 = get_slot_acc(gates_predict1, gates_feed_data, state_predict1, state_feed_data)
        # all_slot_acc += slot_acc1
            # if dia_num == 0 and i < 4:
        # t_file = open('temp.txt', mode='a+')
        # t_file.write('slots_emb :' + str(np.round(slots_emb1,decimals=15).tolist()) + '\n')
        # t_file.write('sqk:'+ str(sqk1.tolist()) + '\n')
        # t_file.close()
        # all_domain_num += domain_acc1
        all_slot_acc += domain_acc1
        if dia_num % 32 == 0 and dia_num!=0:
            # print('%d turn, avg_cost: %f, avg_joint_acc: %f, slot_acc: %f' %(dia_num, dia_cost/dia_num, dia_acc/dia_num, all_slot_acc/dia_num))  
            # show_f1(temp1, temp2)
            print('%d turn, avg_cost: %f, slot_acc: %f' %(dia_num, dia_cost/32.0, all_slot_acc/32.0))  
            all_slot_acc = 0.0
            dia_cost = 0.0
        # if dia_num % 100 == 0 and dia_num != 0:
            # save_predict(temp1, temp2, kind='train')
    # print('epoch: %d, avg_cost: %f, avg_acc: %f, slot_acc: %f' %(epoch,dia_cost/dia_num,dia_acc/dia_num, all_slot_acc/dia_num))
    # show_f1(temp1, temp2)
    # print('%d turn, avg_cost: %f, slot_acc: %f' %(dia_num, dia_cost/dia_num, all_slot_acc/dia_num))
    train_test(main_program, test_dias_data)