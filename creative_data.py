# -*-coding:utf-8 -*-

import json, pickle
import copy 
import os, re
from collections import OrderedDict
# import shutil
# import urllib.request
# from io import BytesIO
# from zipfile import ZipFile
# import difflib
import numpy as np 


FILE_PATH_PREFIX = 'pub_dataset/MultiWOZ_1.0/'
LOAD_DIAG_NUM = 40

ignore_in_goal = ['eod', 'topic', 'messageLen', 'message']

replace_lines = open('pub_dataset/mapping.pair', 'r').readlines()
replace_list = []
for line in replace_lines:
    tok_from, tok_to = line.replace('\n', '').split('\t')
    replace_list.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

domin_list = ['hotel', 'restaurant', 'taxi', 'attraction', 'train']
slots_list = [['name','type', 'parking', 'pricerange', 'internet', 'day', 'stay', 'people', 'area', 'stars'],
                ['food', 'pricerange', 'area', 'name', 'time', 'day', 'people'],
                ['leaveAt', 'destination', 'departure', 'arriveBy'], 
                ['name', 'type', 'area'],
                ['people', 'leaveAt', 'destination', 'day','arriveBy', 'departure']
                ]

# slots = {
#     'taxi':['leaveAt', 'destination', 'departure', 'arrivaBy'],
#     'hotel':['name','type', 'parking', 'pricerange', 'internet', 
#                 'day', 'stay', 'people', 'area', 'star'],
#     'attraction':['name', 'type', 'area'],
#     'train':['people', 'leaveAt', 'destination', 'day','arriveBy', 'departure'],
#     'restaurant' :['food', 'price', 'area', 'name', 'time', 'day', 'people']
# }

SLOTS_MAPPING_FILE_PATH = 'pub_dataset/slots_mapping'

def is_ascii(s):
    return all(ord(c) < 128 for c in s)

def insertSpace(token, text):
    sidx = 0
    while True:
        sidx = text.find(token, sidx)
        if sidx == -1:
            break
        if sidx + 1 < len(text) and re.match('[0-9]', text[sidx - 1]) and \
                re.match('[0-9]', text[sidx + 1]):
            sidx += 1
            continue
        if text[sidx - 1] != ' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx += 1
        if sidx + len(token) < len(text) and text[sidx + len(token)] != ' ':
            text = text[:sidx + 1] + ' ' + text[sidx + 1:]
        sidx += 1
    return text

def normalize(text, clean_value=True):
    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$', '', text)

    # hotel domain pfb30
    text = re.sub(r"b&b", "bed and breakfast", text)
    text = re.sub(r"b and b", "bed and breakfast", text)

    if clean_value:
        # normalize phone number
        ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4,5})', text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m[0], sidx)
                if text[sidx - 1] == '(':
                    sidx -= 1
                eidx = text.find(m[-1], sidx) + len(m[-1])
                text = text.replace(text[sidx:eidx], ''.join(m))

        # normalize postcode
        ms = re.findall('([a-z]{1}[\. ]?[a-z]{1}[\. ]?\d{1,2}[, ]+\d{1}[\. ]?[a-z]{1}[\. ]?[a-z]{1}|[a-z]{2}\d{2}[a-z]{2})',
                        text)
        if ms:
            sidx = 0
            for m in ms:
                sidx = text.find(m, sidx)
                eidx = sidx + len(m)
                text = text[:sidx] + re.sub('[,\. ]', '', m) + text[eidx:]

    # weird unicode bug
    text = re.sub(u"(\u2018|\u2019)", "'", text)

    # if clean_value:
        # replace time and and price
        # text = re.sub(timepat, ' [value_time] ', text)
        # text = re.sub(pricepat, ' [value_price] ', text)
        #text = re.sub(pricepat2, '[value_price]', text)

    # replace st.
    text = text.replace(';', ',')
    text = re.sub('$\/', '', text)
    text = text.replace('/', ' and ')

    # replace other special characters
    text = text.replace('-', ' ')
    text = re.sub('[\"\<>@\(\)]', '', text) # remove

    # insert white space before and after tokens:
    for token in ['?', '.', ',', '!']:
        text = insertSpace(token, text)

    # insert white space for 's
    text = insertSpace('\'s', text)

    # replace it's, does't, you'd ... etc
    text = re.sub('^\'', '', text)
    text = re.sub('\'$', '', text)
    text = re.sub('\'\s', ' ', text)
    text = re.sub('\s\'', ' ', text)
    for fromx, tox in replace_list:
        text = ' ' + text + ' '
        text = text.replace(fromx, tox)[1:-1]

    # remove multiple spaces
    text = re.sub(' +', ' ', text)

    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i < len(tokens):
        if re.match(u'^\d+$', tokens[i]) and \
                re.match(u'\d+$', tokens[i - 1]):
            tokens[i - 1] += tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)

    return text

def process_metadata(sys_metadata):
    result = OrderedDict()

    #change mutiword slot to single word
    # slots_mapping = open(file=SLOTS_MAPPING_FILE_PATH, mode='r')
    # slot_map = {}
    # for line in slots_mapping.readlines():
    #     arr = line.strip('\n').split(',')
    #     if arr:
    #         slot_map[arr[0]] = arr[1]
    #     else:
    #         print('teststststs')

    #change hierachical dict to single level orderdict

    for domin_index, domin in enumerate(domin_list):
        temp = OrderedDict()
        if domin not in sys_metadata:
            for slot in slots_list[domin_index]:
                temp[slot] = ''
            result[domin] = temp
            continue
        for slot in slots_list[domin_index]:
            if slot in sys_metadata[domin]['semi']:
                temp[slot] = sys_metadata[domin]['semi'][slot]
            elif slot in sys_metadata[domin]['book']:
                temp[slot] = sys_metadata[domin]['book'][slot]
            elif sys_metadata[domin]['book']['booked'] and \
                slot in sys_metadata[domin]['book']['booked'][0]:
                temp[slot] = sys_metadata[domin]['book']['booked'][0][slot]
            else:
                temp[slot] = ''
        result[domin] = temp

    # for domin_key, domin_value in sys_metadata.items():
    #     if domin_key not in slots.keys():
    #         continue
    #     temp = OrderedDict()
    #     for attr, value in domin_value['book'].items():
    #         if attr =='booked' and value and 'name' in  value[0].keys(): #need to change
    #             temp['name'] = value[0]['name']
    #         else:
    #             temp['name'] = ''
    #         if attr in slots[domin_key]:
    #             if attr in slot_map.keys():
    #                 attr = slot_map[attr]
    #             temp[attr] = value
    #     for attr, value in domin_value['semi'].items():
    #         if attr in slots[domin_key]:
    #             if attr in slot_map.keys():
    #                 attr = slot_map[attr]
    #             temp[attr] = value
    #     result[domin_key] = temp

    return result

def process_dialog(dialog):
    result = {}
    if len(dialog['log']) % 2 != 0:
        # print('odd turns')
        return {}

    # temp = {}
    # for a_goal in dialog['goal']:
    #     if a_goal in ignore_in_goal:
    #         continue
    #     temp[a_goal] = dialog['goal'][a_goal]
    # result['goal'] = temp
    
    user_turns_list = []
    sys_turns_list = []
    status_list = []

    log = dialog['log']

    for i in range(len(log)):
        # if len(log[i]['text'].strip('\n').split()) > maxlen:
        #     print('too long')
        #     return None
        if i % 2 == 0:  
            text = log[i]['text']
            if not is_ascii(text):
                return None
            text = normalize(text)
            user_turns_list.append(text)
        else:
            text = log[i]['text']
            if not is_ascii(text):
                return None
            text = normalize(text)
            sys_turns_list.append(text)
            status_list.append(process_metadata(log[i]['metadata']))
    
    result['user_turns'] = user_turns_list
    result['sys_turns'] = sys_turns_list
    result['turns_status'] = status_list
    
    return result

def load_diag_data(train_samples_num, test_saples_num, SNG=False):
    data_file_name = FILE_PATH_PREFIX + 'data.json'
    file_read = open(data_file_name, 'r')

    testlistfile_name = FILE_PATH_PREFIX + 'testListFile.json'
    testlistfile = open(testlistfile_name, mode='r')

    json_data = json.load(file_read)
    
    #process testlistfile
    test_name_list = []
    lines = testlistfile.readlines()
    for line in lines:
        line = line.strip('\n')
        test_name_list.append(line)

    train_dialogs_info = {}
    test_dialogs_info = {}
    for dia_index, dialog_name in enumerate(json_data):
        if SNG and'SNG' not in  dialog_name:
            continue
        dialog = json_data[dialog_name]
        a = process_dialog(dialog)
        if a:
            if dialog_name in test_name_list:
                test_dialogs_info[dialog_name] = a
                # print('test')
            else:
                train_dialogs_info[dialog_name] = a
        if len(train_dialogs_info) > train_samples_num and len(test_dialogs_info) > test_saples_num:
            break
    print(len(train_dialogs_info))
    print(len(test_dialogs_info))
    print('load data ok!')
    return train_dialogs_info, test_dialogs_info

def load_slot_value_dict():
    slot_value_dict = {}
    with open('pub_dataset/MultiWOZ_1.0/ontology.json', 'r') as json_file:
        data = json.load(json_file)
        for index, domin_slot in enumerate(data):
            for value in data[domin_slot]:
                if value not in slot_value_dict:
                    slot_value_dict[attr] = index
                else:
                    continue

    return slot_value_dict

def load_slot_value_list():
    slot_value_list = set()
    slot_value_list.add('')
    with open('pub_dataset/MultiWOZ_1.0/ontology.json', 'r') as json_file:
        data = json.load(json_file)
        for index, domin_slot in enumerate(data):
            for value in data[domin_slot]:
                slot_value_list.add(value)
    
    slot_value_list = list(slot_value_list)
    a = slot_value_list.index('')
    slot_value_list[0], slot_value_list[a] = slot_value_list[a], slot_value_list[0]

    return list(slot_value_list)
