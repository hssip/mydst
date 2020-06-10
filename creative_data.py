# -*-coding:utf-8 -*-

import json, pickle
import copy 
import os, re
import shutil
import urllib.request
from collections import OrderedDict
from io import BytesIO
from zipfile import ZipFile
import difflib
import numpy as np 


FILE_PATH_PREFIX = 'pub_dataset/MultiWOZ_1.0/'
LOAD_DIAG_NUM = 10

ignore_in_goal = ['eod', 'topic', 'messageLen', 'message']

replace_lines = open('pub_dataset/mapping.pair', 'r').readlines()
replace_list = []
for line in replace_lines:
    tok_from, tok_to = line.replace('\n', '').split('\t')
    replace_list.append((' ' + tok_from + ' ', ' ' + tok_to + ' '))

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

    if clean_value:
        # replace time and and price
        text = re.sub(timepat, ' [value_time] ', text)
        text = re.sub(pricepat, ' [value_price] ', text)
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
    result = {}
    # sys_metadata = {}
    for domin_key, domin_value in sys_metadata.items():
        temp = {}
        for first_key, first_value in domin_key.items():
            for second_key, second_value in first_value.items():
                temp[second_key] = second_value
        result[domin_key] = domin_value
    
    return result


def process_dialog(dialog, maxlen):
    result = {}
    if len(dialog['log']) % 2 != 0:
        print('odd turns')
        return None

    temp = {}
    for a_goal in dialog['goal']:
        if a_goal in ignore_in_goal:
            continue
        temp[a_goal] = dialog['goal'][a_goal]
    result['goal'] = temp
    
    user_turns_list = []
    sys_turns_list = []
    status_list = []

    log = dialog['log']

    for i in range(len(log)):
        if len(log[i]['text'].strip('\n').split()) > maxlen:
            print('too long')
            return None
        
        if i % 2 == 0:  
            text = log[i]['text']
            if not is_ascii(text):
                return None
            user_turns_list.append(text)
        else:
            text = log[i]['text']
            if not is_ascii(text):
                return None
            else:
                sys_turns_list.append(text)

                status_list.append(process_metadata(log[i]['metadata']))
    
    result['user_turns'] = user_turns_list
    result['sys_turns'] = sys_turns_list
    result['turns_status'] = status_list
    
    return result


def load_diag_data(max_length):
    data_file_name = FILE_PATH_PREFIX + 'data.json'
    file_read = open(data_file_name, 'r')
    json_data = json.load(file_read)
    dialogs_info = {}
    for dia_index, dialog_name in enumerate(json_data):
        dialog = json_data[dialog_name]
        dialogs_info[dialog_name] = process_dialog(dialog, max_length)
        if dia_index > LOAD_DIAG_NUM:
            break
    
    return dialogs_info

def load_domin_info(domin):
    pass
    return 