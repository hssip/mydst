# -*-coding:utf-8 -*-
import numpy as np 
import json, pickle
from gensim import models
from io import BytesIO
from zipfile import ZipFile
from collections import OrderedDict 
import copy

domin_list = ['hotel', 'restaurant', 'taxi', 'attraction', 'train']
slots_list = [['name','type', 'parking', 'pricerange', 'internet', 'day', 'stay', 'people', 'area', 'stars'],
                ['food', 'pricerange', 'area', 'name', 'time', 'day', 'people'],
                ['leaveat', 'destination', 'departure', 'arriveby'], 
                ['name', 'type', 'area'],
                ['people', 'leaveat', 'destination', 'day','arriveby', 'departure']
                ]

GATE_INDEX = ['UPDATE', 'DONTCARE', 'NONE', 'DELETE']

gate2index = {
    'UPDATE':   0,
    'DONTCARE': 1,
    'NONE':     2,
    'DELETE':   3
}

special_slot_value={
    'dontcare':['any', 'does not care', 'dont care'],
    'none':['not men', 'not mentioned', 'fun', 'art', 'not', 
            'not mendtioned', '']
}

def fix_general_label_error(labels, type, slots):
    label_dict = dict([ (l[0], l[1]) for l in labels]) if type else dict([ (l["slots"][0][0], l["slots"][0][1]) for l in labels]) 

    GENERAL_TYPO = {
        # type
        "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports", 
        "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall", 
        "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture", 
        "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
        # area
        "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east", 
        "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre", 
        "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north", 
        "centre of town":"centre", "cb30aq": "none",
        # price
        "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate", 
        # day
        "next friday":"friday", "monda": "monday", 
        # parking
        "free parking":"free",
        # internet
        "free internet":"yes",
        # star
        "4 star":"4", "4 stars":"4", "0 star rarting":"none",
        # others 
        "y":"yes", "any":"dontcare", "n":"no", "does not care":"dontcare", "not men":"none", "not":"none", "not mentioned":"none",
        '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none",  
        }

    for slot in slots:
        if slot in label_dict.keys():
            # general typos
            if label_dict[slot] in GENERAL_TYPO.keys():
                label_dict[slot] = label_dict[slot].replace(label_dict[slot], GENERAL_TYPO[label_dict[slot]])
            
            # miss match slot and value 
            if  slot == "hotel-type" and label_dict[slot] in ["nigh", "moderate -ly priced", "bed and breakfast", "centre", "venetian", "intern", "a cheap -er hotel"] or \
                slot == "hotel-internet" and label_dict[slot] == "4" or \
                slot == "hotel-pricerange" and label_dict[slot] == "2" or \
                slot == "attraction-type" and label_dict[slot] in ["gastropub", "la raza", "galleria", "gallery", "science", "m"] or \
                "area" in slot and label_dict[slot] in ["moderate"] or \
                "day" in slot and label_dict[slot] == "t":
                label_dict[slot] = "none"
            elif slot == "hotel-type" and label_dict[slot] in ["hotel with free parking and free wifi", "4", "3 star hotel"]:
                label_dict[slot] = "hotel"
            elif slot == "hotel-star" and label_dict[slot] == "3 star hotel":
                label_dict[slot] = "3"
            elif "area" in slot:
                if label_dict[slot] == "no": label_dict[slot] = "north"
                elif label_dict[slot] == "we": label_dict[slot] = "west"
                elif label_dict[slot] == "cent": label_dict[slot] = "centre"
            elif "day" in slot:
                if label_dict[slot] == "we": label_dict[slot] = "wednesday"
                elif label_dict[slot] == "no": label_dict[slot] = "none"
            elif "price" in slot and label_dict[slot] == "ch":
                label_dict[slot] = "cheap"
            elif "internet" in slot and label_dict[slot] == "free":
                label_dict[slot] = "yes"

            # some out-of-define classification slot values
            if  slot == "restaurant-area" and label_dict[slot] in ["stansted airport", "cambridge", "silver street"] or \
                slot == "attraction-area" and label_dict[slot] in ["norwich", "ely", "museum", "same area as hotel"]:
                label_dict[slot] = "none"

    return label_dict

def process_belief_state(belief_state):
    result = OrderedDict()
    for domin_num, domin in enumerate(domin_list):
        slots = OrderedDict()
        for slot in slots_list[domin_num]:
            slots[slot] = ''
        result[domin] = slots

    if belief_state:
        for belief in belief_state:
            arr = belief['slots'][0][0].lower().split('-')
            temp = arr[1]
            if 'book' in temp:
                temp = temp.split(' ')[1]
            value = belief['slots'][0][1].lower()
            # if value in 
            
            result[arr[0]][temp] = belief['slots'][0][1].lower()

    return result

def load_diag_data(samples_num=300, data_kind = 'train'):
    path  = 'data/'
    file_name = path + data_kind +'_dials.json'

    #load data
    data_file = open(file_name, mode='r')
    data = json.load(data_file)
    pairs = []
    i=0
    for dia in data:
        i+=1
        if i > samples_num:
            break
        flag = True
        for domain in dia['domains']:
            if domain not in domin_list:
                flag = False
        if not flag:
            continue
        dialog_name = dia['dialogue_idx']
        histr_context = ''
        for log_num, log in enumerate(dia["dialogue"]):
            pair = {}
            domin = log['domain']
            if domain not in domin_list:
                continue
            if log['transcript']:
                histr_context +=  log['transcript'] + ' EOU_token '
            if log['system_transcript']:
                histr_context += log['system_transcript'] + ' EOS_token '
            pair['histr_context'] = '[START_token] ' + histr_context + 'END_token'
            pair['domain'] = domain
            pair['belief_state'] = process_belief_state(log['belief_state'])
            pair['turn_id'] = log['turn_idx']
            pair['dialog_name'] = dialog_name
            pairs.append(pair)
 
    return pairs



    

