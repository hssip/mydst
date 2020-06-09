# -*-coding:utf-8 -*-

import json, pickle
import copy 
import os, re
import shutil

FILE_PATH_PREFIX = 'pub_data'

file_name = 'pub_dataset/MultiWOZ_1.0/data.json'
with open(file_name, 'r') as f:
    data = json.load(f)
    # for a in data:
    print(isinstance(data['SNG01856.json']['goal']['train'], dict))