# -*- coding:utf-8 -*-
import numpy as np

class Slots:

    _vocab_slots = {}
    _embed_slots = {}
    def __init__(self, slots_dict, embed_matri = None, is_embedding = False):
        # super().__init__()
        if is_embedding:
            for domin in slots_dict:
                temp = {}
                for slot in slots_dict[domin]:
                    try:
                        temp[slot] = embed_matri[slot]
                    except KeyError:
                        raise 'cannot find ' + domin + ' , ' + slot + 'in embedding_matri'
        else:
            for domin in slots_dict:
                for slot in slots_dict[domin]:
                    self._raw_slots.appen((domin))
    def __iter__(self):
        self.__iter_value = 0
        return self

    def __next__(self):
        if self.__iter_value < len(self._slots.__len__()):
            embed = self._slots[self.__iter_value]
            self.__iter_value += 1
            return embed
        else:
            raise StopIteration

    def set_slot_value(self, i, value):
        self._vocab[i] = value

    def get_slot_value(self, domin, slot):
        index = self._raw_slots.index((domin, slot))
        if index != -1:
            return self._vocab[index]
        else:
            raise Exception('The pair: ' + domin + ' ' + slot + 'not exists!')
    
