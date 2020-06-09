# -*- coding:utf-8 -*-
import numpy as np

class Slots:

    _slots = []
    _vocab = []

    _raw_slots = []

    def __init__(self, slots_dict, embed_matri):
        super().__init__()
        for domin in range(slots_dict):
            for slot in domin:
                self._slots.append(embed_matri[domin].extend(embed_matri[slot])
                self._vocab.append(None)
                self._raw_slots.append((domin, slot))

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
            raise Exception('The pair: ' + domin + ' ' + slots + 'not exists!')
    
