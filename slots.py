# -*- coding:utf-8 -*-
class Slots:

    _slots = []
    _vocab = []

    def __init__(self, slots_embedding):
        super().__init__()
        for i in range(len(slots_embedding)):
            self._slots.append(slots_embedding[i])
            self._vocab.append()