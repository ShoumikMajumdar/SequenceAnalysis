#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 15:23:48 2019

@author: cxue2
"""

from torch.utils.data import Dataset
from collections import defaultdict
import random
import csv

class ProteinSeqDataset(Dataset):
    vocab =\
        ['-'] +\
        ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P',
         'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    
    def __init__(self, mode, seed=3227):
        self.X, self.Y = [], []
        self.pseudoX, self.pseudoY = [], []
        self.pseudo_hmap= defaultdict(list)
        self._aa_id_map =\
            dict([(self.vocab[i], i) for i in range(len(self.vocab))])
        self._load_file('data.txt')
        
             
        # random shuffle and split into training and validation set
        assert mode in ['TRN', 'VLD','EVAL']
        total_len = len(self.Y)
        idx = list(range(total_len))
        random.seed = seed
        random.shuffle(idx)
        if mode == 'TRN':
            idx = idx[:round(total_len*.6322)]
        elif mode == 'VLD':
            idx = idx[round(total_len*.6322):]
        else:
            idx = idx[0:]
        self.X = [self.X[i] for i in idx]
        self.Y = [self.Y[i] for i in idx]
    
    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def _load_file(self, fn):
        # read txt file
        hmap = defaultdict(list)
        with open(fn) as f:
            parser = csv.reader(f, delimiter=',')
            for i, r in enumerate(parser):
                if len(r) != 7 or i == 1:
                    continue
                hmap['{}-{}'.format(r[0], r[1][0])].append(r[2])
                hmap['{}-{}'.format(r[0], r[1][0])].append(float(r[-1]))
                
                self.pseudo_hmap['{}-{}'.format(r[0], r[1][0])].append(r[2])
                self.pseudo_hmap['{}-{}'.format(r[0], r[1][0])].append(float(r[-1]))
                
        # reorganize data format
        for k, v in hmap.items():
            hmap[k] = ([v[i*2] for i in range(len(v)//2)],
                       [v[i*2+1] for i in range(len(v)//2)])
            
            
        for k, v in self.pseudo_hmap.items():
            self.pseudo_hmap[k] = ([v[i*2] for i in range(len(v)//2)],
                        [v[i*2+1] for i in range(len(v)//2)])    
        #print(hmap)
        
        # convert AA to id and construct X & Y
        for k, v in hmap.items():
            self.X.append([self._aa_id_map[aa] for aa in v[0]])
            self.Y.append(v[1])
        
        bhelx = {}
        for k, v in self.pseudo_hmap.items():
            self.pseudo_hmap[k] = [self._aa_id_map[aa] for aa in v[0]]
            self.pseudoX.append([self._aa_id_map[aa] for aa in v[0]])
            self.pseudoY.append(v[1])    
        
def collate_fn(batch):
    return [batch[i][0] for i in range(len(batch))],\
           [batch[i][1] for i in range(len(batch))]
