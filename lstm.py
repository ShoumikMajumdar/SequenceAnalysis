#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 11:17:17 2019

@author: cxue2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence

class LSTM_Bi(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, device):
        super(LSTM_Bi, self).__init__()
        self.device = device
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, 1)
        
    def forward(self, Xs):
        # sequence lengths
        Xs_len = [len(seq) for seq in Xs]
        Xs_len_max = max(Xs_len)
        
        # list to *.tensor
        Xs = [torch.tensor(seq, device='cpu') for seq in Xs]
        
        # padding
        #Xs = pad_sequence(Xs, batch_first=True).to(self.device)
        Xs = pad_sequence(Xs, batch_first=True)
        
        # embedding
        Xs = self.word_embeddings(Xs)   
        
        # packing the padded sequences
        Xs = pack_padded_sequence(Xs, Xs_len, batch_first=True, enforce_sorted=False)

        # lstm
        lstm_out, _ = self.lstm(Xs)
        
        # unpack outputs
        lstm_out, lstm_out_len = pad_packed_sequence(lstm_out, batch_first=True)
        
        # reshape lstm output
        idx = []
        for i, l in enumerate(Xs_len):
            idx += [i*Xs_len_max+j for j in range(l)]
        #idx = torch.tensor(idx, device=self.device)
        idx = torch.tensor(idx)
        
        lstm_out_valid = lstm_out.reshape(-1, self.hidden_dim*2)
        lstm_out_valid = torch.index_select(lstm_out_valid, 0, idx)
        
        # lstm hidden state to output space
        out = F.relu(self.fc1(lstm_out_valid))
        out = self.fc2(out)
        
        return torch.squeeze(out)
    
    def set_param(self, param_dict):
        try:
            # pytorch tensors
            for pn, _ in self.named_parameters():
                exec('self.%s.data = torch.tensor(param_dict[pn])' % pn)
            
            # hyperparameters
            self.embedding_dim = param_dict['embedding_dim']
            self.hidden_dim = param_dict['hidden_dim']
            self.vocab_size = param_dict['vocab_size']
            
            self.to(self.device)
        except:
            print('Unmatched parameter names or shapes.')        
    
    def get_param(self):
        param_dict = {}
 
        # pytorch tensors
        for pn, pv in self.named_parameters():
            param_dict[pn] = pv.data.cpu().numpy()
            
        # hyperparameters
        param_dict['embedding_dim'] = self.embedding_dim
        param_dict['hidden_dim'] = self.hidden_dim
        param_dict['vocab_size'] = self.vocab_size

        return param_dict
        