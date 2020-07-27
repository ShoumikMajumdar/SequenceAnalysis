#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 6 11:16:13 2019

@author: cxue2
"""

from lstm import LSTM_Bi
from util_data import ProteinSeqDataset, collate_fn
from tqdm import tqdm
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
import time

class ModelLSTM:
    def __init__(self, embedding_dim=64, hidden_dim=64, device='cpu'):
        self.nn = LSTM_Bi(embedding_dim, hidden_dim, len(ProteinSeqDataset.vocab), device)
        #self.to(device)
        
    def fit(self, n_epoch=10, trn_batch_size=128, vld_batch_size=512, lr=.002, save_fp=None):
        # loss function and optimization algorithm
        loss_fn = torch.nn.MSELoss()
        op = torch.optim.Adam(self.nn.parameters(), lr=lr)
        
        # to track minimum validation loss
        min_loss = np.inf
        
        # dataset and dataset loader
        trn_data = ProteinSeqDataset(mode='TRN')
        vld_data = ProteinSeqDataset(mode='VLD')
        trn_dataloader = torch.utils.data.DataLoader(trn_data, trn_batch_size, True, collate_fn=collate_fn)
        vld_dataloader = torch.utils.data.DataLoader(vld_data, vld_batch_size, False, collate_fn=collate_fn)
        
        for epoch in range(n_epoch):
            # training
            self.nn.train()
            loss_avg, cnt = 0, 0
            with tqdm(total=len(trn_data), desc='Epoch {:03d} (TRN)'.format(epoch), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
                for X, Y in trn_dataloader:
                    # targets
                    Y_flatten = []
                    for y in Y:
                        Y_flatten += y
                    Y_flatten = torch.tensor(Y_flatten, device=self.nn.device)
                    
                    # forward and backward routine
                    self.nn.zero_grad()
                    scores = self.nn(X)
                    loss = loss_fn(scores, Y_flatten)
                    loss.backward()
                    op.step()
                    
                    # compute statistics
                    L = len(Y_flatten)
                    loss_avg = (loss_avg * cnt + loss.data.cpu().numpy() * L) / (cnt + L)
                    cnt += L
                    
                    # update progress bar
                    pbar.set_postfix({'loss': '{:.6f}'.format(loss_avg)})
                    pbar.update(len(X))
            
            # validation
            self.nn.eval()
            loss_avg, cnt = 0, 0
            with torch.set_grad_enabled(False):
                with tqdm(total=len(vld_data), desc='          (VLD)'.format(epoch), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
                    for X, Y in vld_dataloader:
                        # targets
                        Y_flatten = []
                        for y in Y:
                            Y_flatten += y
                        Y_flatten = torch.tensor(Y_flatten, device=self.nn.device)
                        
                        # forward routine
                        scores = self.nn(X)
                        loss = loss_fn(scores, Y_flatten)
                        
                        # compute statistics
                        L = len(Y_flatten)
                        loss_avg = (loss_avg * cnt + loss.data.cpu().numpy() * L) / (cnt + L)
                        cnt += L
                        
                        # update progress bar
                        pbar.set_postfix({'loss': '{:.6f}'.format(loss_avg)})
                        pbar.update(len(X))
                        
#                    print(list(zip(Y[0], scores[0:len(Y[0])])))
                    plt.plot(np.abs(np.array(Y[0]) - scores.cpu().numpy()[0:len(Y[0])]))
                    plt.ylim(0, .5)
                    plt.show()
#                    time.sleep(2)
            
            # save model
            if loss_avg < min_loss and save_fp:
                min_loss = loss_avg
                self.save('{}/lstm_{:.6f}.npy'.format(save_fp, loss_avg))
        
            self.plot_scatter(vld_data)
            
    def fit_light(self, n_epoch=10, trn_batch_size=128, vld_batch_size=512, lr=.002, save_fp=None):
        # loss function and optimization algorithm
        loss_fn = torch.nn.MSELoss()
        op = torch.optim.Adam(self.nn.parameters(), lr=lr)
        
        # to track minimum validation loss
        min_loss = np.inf
        
        # dataset and dataset loader
        trn_data = ProteinSeqDataset(mode='TRN')
        vld_data = ProteinSeqDataset(mode='VLD')
        trn_dataloader = torch.utils.data.DataLoader(trn_data, trn_batch_size, True, collate_fn=collate_fn)
        vld_dataloader = torch.utils.data.DataLoader(vld_data, vld_batch_size, False, collate_fn=collate_fn)
        
        for epoch in range(n_epoch):
            # training
            self.nn.train()
            loss_avg, cnt = 0, 0
            with tqdm(total=len(trn_data), desc='Epoch {:03d} (TRN)'.format(epoch), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
                for X, Y in trn_dataloader:
                    # targets
                    Y_flatten = []
                    for y in Y:
                        Y_flatten += y
                    Y_flatten = torch.tensor(Y_flatten, device=self.nn.device)
                    
                    # forward and backward routine
                    self.nn.zero_grad()
                    scores = self.nn(X)
                    loss = loss_fn(scores, Y_flatten)
                    loss.backward()
                    op.step()
                    
                    # compute statistics
                    L = len(Y_flatten)
                    loss_avg = (loss_avg * cnt + loss.data.cpu().numpy() * L) / (cnt + L)
                    cnt += L
                    
                    # update progress bar
                    pbar.set_postfix({'loss': '{:.6f}'.format(loss_avg)})
                    pbar.update(len(X))
            
            # validation
            self.nn.eval()
            loss_avg, cnt = 0, 0
            with torch.set_grad_enabled(False):
                with tqdm(total=len(vld_data), desc='          (VLD)'.format(epoch), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
                    for X, Y in vld_dataloader:
                        # targets
                        Y_flatten = []
                        for y in Y:
                            Y_flatten += y
                        Y_flatten = torch.tensor(Y_flatten, device=self.nn.device)
                        
                        # forward routine
                        scores = self.nn(X)
                        loss = loss_fn(scores, Y_flatten)
                        
                        # compute statistics
                        L = len(Y_flatten)
                        loss_avg = (loss_avg * cnt + loss.data.cpu().numpy() * L) / (cnt + L)
                        cnt += L
                        
                        # update progress bar
                        pbar.set_postfix({'loss': '{:.6f}'.format(loss_avg)})
                        pbar.update(len(X))
                        
#                    print(list(zip(Y[0], scores[0:len(Y[0])])))
                    plt.plot(np.abs(np.array(Y[0]) - scores.cpu().numpy()[0:len(Y[0])]))
                    plt.ylim(0, .5)
                    plt.show()
#                    time.sleep(2)
            
            # save model
            if loss_avg < min_loss and save_fp:
                min_loss = loss_avg
                self.save('{}/lstm_{:.6f}.npy'.format(save_fp, loss_avg))
        
            self.plot_scatter(vld_data)
        
    def plot_scatter(self,scat_data):
        #vld_data = ProteinSeqDataset(mode='VLD')
        loss_fn = torch.nn.MSELoss()
        batch_size = len(scat_data.Y)
        print(batch_size)
        vld_dataloader = torch.utils.data.DataLoader(scat_data, batch_size, False, collate_fn=collate_fn)
        for X,Y in vld_dataloader:
            Y_flatten = []
            for y in Y:
                Y_flatten += y
            Y_flatten = torch.tensor(Y_flatten)
            
            
            print("test")
            scores = self.nn(X)
            
            #print("before loss")
            loss = torch.sqrt(loss_fn(scores,Y_flatten))
            #loss = loss_fn(scores,Y_flatten)
            #print("after loss")
            print(loss.item()*100)

        scores = scores.detach().numpy()
        #print(scores[:20])
        Y_flatten = Y_flatten.detach().numpy()

        plt.figure(figsize=(20,20))
        Xlin = np.linspace(0,1.25,120)
        ylin = np.linspace(0,1.25,120)
        plt.plot(Xlin,ylin,c='b')
        plt.scatter(scores,Y_flatten,c='g',linewidths=1)
        plt.xlabel("PRedicted SASA")
        plt.ylabel("True SASA")
        plt.show()
        
         
    
    def eval(self, fn, batch_size=512):        
        # dataset and dataset loader
        data = ProteinSeqDataset(fn)
        
        if batch_size == -1: batch_size = len(data)
        dataloader = torch.utils.data.DataLoader(data, batch_size, True, collate_fn=collate_fn)
        
        self.nn.eval()
        scores = np.zeros(len(data), dtype=np.float32)
        sys.stdout.flush()
        with torch.set_grad_enabled(False):
            with tqdm(total=len(data), ascii=True, unit='seq', bar_format='{l_bar}{r_bar}') as pbar:
                for n, (batch, batch_flatten) in enumerate(dataloader):
                    actual_batch_size = len(batch)  # last iteration may contain less sequences
                    seq_len = [len(seq) for seq in batch]
                    seq_len_cumsum = np.cumsum(seq_len)
                    #out = self.nn(batch, aa2id_i[self.gapped]).data.cpu().numpy()
                    out = self.nn(batch).data.cpu().numpy()
                    out = np.split(out, seq_len_cumsum)[:-1]
                    batch_scores = []
                    for i in range(actual_batch_size):
                        pos_scores = []
                        for j in range(seq_len[i]):
                            pos_scores.append(out[i][j, batch[i][j]])
                        batch_scores.append(-sum(pos_scores) / seq_len[i])    
                    scores[n*batch_size:(n+1)*batch_size] = batch_scores
                    pbar.update(len(batch))
        return scores
    

    
    
    def save(self, fn):
        param_dict = self.nn.get_param()
        param_dict['gapped'] = self.gapped
        np.save(fn, param_dict)
    
    def load(self, fn):
        param_dict = np.load(fn, allow_pickle=True).item()
        self.gapped = param_dict['gapped']
        self.nn.set_param(param_dict)

    def to(self, device):
        self.nn.to(device)
        self.nn.device = device
        
    def summary(self):
        for n, w in self.nn.named_parameters():
            print('{}:\t{}'.format(n, w.shape))
#        print('LSTM: \t{}'.format(self.nn.lstm_f.all_weights))
#        print('Fixed Length:\t{}'.format(self.nn.fixed_len) )
#        print('Gapped:\t{}'.format(self.gapped))
        print('Device:\t{}'.format(self.nn.device))
            