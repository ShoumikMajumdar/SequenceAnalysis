#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 16:21:11 2019

@author: cxue2
"""
from model_wrapper import ModelLSTM
import torch
import torch.nn as nn
import torch.nn.functional as F
from util_data import ProteinSeqDataset, collate_fn
#from utils_light_heavy import ProteinSeqDataset, collate_fn
import numpy as np
import csv
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt
import math
from Data_Checker import Light_Chains, Heavy_Chains , Remover, Checker
import pandas as pd



#model = ModelLSTM(64, 64, device='cpu')
#model.fit(n_epoch=512, trn_batch_size=32, vld_batch_size=32, lr=.002, save_fp=None)

model = torch.load("e512_63_heavy.tar",map_location=torch.device('cpu'))
#print(model.nn.fc1)
model.nn.eval()

data = ProteinSeqDataset("EVAL")
model.plot_scatter(data)

#Entire data
pseudoX = data.pseudoX
pseudoy = data.pseudoY

newX = {}


#After split
lightX = data.X
lighty = data.Y
map = data.pseudo_hmap

# =============================================================================
# For getting flattened SASA values
# =============================================================================
test_dataloader = torch.utils.data.DataLoader(data, 128, False, collate_fn=collate_fn)
loss_fn = torch.nn.MSELoss()

for X,Y in test_dataloader:
    Y_flatten = []
    for y in Y:
        Y_flatten += y
    Y_flatten = torch.tensor(Y_flatten, device='cpu')
    scores = model.nn(X)
    loss = torch.sqrt(loss_fn(scores, Y_flatten))




scores = scores.detach().numpy()
Y_flatten = Y_flatten.detach().numpy()

plt.figure(figsize=(10,10))
Xlin = np.linspace(0,1.25,120)
ylin = np.linspace(0,1.25,120)
plt.plot(Xlin,ylin,c='b')
plt.scatter(scores,Y_flatten,c='g')
plt.xlabel("PRedicted SASA")
plt.ylabel("True SASA")
plt.show()









# with open('flatenned_Y.data', 'wb') as filehandle:
#     # store the data as binary data stream
#     pickle.dump(Y_flatten, filehandle)

# =============================================================================

# For Getting SASA values for sequences
# =============================================================================


# hmap = data.pseudo_hmap



# X = data.X
# y = data.Y




# =============================================================================
#           Prediction on data
# =============================================================================
# for i in range(len(X)):
#     inputs = []
#     inputs.append(X[i])
#     preds = model.nn(inputs)
#     outputs.append(preds.detach().numpy())
#     inputs.clear()


# =============================================================================
#       Write as a text file
# =============================================================================
# with open('listfile.txt', 'w') as filehandle:
#     for listitem in places:
#         filehandle.write('%s\n' % listitem)


# =============================================================================
#               Write as data file using PICKLE
# =============================================================================
# with open('outputs_800_1.data', 'wb') as filehandle:
#     # store the data as binary data stream
#     pickle.dump(outputs, filehandle)
    
    
# =============================================================================
#           Read .data file
# =============================================================================
# with open('outputs_800_1.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     outputs = pickle.load(filehandle)



# =============================================================================
#         Average SASA values
# =============================================================================
# average_Sasa = []
# #Average SASA
# for i in range(len(outputs)):
#     average_Sasa.append(np.mean(outputs[i]))
        
    

# Average_ground_truth = []
# for i in range(len(outputs)):
#     Average_ground_truth.append(np.mean(pseudoY[i]))




    
# =============================================================================
#     Compute Sequence Level Loss
# =============================================================================


# for index in range(len(outputs)):
#     pseudoY[index]=np.array(pseudoY[index])
    
    

# loss = []
# for index in range(len(outputs)):
#     size = len(outputs[index])
#     temp = np.empty(size)
#     for j in range(size):
#         #print(j)
#         #print(pseudoY[index][i])
#         #print(outputs[index][i])
#         mea = outputs[index][j] - pseudoY[index][j]
#         temp[j] = mea**2
#         #print(mea**2)
#     loss.append(temp)


# # Mean absolute error

# loss_MAE = []
# for index in range(len(outputs)):
#     size = len(outputs[index])
#     temp_mae = np.empty(size)
#     for j in range(size):
#         #print(j)
#         #print(pseudoY[index][i])
#         #print(outputs[index][i])
#         mean_abs_error = outputs[index][j] - pseudoY[index][j]
#         temp_mae[j] = np.abs(mean_abs_error)
#         #print(mea**2)
#     loss_MAE.append(temp_mae)



# MAE =[]
# MSE = []
# RMSE=[]
# for index in range(len(loss)):
#     MSE.append(np.mean(loss[index]))
#     RMSE.append(math.sqrt(np.mean(loss[index])))
#     MAE.append(np.mean(loss_MAE[index]))
    
    
# print(np.mean(MSE))
# print(np.mean(RMSE)*100)
# print(np.mean(MAE)*100)






hmap = data.pseudo_hmap
keys = list(hmap.keys())


