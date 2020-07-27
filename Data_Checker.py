#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 22:55:04 2020

@author: robertsteger
"""
import numpy as np
import pandas as pd
#seaborn

        
def Light_Chains(seq, X1):
    """
    input sequence title for light chains and array of data, this will create list-l that 
    will generate all SASA values and then if seqence is missing will fill in 
    space with False
    returns list-l
    """
    l = []
    for i in range(120):
       l.append("#")
    row, col = X1.shape
    count = 1
    flag = False
    l_index = 0
    for k in range(row):
        i = k
        while(X1[i][0] == seq):
            flag = True
            residue= X1[i][1]
            length = len(residue)
            if residue[length-1:] in "ABCDEFGHIJKLMNOPQRISTUVWXYZ":
                num = int(residue[1:length-1])
            else:
                 num = int(residue[1:])
            if residue[0] == "L":
                if residue[length-1:] in "ABCDEFGHIJKLMNOPQRISTUVWXYZ":
                    if count-1 == num:
                        l[l_index] = X1[i][6]
                        count-=1
                        i+=1
                    elif count == num:
                        l[l_index] = X1[i][6]
                        count-=1
                        i+=1
                elif count == num:
                    l[l_index] = X1[i][6]
                    i+=1
                else:
                    l[l_index] = False
            else:
                break
            
            count+=1
            l_index+=1
            
        if flag == True:
            break

    return l

def Heavy_Chains(seq, X1):
    """
    input sequence title for heavy chains and array of data, this will create list-l that 
    will generate all SASA values and then if seqence is missing will fill in 
    space with False
    returns list-l
    """
    l = []
    for i in range(130):
       l.append("#")
    row, col = X1.shape
    count = 1
    l_index = 0
    flag = False
    for k in range(row):
        i = k
        while(X1[i][0] == seq):
            flag = True
            residue= X1[i][1]
            #print(residue)
            length = len(residue)
            if residue[length-1:] in "ABCDEFGHIJKLMNOPQRISTUVWXYZ":
                num = int(residue[1:length-1])
            else:
                 num = int(residue[1:])
                 
            if residue[0] == "H":
                if residue[length-1:] in "ABCDEFGHIJKLMNOPQRISTUVWXYZ":
                    if count-1 == num:
                        l[l_index] = X1[i][6]
                        count-=1
                        i+=1
                    elif count == num:
                        l[l_index] = X1[i][6]
                        count-=1
                        i+=1
                elif count == num:
                    l[l_index] = X1[i][6]
                    i+=1
                else:
                    l[l_index] = False
                
                count+=1
                #print(l_index)
                
                l_index+=1
            else:
                i+=1
        if flag == True:
            break
    
    return l
    
def Remover(SASA_list, index):
    #removes specified indexes
    SASA_list.pop(index)
    
def Checker(SASA_list):
    #Checks to see if there is missing data(will return False if there is)
    #Prints indicies of missing data
    #Can also remove these indexes
    flag = True
    i = 0
    while(i < len(SASA_list)):
        if SASA_list[i] is False:
            print(i)
            #Remover(SASA_list, i)
            flag = False
            #i-=1
        i+=1
    return flag
    
    
def Shorten(l):
    #Original lsit is size 130, this creates list of exact amount
    size = 0
    short_l = []
    for i in range(len(l)):
        short_l.append(0)
        if l[i] == "#":
            size = i
            break
    for j in range(size):
        short_l[j] = l[j]
        #print(short_l[j])
    
    return short_l

#import data and convet to array
data_df1 = pd.read_csv("38sequence_data.txt")
X1 = np.array(data_df1.values)
row, col =  X1.shape

#using functions

#l = Heavy_Chains("1mim-HL", X1)
l = Light_Chains("1mim-HL", X1)



SASA_list = Shorten(l)# I can prob just add this into H/L_chain funcs 
print(len(SASA_list))
print()
print(Checker(SASA_list))
print(SASA_list)
