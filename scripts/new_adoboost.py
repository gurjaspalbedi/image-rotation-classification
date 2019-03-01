# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 00:21:43 2018

@author: gurjaspal
"""

#!/usr/bin/env python3
# Author: Sahil

import pickle
import pandas as pd
import numpy as np
import scipy.spatial.distance as dist
import random
import math
from random import sample, randint
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier


train_data_loc = '../data/train-data.txt'
test_data_loc = '../data/test-data.txt'

train_df = pd.read_csv(train_data_loc, header=None, delim_whitespace=True)
test_df = pd.read_csv(test_data_loc, header=None, delim_whitespace=True)

train_data = np.array(np.array(train_df)[:, 2:], dtype=int)
train_label = np.array(np.array(train_df)[:, 1].T, dtype=int)
train_label.resize((train_label.shape[0], 1))
test_data = np.array(np.array(test_df)[:, 2:], dtype=int)
test_label = np.array(np.array(test_df)[:, 1].T, dtype=int)
test_label.resize((test_label.shape[0], 1))


degrees = [ 0,90,180,270]
#training on test data
length = len(test_data)
decicion_stump_count = 10
for degree in degrees:
    error = 1
    predictions = []
    error = 1
    stumps = []
    while len(stumps) < decicion_stump_count:
        first_idx = np.random.randint(0,191)
        second_idx = np.random.randint(0,191)
        label = [0] * len(train_data)
        weights = [1.0/length] * length
       
        for m, row in enumerate(test_data):
            first_point = row[first_idx]
            second_point = row[second_idx]
            
            
            if first_point - second_point > 0:
                label[m] = degree
            else:
                label[m] = -1
                
        error = 0
        for i in range(length):
            if label[i] != test_label[i]:
                error = error + weights[i]
#            print(error)
        print("adding stump")
        print(error)
        stumps.append(error)
        
        
    
    
        
        
        
        
        




























