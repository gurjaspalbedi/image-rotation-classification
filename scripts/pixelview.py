# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 21:50:03 2018

@author: gurjaspal
"""

import numpy as np
import scipy.misc as smp

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
# Create a 1024x1024x3 array of 8 bit unsigned integers
data = np.zeros( (1024,1024,3), dtype=np.uint8 )

data[512,512] = [254,0,0]       # Makes the middle pixel red
data[512,513] = [0,0,255]       # Makes the next pixel blue

img = smp.toimage( data )       # Create a PIL image
img.show() 