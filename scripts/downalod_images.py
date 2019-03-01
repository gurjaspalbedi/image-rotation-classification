# -*- coding: utf-8 -*-
"""
Created on Sat Dec  8 18:11:11 2018

@author: gurjaspal
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as dist
import random
import math
from random import sample, randint
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier
import requests


train_data_loc = '../data/train-data.txt'
test_data_loc = '../data/test-data.txt'

train_df = pd.read_csv(train_data_loc, header=None, delim_whitespace=True)
test_df = pd.read_csv(test_data_loc, header=None, delim_whitespace=True)

train_data = np.array(np.array(train_df)[:,2:], dtype=int)
train_label = np.array(np.array(train_df)[:,1].T, dtype=int)
train_label.resize((train_label.shape[0], 1))
test_data = np.array(np.array(test_df)[:,2:], dtype=int)
test_label = np.array(np.array(test_df)[:,1].T, dtype=int)
test_label.resize((test_label.shape[0], 1))


for row in train_df:
    image = train_df[0][0].split("/")[1]
    url = 'http://www.flickr.com/photo_zoom.gne?id=' + image
    response = requests.get(url)
    if response.status_code == 200:
        print("reponse")
        print(response.content)
        with open("/sample.jpg", 'wb') as f:
            f.write(response.content)
    else:
        print("error")
    
    