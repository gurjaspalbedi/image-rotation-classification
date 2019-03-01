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

print(train_data.shape, test_data.shape)
print(train_label.shape, test_label.shape)

length = len(train_data)
i = 0
hypthesis_count = 200
result = []
predictions = [0, 90, 180, 270]
list_of_alphas = []
list_of_decision_stumps = []
# these are the only predictions possible
# for each we get one hypotheisi: a0h0 + a1h1


def normalise(arr):
    s = sum(arr)
    for i in range(len(arr)):
        arr[i] = arr[i]/s
    return arr


alpha = []
for predict in predictions:

    decision_stumps = []
    hyp = []
    weights = [1.0 / length] * length
    for i in range(hypthesis_count):
        #        till hypothesis count to get series till a99* h99
        error = 0
        test = []
#       giving equal weights for all
#        we will classify each image if it is 0 or not, 90 or not and so on
        labels = [0] * len(train_data)
#        [first_index, second_index]= random.sample(range(0, 191), 2)
        first_index = random.randint(0, 191)
        second_index = random.randint(0, 191)
        for m, row in enumerate(train_data):

            first_number = row[first_index]
            second_number = row[second_index]
            test.append(first_number-second_number)

#            hypothesis evaluation
            if first_number - second_number > 0:
                labels[m] = predict
            else:
                labels[m] = -1


#        error is sum of all the weights which we predicted wrong
        for j in range(length):
            if labels[j] != train_label[j][0]:
                error = error + weights[j]

#        updating the weight for each training set
        for k in range(length):
            if labels[k] == train_label[k][0]:
                weights[k] = weights[k] * (error / (1 - error))
#                weights[k] = weights[k] * ((1-error) /error)

        normalise(weights)
        hyp.append((math.log((1 - error) / error)))
#        hyp.append((math.log((error)/ (1-error))))
        decision_stumps.append([first_index, second_index])
#        keeping the a part of the hypothesis, we will have 100 such values
    alpha.append(hyp)
    list_of_decision_stumps.append(decision_stumps)

    print("adding hypothesis")


#    hypothesis will have four hypothesis
#    list_of_alphas.append(alpha)


# testing
correct = 0
for z in range(len(test_data)):
    row = test_data[z]
    max_value = 0
    first = 0
    second = 0
    s = 0
    for i in range(len(alpha)):
        for j in range(len(alpha[i])):
            s = s + alpha[i][j]*(row[list_of_decision_stumps[i]
                                     [j][0]] - row[list_of_decision_stumps[i][j][1]])
        if s > max_value:
            max_value = s
            tag = predictions[i]
    if tag == test_label[z]:
        correct += 1


# for index, row in enumerate(test_data):
print(correct)
print(len(test_data))
print(correct/len(test_data))


#
# may be max possible accuracy
#clf = AdaBoostClassifier()
#clf.fit(train_data, train_label.ravel())
#pred_data = clf.predict(test_data)
#score = accuracy_score(test_label.ravel(), pred_data)
#print('The accuracy for adaboost classification is {}'.format(score))
