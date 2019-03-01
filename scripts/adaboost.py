#!/usr/bin/env python3
# Author: Sahil
import pickle
import pandas as pd
import numpy as np
import random
import math


hypthesis_count = 200 # number of alphas you want, we get accuracy around 47 using 100
result = []
predictions = [0, 90, 180, 270]

# these are the only predictions possible
# for each we get one hypotheisi: a0h0 + a1h1


def normalise(arr):
    s = sum(arr)
    for i in range(len(arr)):
        arr[i] = arr[i]/s
    return arr

#This is the function which can be used to get the pixels from one quadrant
#When you rotate the image by 90 we take pixel for second quadrant
# this is just experimentation, this gives me 50% accuracy for now
def get_first_quadrant(pred, index):

#    return_list = list(range(0,192,3))
    if pred ==0 :
#        range for the first quadrant
        ranges = [[1,4],[9,12],[17,20],[25,28]]
    if pred == 90:
#        range for the second quadrant
        ranges = [[5,8],[13,16],[21,24],[29,32]]
    if pred == 180:
#        range for the third quadrant
        ranges =  [[37,40],[45,48],[53,56],[61,64]]
    if pred == 270:
#        range for the third quadrant
        ranges =  [[33,36],[41,44],[49,52],[57,60]]
    return_list = []
    for item in ranges:
#        we multiple each position by index to determine if we want red, green or blue for particular quadrant
        return_list = return_list + list(range(item[0]*index,(item[1]*index)+1))
#        returns the list with all the values from which you can take two indexes for stumps
    return return_list

#I tried this function just to get values from quadrant without taking degree into consideration
#    index here means quadrant
#    NOT USING FOR NOW, BUT CAN BE USED TO DO SOME EXPERIMENTS
def get_quadrant(index, color):

#    return_list = list(range(0,192,3))
    if index ==1 :
        ranges = [[1,4],[9,12],[17,20],[25,28]]
    if index == 2:
        ranges = [[5,8],[13,16],[21,24],[29,32]]
    if index == 3:
        ranges =  [[37,40],[45,48],[53,56],[61,64]]
    if index == 4:
        ranges =  [[33,36],[41,44],[49,52],[57,60]]
    return_list = []
    for item in ranges:
        return_list = return_list + list(range(item[0]*color,(item[1]*color)+1))
    return return_list


def train_adaboost(train_data, train_label):
    list_of_decision_stumps = []
    alpha = []
    stum_pred = []
    for predict in predictions:
        decision_stumps = []
        hyp = []

        weights = [1.0 / len(train_data)] * len(train_data)
    #    while len(decision_stumps) < hypthesis_count:
        for i in range(hypthesis_count):
            #        till hypothesis count to get series till a99* h99
            error = 0
            labels = np.zeros(len(train_data))
            [first_index, second_index] = random.sample(get_first_quadrant(predict,3),2)

            first_index = first_index - 1
            second_index = second_index - 1
            random_stumps = []
            for m, row in enumerate(train_data):

                first_number = row[first_index]
                second_number = row[second_index]
    #            hypothesis evaluation
                if first_number - second_number > 0:
                    labels[m] = predict
                else:
                    labels[m] = -1
            stum_pred.append(labels)
    #        error is sum of all the weights which we predicted wrong
            for j in range(len(train_data)):
                if labels[j] != train_label[j][0]:
                    error = error + weights[j]

    #doing this just to avoid math log error
            if error >= 1.0:
                error = 0.99
    #        updating the weight for each training set
            for k in range(len(train_data)):
                if labels[k] == train_label[k][0]:
                    weights[k] = weights[k] * (error / (1 - error))


            normalise(weights)

    #        try catch just to see why we were getting error in this line
            try:
                hyp.append((np.log((1 - error) / error)))
            except:
                print(error)
                break

    #saving the decision stumps indexes so that we can use the same during the testing
            decision_stumps.append([first_index, second_index])
    #        keeping the a part of the hypothesis, we will have 100 such values
        alpha.append(hyp)
        list_of_decision_stumps.append(decision_stumps)
    return {'alpha': alpha, 'list_of_decision_stumps': list_of_decision_stumps}


def predict_adaboost(test_data, model_loc):
    f = open(model_loc, 'rb')
    model = pickle.loads(f.read())
    res = []
    list_of_decision_stumps, alpha = model['list_of_decision_stumps'], model['alpha']
    # testing
    for z in range(len(test_data)):
        row = test_data[z]
        max_value = -math.inf
        s = 0
        for i in range(len(alpha)):
            for j in range(len(alpha[i])):
    #            we multiple a and h for given decision stump and then sum them
    #            we do this for 0, 90, 180 and 270
    #            we classify the image as the maximum value we get from these four equations
                s = s + alpha[i][j]*(row[list_of_decision_stumps[i][j][0]] - row[list_of_decision_stumps[i][j][1]])
            if s > max_value:
                max_value = s
                tag = predictions[i]
        res.append(tag)
    return res
