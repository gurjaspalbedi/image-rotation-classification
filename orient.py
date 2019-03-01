#!/usr/bin/env python3

import sys
import pickle
import pandas as pd
import numpy as np
import scipy.spatial.distance as dist

from random import sample, randint
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score
from collections import Counter
from scripts.random_forest import train_random_forest_model, predict_by_random_forest_model
from scripts.knn import MyKNeighborsClassifier
from scripts.adaboost import train_adaboost, predict_adaboost


if __name__ == '__main__':
    train_data_loc = 'data/train-data.txt'

    mode = sys.argv[1]
    test_data_loc = 'data/test-data.txt'
    model_loc = sys.argv[3]
    model_name = sys.argv[4]

    train_df = pd.read_csv(train_data_loc, header=None, delim_whitespace=True)
    test_df = pd.read_csv(test_data_loc, header=None, delim_whitespace=True)
    train_data = np.array(np.array(train_df)[:,2:], dtype=int)
    train_id = np.array(np.array(train_df)[:,0].T)
    train_label = np.array(np.array(train_df)[:,1].T, dtype=int)
    train_label.resize((train_label.shape[0], 1))
    test_data = np.array(np.array(test_df)[:,2:], dtype=int)
    test_id = np.array(np.array(test_df)[:,0].T)
    test_label = np.array(np.array(test_df)[:,1].T, dtype=int)
    test_label.resize((test_label.shape[0], 1))

    if model_name == 'nearest':
        if mode == 'train':
            knn_clf = MyKNeighborsClassifier(k=20)
            knn_clf.fit(train_data, train_label)
            model = knn_clf
            f = open(model_loc, 'wb')
            pickle.dump(model, f)
        else:
            f = open(model_loc, 'rb')
            model = pickle.loads(f.read())
            f.close()
            pred = model.predict(test_data)
            f = open('output.txt', 'w')
            for photo_id, orientation in zip(test_id, pred):
                f.write('%s %s\n' % ( photo_id, orientation))

    if model_name == 'adaboost':
        if mode == 'train':
            model = train_adaboost(train_data, train_label)
            f = open(model_loc, 'wb')
            pickle.dump(model, f)
        else:
            correct = 0
            pred = predict_adaboost(test_data, model_loc)
            f = open('output.txt', 'w')
            for photo_id, orientation in zip(test_id, pred):
                f.write('%s %s\n' % ( photo_id, orientation))
            for i,item in enumerate(pred):
                if pred[i] == test_label[i]:
                    correct = correct + 1
            print(correct/ len(test_data))
                    

    if model_name in ['forest', 'best']:
        if mode == 'train':
            model = train_random_forest_model(train_data, train_label)
            f = open('models/' + model_name + '_model.txt', 'wb')
            pickle.dump(model, f)
        else:
            pred = predict_by_random_forest_model(test_data, test_label, model_loc)
            f = open('output.txt', 'w')
            for photo_id, orientation in zip(test_id, pred):
                f.write('%s %s\n' % ( photo_id, orientation))
