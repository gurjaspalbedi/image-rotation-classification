#!/usr/bin/env python3

import pickle
import pandas as pd
import numpy as np
import scipy.spatial.distance as dist

from random import sample, randint
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score
from collections import Counter



# idea and partial code adapted from Google Developers
# https://www.youtube.com/watch?v=LDRbO9a6XPU

def count_class_numbers(rows):
    """ count the numbers of each class """
    c = Counter()
    for row in rows:
        c[row[-1]] += 1
    return c


def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    counts = count_class_numbers(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = float(counts[lbl]) / len(rows)
        impurity -= prob_of_lbl**2
    return impurity


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def find_best_split(rows):
    best_gain = float('-inf')
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1

    for col in range(n_features):
        values = set([row[col] for row in rows])
        for val in values:
            question = Question(col, val)

            true_rows, false_rows = partition(rows, question)
            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            current_gain = info_gain(true_rows, false_rows, current_uncertainty)
            if current_gain >= best_gain:
                best_gain, best_question = current_gain, question

    return best_gain, best_question


class Question:

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        val = example[self.column]
        return val >= self.value



class EndNode:
    def __init__(self, rows):
        self.predictions = count_class_numbers(rows)

    def get_element(self):
        return list(self.predictions.keys())[0]


class DecisionNode:

    def __init__(self, question=None, true_branch=None, false_branch=None):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


class MyDecisionTreeClassifier:

    def __init__(self):
        self.my_tree = DecisionNode()

    def build_tree(self, rows):
        """Builds the tree.

        Rules of recursion:
        1) Believe that it works.
        2) Start by checking for the base case (no further information gain).
        3) Prepare for giant stack traces.
        """
        gain, question = find_best_split(rows)
        if gain == 0 or not question:
            return EndNode(rows)

        true_rows, false_rows = partition(rows, question)
        true_branch = self.build_tree(true_rows)
        false_branch = self.build_tree(false_rows)
        return DecisionNode(question, true_branch, false_branch)

    def predict_one(self, row, node):
        if isinstance(node, EndNode):
            return node.get_element()

        if node.question.match(row):
            return self.predict_one(row, node.true_branch)
        return self.predict_one(row, node.false_branch)

    def fit(self, train_data, train_label):
        training_data = np.concatenate((train_data, train_label), axis=1)
        self.my_tree = self.build_tree(training_data)

    def predict(self, test_data, test_label):
        testing_data = np.concatenate((test_data, test_label), axis=1)
        return np.array([self.predict_one(row, self.my_tree) for row in testing_data])


def dump_tree(tree_id, rowids, colids, clf):
    tree = {
        'rowids': rowids,
        'colids': colids,
        'clf': clf
    }
    return tree


def train_random_forest_model(train_data, train_label):
    all_rowids = list(range(train_data.shape[0]))
    all_colids = list(range(train_data.shape[1]))

    trees = []
    for tree_id in range(100):
        print(tree_id)
        rowids = tuple(sample(all_rowids, randint(200, 1000)))
        colids = tuple(sample(all_colids, randint(8, 20)))
        clf = MyDecisionTreeClassifier()
        clf.fit(train_data[rowids,:][:, colids], train_label[rowids,:])
        tree = dump_tree(tree_id, rowids, colids, clf)
        trees.append(tree)
    return trees


def predict_by_random_forest_model(test_data, test_label, model_loc):
    f = open(model_loc, 'rb')
    models = pickle.loads(f.read())

    res = []
    for model in models:
        clf = model['clf']
        rowids, colids = model['rowids'], model['colids']
        pred = clf.predict(test_data[:,colids], test_label)
        res.append(pred)

    my = np.array(res)
    final = []
    for i in range(my.shape[1]):
        c = Counter(my[:,i].ravel())
        val = c.most_common(n=1)[0][0]
        final.append(val)
    pred = np.array(final)
    return pred

