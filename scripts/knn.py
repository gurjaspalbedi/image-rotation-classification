import pickle
import pandas as pd
import numpy as np
import scipy.spatial.distance as dist

from random import sample, randint
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score
from collections import Counter


class MyKNeighborsClassifier:

    def __init__(self, k=5):
        self.cache = {}
        self.k = k

    def fit(self, train_data, train_label, k=5):
        self.train_data = train_data
        self.train_label = train_label
        self.k = k

    def predict(self, test_data):
        d = defaultdict(list)
        res = []

        # get k nearest neighbors
        for i, test in enumerate(test_data):
            for j, train in enumerate(self.train_data):
                distance = self.cache.get((i, j)) or dist.cityblock(train, test)
                self.cache[(i, j)] = distance
                d[i].append((j, distance))
            d[i] = sorted(d[i], key=lambda x: x[1])[:self.k]

        # let em vote
        for k, v in d.items():
            labels = Counter([self.train_label[i,0] for i, _ in v])
            my_label = labels.most_common(n=1)[0][0]
            res.append(my_label)

        return np.array(res)
