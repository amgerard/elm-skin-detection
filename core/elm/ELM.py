# -*- coding: utf-8 -*-
__author__ = 'ivanvallesperez'

import numpy as np
from scipy.special import expit

def sigmoid(x):
    ex = np.exp(-x)
    return 1 / (1 + ex)

class ELMRegressor():
    def __init__(self, n_hidden_units):
        self.n_hidden_units = n_hidden_units

    def fit(self, X, labels):
        X = np.column_stack([X, np.ones([X.shape[0], 1])])
        self.random_weights = np.random.randn(X.shape[1], self.n_hidden_units)
        G = expit(X.dot(self.random_weights))
        # G = np.tanh(X.dot(self.random_weights))
        self.w_elm = np.linalg.pinv(G).dot(labels)

    def predict(self, X):
        X = np.column_stack([X, np.ones([X.shape[0], 1])])
        G = expit(X.dot(self.random_weights))
        #G = np.tanh(X.dot(self.random_weights))
        return G.dot(self.w_elm)

    def copy(self):
        elm = ELMRegressor(self.n_hidden_units)
        elm.random_weights = np.copy(self.random_weights)
        elm.w_elm = np.copy(self.w_elm)
        return elm
