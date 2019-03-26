# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 20:32:56 2019

@author: chris
"""

X = np.array([[1., 0.], [2., 1.], [0., 0.]])
y = np.array([0, 1, 2])
from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=0)
print(X,y)