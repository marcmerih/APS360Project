# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:05:53 2019

@author: chris
"""

import time
import numpy as np


def FS(A,b,y,N):
    n = len(A)
    y_ = b[0] / A[0][0]
    y[N-n] = y_
    if n == 1:
        return y
        
    B2 = b[1:]
    B2 = B2 - (y_)*np.transpose(A)[0][1:]
    A_ = np.transpose(A[1:])[1:]
    return FS(np.transpose(A_),B2,y,N)
    

def lab09_palumb21(A,b,y,N):
    '''
    Your function should solve Ay = b equation for given A, b, y, N variables to find
    y using two different methods (Note that input y is a vector of zeros). First method is
    to implement the Forward substitution (FS) algorithm, and second method is to use the
    numpy.linalg.solve(A,b) function. For each method, you should record the computational time, 
    store them in time1 and time2 and return the following values:
    '''
    
    
    #Forward substitution (FS)for inverting a lower triangular matrix
    t1_ = time.time()
    y1 = FS(A,b,y,N)
    t1 = time.time() - t1_
    
    
    t2_ = time.time()
    y2 = np.linalg.solve(A,b)
    t2 = time.time() - t2_
    
    
    
    
    return y1,t1,t2

A = [[-2,0,0,0],[1,-1,0,0],[0,-7,3,0],[-2,8,6,5]]
b = [1,2,3,4]
y = [0,0,0,0]
N = 4
print(lab09_palumb21(A,b,y,N))
