# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:04:14 2019

@author: chris
"""
import definitions_v2 as d
import matplotlib.pyplot as plt
import numpy as np
import pickle


modelBase = d.BaseModel(input_size = 600)
iterations,train_acc = d.train(mdl = modelBase,epochs = 40,batch_size = 16, learning_rate =0.0003) #,val_acc
#d.plot(iterations,train_acc, val_acc)

