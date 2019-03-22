# -*- coding: utf-8 -*-
"""
Created on Thur Mar 21 08:43:49 2019

@author: marc
"""
import definitions_v3_alex as da
import matplotlib.pyplot as plt
import numpy as np
import pickle


modelBase = da.AlexNet()
iterations,train_acc = da.alextrain(modelBase,batch_size = 32, num_epochs = 10,lr = 0.0001)
da.plot(iterations,train_acc)
