
import definitions_res as d
import matplotlib.pyplot as plt
import numpy as np
import pickle


modelres = d.ResNet()
iterations,train_acc = d.train(mdl = modelres,epochs = 20,batch_size = 16) #,val_acc
#d.plot(iterations,train_acc, val_acc)

