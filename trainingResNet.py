
import definitions_res_test as d
import matplotlib.pyplot as plt
import numpy as np
import pickle


modelres5 = d.ResNet5()
iterations,train_acc,val_acc = d.train(mdl = modelres5,epochs = 30,batch_size = 32, learning_rate =0.001, weight_decay =0.001) #,val_acc
d.plot(iterations,train_acc, val_acc)

