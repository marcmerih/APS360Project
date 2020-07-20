<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:04:14 2019
@author: chris
"""
import definitions_v4_vgg as d
import matplotlib.pyplot as plt
import numpy as np
import pickle


model = d.VGG()
iterations,train_acc, val_acc = d.train(mdl = model,epochs = 10,batch_size = 32)
d.plot(iterations,train_acc, val_acc)
=======
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 13:04:14 2019
@author: chris
"""
import definitions_v4_vgg as d
import matplotlib.pyplot as plt
import numpy as np
import pickle


model = d.VGG()
iterations,train_acc, val_acc = d.train(mdl = model,epochs = 10,batch_size = 32)
d.plot(iterations,train_acc, val_acc)
>>>>>>> 722e9cad46f02ab54b7c8ddde8ebef8d95712429
