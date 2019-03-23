<<<<<<< HEAD
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


model = d.Model(input_size = 600)
iterations,train_acc, val_acc = d.train(mdl = modelBase,epochs = 10,batch_size = 32)
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


model = d.Model(input_size = 600)
iterations,train_acc, val_acc = d.train(mdl = modelBase,epochs = 10,batch_size = 32)
d.plot(iterations,train_acc, val_acc)
>>>>>>> a23278a45f045abb6a38b1d574e81763a2298e25
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


model = d.Model(input_size = 600)
iterations,train_acc, val_acc = d.train(mdl = modelBase,epochs = 10,batch_size = 32)
d.plot(iterations,train_acc, val_acc)
>>>>>>> 2044b5a1553748ec92cbae85c5296a6a6ce38122
