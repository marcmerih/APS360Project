import definitions as d
import matplotlib.pyplot as plt
import numpy as np

#---------------------Data Visualization Examples----------------------------------
    

#trainTemp,valTemp,testTemp = d.get_data_loader(1)
#k = 0
#for images, labels in trainTemp:
#    # since batch_size = 1, there is only 1 image in `images`
#    image = images[0]
#    # place the colour channel at the end, instead of at the beginning
#    img = np.transpose(image, [1,2,0])
#    # normalize pixel intensity values to [0, 1]
#    img = img / 2 + 0.5
#    plt.subplot(3, 5, k+1)
#    plt.axis('off')
#    plt.imshow(img)
#
#    k += 1
#    if k > 14:
#        break
    
modelBase = d.BaseModel(input_size = 400)
iterations,train_acc, val_acc = d.train(mdl = modelBase,epochs = 5)
d.plot(iterations,train_acc, val_acc)
