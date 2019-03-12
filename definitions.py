#----------------------Imports------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import transforms
from torchvision import *
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import time as t
import torch.optim as optim
from PIL import Image, ImageOps


#--------------------Data Loading and Splitting ---------------------------------
def get_data_loader(batch_size):

    train_path = r'C:\Users\chris\OneDrive\Documents\3rd Year Labs\AI Project\trainData'
    val_path = r'C:\Users\chris\OneDrive\Documents\3rd Year Labs\AI Project\valData'
    test_path = r'C:\Users\chris\OneDrive\Documents\3rd Year Labs\AI Project\testData'
    
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainSet = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=False)

    valSet = torchvision.datasets.ImageFolder(root=val_path, transform=transform)
    val_data_loader = torch.utils.data.DataLoader(valSet, batch_size=batch_size, shuffle=False)

    testSet = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    test_data_loader  = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=False)
    return train_data_loader ,val_data_loader,test_data_loader


    
#--------------------Base Model----------------------------------------------------

class BaseModel(nn.Module):
    def __init__(self, input_size = 400):
        super(BaseModel, self).__init__()
        self.name = "Base"
        self.input_size = input_size
        self.conv = nn.Conv2d(3, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(5 * 99 * 99, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = self.pool(x)
        x = x.view(-1, 5 * 99 * 99)
        x = self.fc(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x
    
    
#-------------------Train Loop (Ft. Get Accuracy & Plotting)----------------------------------------
        


def get_accuracy(model):
    _,valSet_,__ = get_data_loader(189)
    data_ = valSet_
    correct = 0
    total = 0
    for imgs, labels in data_:
        output = model(imgs) 
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item() #compute how many predictions were correct
        total += imgs.shape[0] #get the total ammount of predictions
        break 
    return correct / total
        


def train(mdl,epochs= 20,batch_size = 64,learning_rate =0.1):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mdl.parameters(), lr=learning_rate, momentum=0.9)
    trainSet,valSet,testSet = get_data_loader(batch_size)
    train_acc, val_acc = [], []
    n = 0 # the number of iterations
    print("--------------Starting--------------")
    for epoch in range(epochs):  # loop over the dataset multiple times
        t1 = t.time()
        correct = 0
        total = 0
        
        for img, label in iter(trainSet):

            out = mdl(img)
           
            #---------Get statistics for train---------
            pred = out.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item()
            total += img.shape[0]
            #-----------------Done------------------
            
            loss = criterion(out, label)  
            loss.backward() 
            
            optimizer.step()  
            optimizer.zero_grad()     
  
        # Calculate the statistics
        train_acc.append(correct/total)
        
        val_acc.append(get_accuracy(mdl))  # compute validation accuracy
        n += 1

        
        print("Epoch",n,"Done in:",t.time() - t1, "With Training Accuracy:",train_acc[-1], "And Validation Accuracy:",val_acc[-1])


        # Save the current model (checkpoint) to a file
        model_path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(mdl.name,batch_size,learning_rate,epoch)
        torch.save(mdl.state_dict(), model_path)

    iterations = list(range(1,epochs + 1))
    
    print("--------------Finished--------------")
    
    return iterations,train_acc, val_acc



def plot(iterations,train_acc, val_acc):
    plt.title("Training Curve")
    plt.plot(iterations, train_acc, label="Train")
    plt.plot(iterations, val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))


