<<<<<<< HEAD
 # -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 12:53:53 2019

@author: chris
"""
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
from scipy import ndimage as ndimage
from PIL import Image, ImageOps


#--------------------Data Loading and Splitting ---------------------------------
def get_data_loader(batch_size):

    train_path = 'trainData'
    val_path = 'trainData'
#test_path = 'testData'

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainSet = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True)

    valSet = torchvision.datasets.ImageFolder(root=val_path, transform=transform)
    val_data_loader = torch.utils.data.DataLoader(valSet, batch_size=batch_size, shuffle=True)

#    testSet = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
 #   test_data_loader  = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=True)
    return train_data_loader ,val_data_loader #,test_data_loader



#--------------------Base Model----------------------------------------------------

class BaseModel(nn.Module):
    def __init__(self, input_size):
        super(BaseModel, self).__init__()
        self.name = "Base"
        self.input_size = ((input_size - 2)/2)
        self.conv1 = nn.Conv2d(3, 5, 3)
        self.conv2 = nn.Conv2d(5, 7, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(7,10,3)
        self.conv4 = nn.Conv2d(10, 12, 6)
        self.fc1 = nn.Linear(int(12 * 140 * 140), 30000)
        self.fc2 = nn.Linear(30000,10000)
        self.fc3 = nn.Linear(10000,1000)
        self.fc4 = nn.Linear(1000,100)
        self.fc5 = nn.Linear(100,2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1,int(10*77*77))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x


#-------------------Train Loop (Ft. Get Accuracy & Plotting)----------------------------------------



def get_accuracy(model,set_, batch_size):
    batch_size=16
    label_ = [0]*(batch_size*2)
    for i in range(batch_size,batch_size*2):
        label_[i] = 1

    label = torch.tensor(label_).cuda()

    trainSet_,valSet_ = get_data_loader(batch_size)
    if set_ == "train":
        data_ = trainSet_
    elif set_ == "val":
        data_ = valSet_


    correct = 0
    total = 0
    for img,batch in data_:
        img,batch=img.cuda(),batch.cuda()
        if(len(batch)==batch_size):

            b = torch.split(img,600,dim=3)
            img = torch.cat(b, 0)

            output = model(img).cuda()

            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item() #compute how many predictions were correct
            total += img.shape[0] #get the total ammount of predictions

    return correct / total



def train(mdl,epochs= 20,batch_size = 32,learning_rate =0.0001):
    mdl.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mdl.parameters(), lr=learning_rate, momentum=0.9)
    trainSet,valSet = get_data_loader(batch_size)
    train_acc, val_acc = [], []
    n = 0 # the number of iterations

    label_ = [0]*(batch_size*2)
    for i in range(batch_size,batch_size*2):
        label_[i] = 1

    label = torch.tensor(label_).cuda()

    print("--------------Starting--------------")



    for epoch in range(epochs):  # loop over the dataset multiple times
        t1 = t.time()
        itera = 0
        filteredimg=[]
        for img,batch in iter(trainSet):

            if(len(batch)!=batch_size):
                break
            img,batch=img.cuda(),batch.cuda()
            b = torch.split(img,600,dim=3)


            img = torch.cat(b, 0)

         #   print(label)

            itera += batch_size*2

            out = mdl(img)

            loss = criterion(out, label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
           # print(itera)
        # Calculate the statistics
        train_acc.append(get_accuracy(mdl,"train", batch_size))

     #   val_acc.append(get_accuracy(mdl,"val"))  # compute validation accuracy
        n += 1


        print("Epoch",n,"Done in:",t.time() - t1, "With Training Accuracy:",train_acc[-1])#, "And Validation Accuracy:",val_acc[-1])


        # Save the current model (checkpoint) to a file
        model_path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(mdl.name,batch_size,learning_rate,epoch)
        torch.save(mdl.state_dict(), model_path)

    iterations = list(range(1,epochs + 1))

    print("--------------Finished--------------")

    return iterations,train_acc #, val_acc



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
=======
 # -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 12:53:53 2019

@author: chris
"""
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
from scipy import ndimage as ndimage
from PIL import Image, ImageOps


#--------------------Data Loading and Splitting ---------------------------------
def get_data_loader(batch_size):

    train_path = 'trainData'
    val_path = 'trainData'
#test_path = 'testData'

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainSet = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True)

    valSet = torchvision.datasets.ImageFolder(root=val_path, transform=transform)
    val_data_loader = torch.utils.data.DataLoader(valSet, batch_size=batch_size, shuffle=True)

#    testSet = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
 #   test_data_loader  = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=True)
    return train_data_loader ,val_data_loader #,test_data_loader



#--------------------Base Model----------------------------------------------------

class BaseModel(nn.Module):
    def __init__(self, input_size):
        super(BaseModel, self).__init__()
        self.name = "Base"
        self.input_size = ((input_size - 2)/2)
        self.conv1 = nn.Conv2d(3, 5, 3)
        self.conv2 = nn.Conv2d(5, 7, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(7,10,3)
        self.conv4 = nn.Conv2d(10, 12, 6)
        self.fc1 = nn.Linear(int(12 * 140 * 140), 30000)
        self.fc2 = nn.Linear(30000,10000)
        self.fc3 = nn.Linear(10000,1000)
        self.fc4 = nn.Linear(1000,100)
        self.fc5 = nn.Linear(100,2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1,int(10*77*77))
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x


#-------------------Train Loop (Ft. Get Accuracy & Plotting)----------------------------------------



def get_accuracy(model,set_, batch_size):
    batch_size=16
    label_ = [0]*(batch_size*2)
    for i in range(batch_size,batch_size*2):
        label_[i] = 1

    label = torch.tensor(label_).cuda()

    trainSet_,valSet_ = get_data_loader(batch_size)
    if set_ == "train":
        data_ = trainSet_
    elif set_ == "val":
        data_ = valSet_


    correct = 0
    total = 0
    for img,batch in data_:
        img,batch=img.cuda(),batch.cuda()
        if(len(batch)==batch_size):

            b = torch.split(img,600,dim=3)
            img = torch.cat(b, 0)

            output = model(img).cuda()

            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item() #compute how many predictions were correct
            total += img.shape[0] #get the total ammount of predictions

    return correct / total



def train(mdl,epochs= 20,batch_size = 32,learning_rate =0.0001):
    mdl.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mdl.parameters(), lr=learning_rate, momentum=0.9)
    trainSet,valSet = get_data_loader(batch_size)
    train_acc, val_acc = [], []
    n = 0 # the number of iterations

    label_ = [0]*(batch_size*2)
    for i in range(batch_size,batch_size*2):
        label_[i] = 1

    label = torch.tensor(label_).cuda()

    print("--------------Starting--------------")



    for epoch in range(epochs):  # loop over the dataset multiple times
        t1 = t.time()
        itera = 0
        filteredimg=[]
        for img,batch in iter(trainSet):

            if(len(batch)!=batch_size):
                break
            img,batch=img.cuda(),batch.cuda()
            b = torch.split(img,600,dim=3)


            img = torch.cat(b, 0)

         #   print(label)

            itera += batch_size*2

            out = mdl(img)

            loss = criterion(out, label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
           # print(itera)
        # Calculate the statistics
        train_acc.append(get_accuracy(mdl,"train", batch_size))

     #   val_acc.append(get_accuracy(mdl,"val"))  # compute validation accuracy
        n += 1


        print("Epoch",n,"Done in:",t.time() - t1, "With Training Accuracy:",train_acc[-1])#, "And Validation Accuracy:",val_acc[-1])


        # Save the current model (checkpoint) to a file
        model_path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(mdl.name,batch_size,learning_rate,epoch)
        torch.save(mdl.state_dict(), model_path)

    iterations = list(range(1,epochs + 1))

    print("--------------Finished--------------")

    return iterations,train_acc #, val_acc



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
>>>>>>> 722e9cad46f02ab54b7c8ddde8ebef8d95712429
