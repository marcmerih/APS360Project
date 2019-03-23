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
from PIL import Image, ImageOps
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input, Flatten, Dense
from keras.models import Model
import numpy as np


#--------------------Data Loading and Splitting ---------------------------------
def get_data_loader(batch_size):

    train_path = r'trainData'
    val_path = r'valData'
    test_path = r'testData'

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainSet = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True)

    valSet = torchvision.datasets.ImageFolder(root=val_path, transform=transform)
    val_data_loader = torch.utils.data.DataLoader(valSet, batch_size=batch_size, shuffle=True)

    testSet = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    test_data_loader  = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=True)
    return train_data_loader ,val_data_loader,test_data_loader



#--------------------Base Model----------------------------------------------------

class Model(nn.Module):
    def __init__(self, input_size):
        super(Model, self).__init__()
        self.name = "Base"
        self.input_size = Input(shape=(3,600,600),name = 'image_input')
        model_vgg16_conv = VGG16(weights='imagenet', include_top=False)
        output_vgg16_conv = self.model_vgg16_conv(input_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.x1 = Flatten(name='flatten')(output_vgg16_conv)
        self.x2 = Dense(4096, activation='relu', name='fc1')(x)
        self.x3 = Dense(4096, activation='relu', name='fc2')(x)
        self.x4 = Dense(2, activation='softmax', name='predictions')(x)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(model_vgg16_conv)))
        x = self.pool(F.relu(self.conv2(output_vgg16_conv)))
        x = x.view(-1,int(7*147 * 147) )
        x = self.fc1(x1)
        x = self.fc2(x2)
        x = self.fc2(x3)
        x = self.fc2(x4)
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
