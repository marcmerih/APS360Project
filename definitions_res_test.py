
# -*- coding: utf-8 -*-
"""
Created on Thur Mar 21 08:20:07 2019

@author: marc
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

from resnet import *
import grid_search as gd
import pickle

import torchvision.models as models
resnet18 = resnet18(pretrained=True)


#--------------------Data Loading and Splitting ---------------------------------
def get_data_loader(batch_size):

    train_path = r'trainData'
    val_path = r'valData'
    #test_path = r'testData'

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainSet = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True)

    valSet = torchvision.datasets.ImageFolder(root=val_path, transform=transform)
    val_data_loader = torch.utils.data.DataLoader(valSet, batch_size=batch_size, shuffle=True)

    #testSet = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    #test_data_loader  = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=True)
    return train_data_loader , val_data_loader #test_data_loader



#--------------------Base Model----------------------------------------------------

class BaseModel(nn.Module):
    def __init__(self, input_size):
        super(BaseModel, self).__init__()
        self.name = "Base"
        self.input_size = ((input_size - 2)/2)
        self.conv1 = nn.Conv2d(3, 5, 3)
        self.conv2 = nn.Conv2d(5, 7, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(int(7 * 147 * 147), 1000)
        self.fc2 = nn.Linear(1000,2)


    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,int(7*147 * 147) )
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x



class ResNet(nn.Module):
    def __init__(self,):
        super(ResNet, self).__init__()
        self.name = "ResNet"
        self.fc1 = nn.Linear( 86528,300)
        self.fc2 = nn.Linear( 300,100)
        self.fc3 = nn.Linear( 100,32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        #print(x.size())
        #x = x.view(-1, 86528)
        #print(x.size())
        x = F.relu(self.fc1(x))
        #print(x.size())
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        #x = x.squeeze(1)
        #print(x.size(),"\n\n\n")
        return x.cuda()

class ResNet5(nn.Module):
    def __init__(self,dropout):
        super(ResNet5, self).__init__()
        self.name = "ResNet"
        self.fc1 = nn.Linear(86528,400)
        self.fc2 = nn.Linear(400,200)
        self.fc3 = nn.Linear(200,90)
        self.fc4 = nn.Linear(90, 32)
        self.fc5 = nn.Linear(32, 2)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        
    def forward(self, x):
        #print(x.size())
        #x = x.view(-1, 86528)
        #print(x.size())
        x = F.relu(self.dropout1(self.fc1(x)))
        #print(x.size())
        x = F.relu(self.dropout2(self.fc2(x)))
        x = F.relu(self.dropout3(self.fc3(x)))
        x = F.relu(self.dropout4(self.fc4(x)))
        x = self.fc5(x)
        #x = x.squeeze(1)
        #print(x.size(),"\n\n\n")
        return x


#-------------------Train Loop (Ft. Get Accuracy & Plotting)----------------------------------------

def get_accuracy(model,set_,batch_size):
    label_ = [0]*(batch_size*2)
    for i in range(1,batch_size*2,2):
        label_[i] = 1


    label = torch.tensor(label_).cuda()

    trainSet_,valSet_ = get_RN_data_loader(batch_size)
    if set_ == "train":
        data_ = trainSet_
    elif set_ == "val":
        data_ = valSet_

    model = model.cuda()
    correct = 0
    total = 0
    for res, batch in data_:
     #   b = torch.split(img,600,dim=3)
      #  img = torch.cat(b, 0)
        res = res.cuda()
        res = res.view(-1, 86528).cuda()
        output = model(res).cuda()
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(label.view_as(pred)).sum().item() #compute how many predictions were correct
        total += res.shape[0] #get the total ammount of predictions
        break

    return correct / total
#
from sklearn.utils import shuffle

def train(mdl,epochs= 20,batch_size = 32,learning_rate =0.001, weight_decay = 0.001):
    #criterion = nn.BCEWithLogitsLoss()
    mdl.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mdl.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    #trainSet,valSet = get_data_loader(batch_size)
    trainSet,valSet = get_RN_data_loader(batch_size)
    train_acc, val_acc = [], []
    n = 0 # the number of iterations

#    label_ = [0]*(batch_size)
#    label_.extend([1]*(batch_size))
#
    label_ = [0]*(batch_size*2)
    for i in range(1,batch_size*2,2):
        label_[i] = 1


    label = torch.tensor(label_).cuda()
    #mdl = mdl.cuda()
    print("--------------Starting--------------")



    for epoch in range(epochs):  # loop over the dataset multiple times
        t1 = t.time()
        for res,batch in iter(trainSet):

            if len(batch)==batch_size:

                res = res.view(-1, 86528).cuda()

                out = mdl(res).cuda()

                #print(out.size())
                loss = criterion(out, label)
                loss.backward()

                optimizer.step()
                optimizer.zero_grad()
                #print("Iteration Done")

        # Calculate the statistics
        train_acc.append(get_accuracy(mdl,"train",batch_size = 155))

        val_acc.append(get_accuracy(mdl,"val",batch_size = 155))  # compute validation accuracy
        n += 1


        print("Epoch",n,"Done in:",t.time() - t1, "With Training Accuracy:",train_acc[-1], "And Validation Accuracy:",val_acc[-1])


            # Save the current model (checkpoint) to a file
           # model_path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(mdl.name,batch_size,learning_rate,epoch)
           # torch.save(mdl.state_dict(), model_path)

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


#RtrainSet,RvalSet = get_data_loader(1)

def RNFeatures(dataSet,type_):
    data_ = dataSet
    i = 0
    j= 0

    for img, label in data_:
        b = torch.split(img,600,dim=3)

        img = torch.cat(b, 0)
        print(img.size())
        output = resnet18(img)



        if type_ == 'train':
            tensor_path = "tensor_set{0}_number{1}".format(type_,i)

            torch.save(output,r'C:/Users/chris/OneDrive/Documents/GitHub/APS360Project/RtrainData/'+tensor_path)
            i+=1
        elif type_ == 'val':
            tensor_path = "tensor_set{0}_number{1}".format(type_,j)

            torch.save(output,r'C:/Users/chris/OneDrive/Documents/GitHub/APS360Project/RvalData/'+tensor_path)
            j+=1
#print("Train")
#RNFeatures(RtrainSet,'train')
#print("Validation")
#RNFeatures(RvalSet,'val')

def get_RN_data_loader(batch_size):

    train_path = r'RtrainData'
    val_path = r'RvalData'
    #train_path = r'C:/Users/chris/OneDrive/Documents/GitHub/APS360Project/RtrainData'
    #val_path = r'C:/Users/chris/OneDrive/Documents/GitHub/APS360Project/RvalData'


    trainSet = torchvision.datasets.DatasetFolder(root=train_path,loader = torch.load,extensions = list(['']))
    train_data_loader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True)

    valSet = torchvision.datasets.DatasetFolder(root=val_path,loader = torch.load,extensions = list(['']))
    val_data_loader = torch.utils.data.DataLoader(valSet, batch_size=batch_size, shuffle=True)

    #testSet = torchvision.datasets.DatasetFolder(root=test_path,loader = torch.load,extensions = list(['']))
    #test_data_loader  = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=True)

    return train_data_loader ,val_data_loader

#RtrainSet,RvalSet = get_RN_data_loader(16)







