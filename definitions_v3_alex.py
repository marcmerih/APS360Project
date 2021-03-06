<<<<<<< HEAD
<<<<<<< HEAD
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
import torchvision.models as tvm
from a3code import AlexNetFeatures

from resnet import * as


import torchvision.models as models
resnet101 = resnet.resnet101(pretrained=True)

#--------------------Data Loading and Splitting ---------------------------------
def get_data_loader(batch_size):

    train_path = r'trainData'
    #val_path = r'valData'
    #test_path = r'testData'

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainSet = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True)

    #valSet = torchvision.datasets.ImageFolder(root=val_path, transform=transform)
    #val_data_loader = torch.utils.data.DataLoader(valSet, batch_size=batch_size, shuffle=True)

    #testSet = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    #test_data_loader  = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=True)
    return train_data_loader #, val_data_loader, #test_data_loader



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
        self.fc1 = nn.Linear( 256* 6 * 6, 32)
        self.fc2 = nn.Linear(32, 9)

    def forward(self, x):
        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#-------------------Train Loop (Ft. Get Accuracy & Plotting)----------------------------------------



def get_accuracy(model,set_):
    label_ = [0]*(300)
    for i in range(0,300,2):
        label_[i] = 1

    label = torch.tensor(label_)

    trainSet_,valSet_,__ = get_data_loader(150)
    if set_ == "train":
        data_ = trainSet_
    #elif set_ == "val":
        #data_ = valSet_


    correct = 0
    total = 0
    for img, _ in data_:
        b = torch.split(img,600,dim=3)
        img = torch.cat(b, 0)


        output = model(img)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(label.view_as(pred)).sum().item() #compute how many predictions were correct
        total += img.shape[0] #get the total ammount of predictions
        break

    return correct / total



def train(mdl,epochs= 20,batch_size = 32,learning_rate =0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mdl.parameters(), lr=learning_rate, momentum=0.9)
    trainSet = get_data_loader(batch_size)
    train_acc, val_acc = [], []
    n = 0 # the number of iterations

    label_ = [0]*(batch_size*2)
    for i in range(0,batch_size*2,2):
        label_[i] = 1

    label = torch.tensor(label_)

    print("--------------Starting--------------")



    for epoch in range(epochs):  # loop over the dataset multiple times



        t1 = t.time()

        itera = 0
        for img,_ in iter(trainSet):


            b = torch.split(img,600,dim=3)


            img = torch.cat(b, 0)

            print(img.size())


            itera += batch_size*2
            res = resnet(img)
            out = mdl(res)

            loss = criterion(out, label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            print(itera)
        break
        # Calculate the statistics
        train_acc.append(get_accuracy(mdl,"train"))

        #val_acc.append(get_accuracy(mdl,"val"))  # compute validation accuracy
        n += 1


        print("Epoch",n,"Done in:",t.time() - t1, "With Training Accuracy:",train_acc[-1])#, "And Validation Accuracy:",val_acc[-1])


        # Save the current model (checkpoint) to a file
        model_path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(mdl.name,batch_size,learning_rate,epoch)
        torch.save(mdl.state_dict(), model_path)

    iterations = list(range(1,epochs + 1))

    print("--------------Finished--------------")

    return iterations,train_acc, val_acc



def plot(iterations,train_acc, val_acc):
    plt.title("Training Curve")
    plt.plot(iterations, train_acc, label="Train")
    #plt.plot(iterations, val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    #print("Final Validation Accuracy: {}".format(val_acc[-1]))

from a3code import AlexNetFeatures
myfeature_model = AlexNetFeatures() #loads pre-trained weights
atrain_loader = get_data_loader(1)
a=0
for img, l in atrain_loader:
    features=myfeature_model(img)
    i=str(a)
    label=l[0].item()
    print(features.shape)
    break

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Linear(256*6*6, 50)
        self.layer2 = nn.Linear(50, 20)
        self.layer3 = nn.Linear(20, 2)
    def forward(self, img):
        flattened = img.view(-1,256*6*6)
        activation1 = F.relu(self.layer1(flattened))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return output

def get_alex_data_loader(batch_size, shuffle=True):
    np.random.seed(1000) # Fixed numpy random seed for reproducible shuffling

    train_sampler = torchvision.datasets.DatasetFolder(root='trainData', loader=torch.load, extensions=list(['']))
    alex_train_loader = torch.utils.data.DataLoader(train_sampler, batch_size=batch_size, shuffle=shuffle)
    #val_sampler =  torchvision.datasets.DatasetFolder(root='valData', loader=torch.load, extensions=list(['']))
    #alex_val_loader = torch.utils.data.DataLoader(val_sampler, batch_size=batch_size, shuffle=shuffle)

    #test_sampler =  torchvision.datasets.DatasetFolder(root='testData', loader=torch.load, extensions=list(['']))
        #alex_test_loader = torch.utils.data.DataLoader(test_sampler, batch_size=batch_size, shuffle=shuffle)

    return alex_train_loader#, alex_train_loader, alex_test_loader


def get_alex_accuracy(model, train=True):
    batch_size=16
    label_ = [0]*(batch_size*2)
    for i in range(0,batch_size*2,2):
        label_[i] = 1

    if train:
        data = alex_train_loader
    #else:
    #    train, data, test = alex_train_loader, alex_val_loader, alex_test_loader = get_alex_data_loader(1)

    correct = 0
    total = 0
    for inputs, labels in data:
        output = model(inputs) # We don't need to run F.softmax
        pred = output.argmax()
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += inputs.shape[0]
    return correct / total

def alextrain(model, batch_size=32, num_epochs=15, lr=0.0001):

    atrain_loader = get_alex_data_loader(1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    iters, losses, train_acc, val_acc = [], [], [], []

    # training
    n = 0 # the number of iterations
    label_ = [0]*(batch_size*2)
    for i in range(0,batch_size*2,2):
        label_[i] = 1

    label = torch.tensor(label_).cuda()

    for epoch in range(num_epochs):
        correct=0
        total=0
        #to store the iteration that reached 100% accuracy first.
        for inputs, labels in iter(atrain_loader):
            out = model(inputs)             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch
            # save the current training information
            pred = out.argmax()
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += inputs.shape[0]
        train_acc.append(correct/total) # compute training accuracy
        #val_acc.append(get_alex_accuracy(model, train=False))  # compute validation accuracy
        n += 1
        iters.append(n)


    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    #plt.plot(iters, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    #print("Final Validation Accuracy: {}".format(val_acc[-1]))

    return iterations, train_acc
=======
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

from resnet import * as


import torchvision.models as models
resnet101 = resnet.resnet101(pretrained=True)

#--------------------Data Loading and Splitting ---------------------------------
def get_data_loader(batch_size):

    train_path = r'trainData'
    #val_path = r'valData'
    #test_path = r'testData'

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainSet = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True)

    #valSet = torchvision.datasets.ImageFolder(root=val_path, transform=transform)
    #val_data_loader = torch.utils.data.DataLoader(valSet, batch_size=batch_size, shuffle=True)

    #testSet = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    #test_data_loader  = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=True)
    return train_data_loader #, val_data_loader, #test_data_loader



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
        self.fc1 = nn.Linear( 256* 6 * 6, 32)
        self.fc2 = nn.Linear(32, 9)

    def forward(self, x):
        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#-------------------Train Loop (Ft. Get Accuracy & Plotting)----------------------------------------



def get_accuracy(model,set_):
    label_ = [0]*(300)
    for i in range(0,300,2):
        label_[i] = 1

    label = torch.tensor(label_)

    trainSet_,valSet_,__ = get_data_loader(150)
    if set_ == "train":
        data_ = trainSet_
    #elif set_ == "val":
        #data_ = valSet_


    correct = 0
    total = 0
    for img, _ in data_:
        b = torch.split(img,600,dim=3)
        img = torch.cat(b, 0)


        output = model(img)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(label.view_as(pred)).sum().item() #compute how many predictions were correct
        total += img.shape[0] #get the total ammount of predictions
        break

    return correct / total



def train(mdl,epochs= 20,batch_size = 32,learning_rate =0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mdl.parameters(), lr=learning_rate, momentum=0.9)
    trainSet = get_data_loader(batch_size)
    train_acc, val_acc = [], []
    n = 0 # the number of iterations

    label_ = [0]*(batch_size*2)
    for i in range(0,batch_size*2,2):
        label_[i] = 1

    label = torch.tensor(label_)

    print("--------------Starting--------------")



    for epoch in range(epochs):  # loop over the dataset multiple times



        t1 = t.time()

        itera = 0
        for img,_ in iter(trainSet):


            b = torch.split(img,600,dim=3)


            img = torch.cat(b, 0)

            print(img.size())


            itera += batch_size*2
            res = resnet(img)
            out = mdl(res)

            loss = criterion(out, label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            print(itera)
        break
        # Calculate the statistics
        train_acc.append(get_accuracy(mdl,"train"))

        #val_acc.append(get_accuracy(mdl,"val"))  # compute validation accuracy
        n += 1


        print("Epoch",n,"Done in:",t.time() - t1, "With Training Accuracy:",train_acc[-1])#, "And Validation Accuracy:",val_acc[-1])


        # Save the current model (checkpoint) to a file
        model_path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(mdl.name,batch_size,learning_rate,epoch)
        torch.save(mdl.state_dict(), model_path)

    iterations = list(range(1,epochs + 1))

    print("--------------Finished--------------")

    return iterations,train_acc, val_acc



def plot(iterations,train_acc, val_acc):
    plt.title("Training Curve")
    plt.plot(iterations, train_acc, label="Train")
    #plt.plot(iterations, val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    #print("Final Validation Accuracy: {}".format(val_acc[-1]))

from a3code import AlexNetFeatures
myfeature_model = AlexNetFeatures() #loads pre-trained weights
atrain_loader = get_data_loader(1)
a=0
for img, l in atrain_loader:
    features=myfeature_model(img)
    i=str(a)
    label=l[0].item()
    print(features.shape)
    break

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Linear(256*6*6, 50)
        self.layer2 = nn.Linear(50, 20)
        self.layer3 = nn.Linear(20, 2)
    def forward(self, img):
        flattened = img.view(-1,256*6*6)
        activation1 = F.relu(self.layer1(flattened))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return output

def get_alex_data_loader(batch_size, shuffle=True):
    np.random.seed(1000) # Fixed numpy random seed for reproducible shuffling

    train_sampler = torchvision.datasets.DatasetFolder(root='trainData', loader=torch.load, extensions=list(['']))
    alex_train_loader = torch.utils.data.DataLoader(train_sampler, batch_size=batch_size, shuffle=shuffle)
    #val_sampler =  torchvision.datasets.DatasetFolder(root='valData', loader=torch.load, extensions=list(['']))
    #alex_val_loader = torch.utils.data.DataLoader(val_sampler, batch_size=batch_size, shuffle=shuffle)

    #test_sampler =  torchvision.datasets.DatasetFolder(root='testData', loader=torch.load, extensions=list(['']))
        #alex_test_loader = torch.utils.data.DataLoader(test_sampler, batch_size=batch_size, shuffle=shuffle)

    return alex_train_loader#, alex_train_loader, alex_test_loader


def get_alex_accuracy(model, train=True):
    batch_size=16
    label_ = [0]*(batch_size*2)
    for i in range(0,batch_size*2,2):
        label_[i] = 1

    if train:
        data = alex_train_loader
    #else:
    #    train, data, test = alex_train_loader, alex_val_loader, alex_test_loader = get_alex_data_loader(1)

    correct = 0
    total = 0
    for inputs, labels in data:
        output = model(inputs) # We don't need to run F.softmax
        pred = output.argmax()
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += inputs.shape[0]
    return correct / total

def alextrain(model, batch_size=32, num_epochs=15, lr=0.0001):

    atrain_loader = get_alex_data_loader(1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    iters, losses, train_acc, val_acc = [], [], [], []

    # training
    n = 0 # the number of iterations
    label_ = [0]*(batch_size*2)
    for i in range(0,batch_size*2,2):
        label_[i] = 1

    label = torch.tensor(label_).cuda()

    for epoch in range(num_epochs):
        correct=0
        total=0
        #to store the iteration that reached 100% accuracy first.
        for inputs, labels in iter(atrain_loader):
            out = model(inputs)             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch
            # save the current training information
            pred = out.argmax()
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += inputs.shape[0]
        train_acc.append(correct/total) # compute training accuracy
        #val_acc.append(get_alex_accuracy(model, train=False))  # compute validation accuracy
        n += 1
        iters.append(n)


    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    #plt.plot(iters, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    #print("Final Validation Accuracy: {}".format(val_acc[-1]))

    return iterations, train_acc
>>>>>>> 2044b5a1553748ec92cbae85c5296a6a6ce38122
=======
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

from resnet import * as


import torchvision.models as models
resnet101 = resnet.resnet101(pretrained=True)

#--------------------Data Loading and Splitting ---------------------------------
def get_data_loader(batch_size):

    train_path = r'trainData'
    #val_path = r'valData'
    #test_path = r'testData'

    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainSet = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    train_data_loader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True)

    #valSet = torchvision.datasets.ImageFolder(root=val_path, transform=transform)
    #val_data_loader = torch.utils.data.DataLoader(valSet, batch_size=batch_size, shuffle=True)

    #testSet = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    #test_data_loader  = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=True)
    return train_data_loader #, val_data_loader, #test_data_loader



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
        self.fc1 = nn.Linear( 256* 6 * 6, 32)
        self.fc2 = nn.Linear(32, 9)

    def forward(self, x):
        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#-------------------Train Loop (Ft. Get Accuracy & Plotting)----------------------------------------



def get_accuracy(model,set_):
    label_ = [0]*(300)
    for i in range(0,300,2):
        label_[i] = 1

    label = torch.tensor(label_)

    trainSet_,valSet_,__ = get_data_loader(150)
    if set_ == "train":
        data_ = trainSet_
    #elif set_ == "val":
        #data_ = valSet_


    correct = 0
    total = 0
    for img, _ in data_:
        b = torch.split(img,600,dim=3)
        img = torch.cat(b, 0)


        output = model(img)
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(label.view_as(pred)).sum().item() #compute how many predictions were correct
        total += img.shape[0] #get the total ammount of predictions
        break

    return correct / total



def train(mdl,epochs= 20,batch_size = 32,learning_rate =0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mdl.parameters(), lr=learning_rate, momentum=0.9)
    trainSet = get_data_loader(batch_size)
    train_acc, val_acc = [], []
    n = 0 # the number of iterations

    label_ = [0]*(batch_size*2)
    for i in range(0,batch_size*2,2):
        label_[i] = 1

    label = torch.tensor(label_)

    print("--------------Starting--------------")



    for epoch in range(epochs):  # loop over the dataset multiple times



        t1 = t.time()

        itera = 0
        for img,_ in iter(trainSet):


            b = torch.split(img,600,dim=3)


            img = torch.cat(b, 0)

            print(img.size())


            itera += batch_size*2
            res = resnet(img)
            out = mdl(res)

            loss = criterion(out, label)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            print(itera)
        break
        # Calculate the statistics
        train_acc.append(get_accuracy(mdl,"train"))

        #val_acc.append(get_accuracy(mdl,"val"))  # compute validation accuracy
        n += 1


        print("Epoch",n,"Done in:",t.time() - t1, "With Training Accuracy:",train_acc[-1])#, "And Validation Accuracy:",val_acc[-1])


        # Save the current model (checkpoint) to a file
        model_path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(mdl.name,batch_size,learning_rate,epoch)
        torch.save(mdl.state_dict(), model_path)

    iterations = list(range(1,epochs + 1))

    print("--------------Finished--------------")

    return iterations,train_acc, val_acc



def plot(iterations,train_acc, val_acc):
    plt.title("Training Curve")
    plt.plot(iterations, train_acc, label="Train")
    #plt.plot(iterations, val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    #print("Final Validation Accuracy: {}".format(val_acc[-1]))

from a3code import AlexNetFeatures
myfeature_model = AlexNetFeatures() #loads pre-trained weights
atrain_loader = get_data_loader(1)
a=0
for img, l in atrain_loader:
    features=myfeature_model(img)
    i=str(a)
    label=l[0].item()
    print(features.shape)
    break

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Linear(256*6*6, 50)
        self.layer2 = nn.Linear(50, 20)
        self.layer3 = nn.Linear(20, 2)
    def forward(self, img):
        flattened = img.view(-1,256*6*6)
        activation1 = F.relu(self.layer1(flattened))
        activation2 = F.relu(self.layer2(activation1))
        output = self.layer3(activation2)
        return output

def get_alex_data_loader(batch_size, shuffle=True):
    np.random.seed(1000) # Fixed numpy random seed for reproducible shuffling

    train_sampler = torchvision.datasets.DatasetFolder(root='trainData', loader=torch.load, extensions=list(['']))
    alex_train_loader = torch.utils.data.DataLoader(train_sampler, batch_size=batch_size, shuffle=shuffle)
    #val_sampler =  torchvision.datasets.DatasetFolder(root='valData', loader=torch.load, extensions=list(['']))
    #alex_val_loader = torch.utils.data.DataLoader(val_sampler, batch_size=batch_size, shuffle=shuffle)

    #test_sampler =  torchvision.datasets.DatasetFolder(root='testData', loader=torch.load, extensions=list(['']))
        #alex_test_loader = torch.utils.data.DataLoader(test_sampler, batch_size=batch_size, shuffle=shuffle)

    return alex_train_loader#, alex_train_loader, alex_test_loader


def get_alex_accuracy(model, train=True):
    batch_size=16
    label_ = [0]*(batch_size*2)
    for i in range(0,batch_size*2,2):
        label_[i] = 1

    if train:
        data = alex_train_loader
    #else:
    #    train, data, test = alex_train_loader, alex_val_loader, alex_test_loader = get_alex_data_loader(1)

    correct = 0
    total = 0
    for inputs, labels in data:
        output = model(inputs) # We don't need to run F.softmax
        pred = output.argmax()
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += inputs.shape[0]
    return correct / total

def alextrain(model, batch_size=32, num_epochs=15, lr=0.0001):

    atrain_loader = get_alex_data_loader(1)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    iters, losses, train_acc, val_acc = [], [], [], []

    # training
    n = 0 # the number of iterations
    label_ = [0]*(batch_size*2)
    for i in range(0,batch_size*2,2):
        label_[i] = 1

    label = torch.tensor(label_).cuda()

    for epoch in range(num_epochs):
        correct=0
        total=0
        #to store the iteration that reached 100% accuracy first.
        for inputs, labels in iter(atrain_loader):
            out = model(inputs)             # forward pass
            loss = criterion(out, labels) # compute the total loss
            loss.backward()               # backward pass (compute parameter updates)
            optimizer.step()              # make the updates for each parameter
            optimizer.zero_grad()         # a clean up step for PyTorch
            # save the current training information
            pred = out.argmax()
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += inputs.shape[0]
        train_acc.append(correct/total) # compute training accuracy
        #val_acc.append(get_alex_accuracy(model, train=False))  # compute validation accuracy
        n += 1
        iters.append(n)


    plt.title("Training Curve")
    plt.plot(iters, train_acc, label="Train")
    #plt.plot(iters, val_acc, label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    #print("Final Validation Accuracy: {}".format(val_acc[-1]))

    return iterations, train_acc
>>>>>>> 2044b5a1553748ec92cbae85c5296a6a6ce38122
