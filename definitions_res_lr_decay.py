import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
      #  x = x.view(x.size(0), -1)
      #  x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model



def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model



def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model



def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model



def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model

class ResNet4(nn.Module):
    def __init__(self,dropout):
        super(ResNet4, self).__init__()
        self.name = "ResNet"
        self.fc1 = nn.Linear( 86528,300)
        self.fc2 = nn.Linear( 300,100)
        self.fc3 = nn.Linear( 100,32)
        self.fc4 = nn.Linear(32, 2)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)



    def forward(self, x):
        #print(x.size())
        #x = x.view(-1, 86528)
        #print(x.size())
        x = F.relu(self.dropout1(self.fc1(x)))
        #print(x.size())
        x = F.relu(self.dropout2(self.fc2(x)))
        x = F.relu(self.dropout3(self.fc3(x)))
        x = self.fc4(x)
        #x = x.squeeze(1)
        #print(x.size(),"\n\n\n")
        return x

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

class ResNet6(nn.Module):
    def __init__(self):
        super(ResNet6, self).__init__()
        self.name = "ResNet"
        self.fc1 = nn.Linear(86528,500)
        self.fc2 = nn.Linear(500,200)
        self.fc3 = nn.Linear(200,120)
        self.fc4 = nn.Linear(120, 90)
        self.fc5 = nn.Linear(90, 32)
        self.fc6 = nn.Linear(32, 2)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.dropout5 = nn.Dropout(dropout)



    def forward(self, x):
        #print(x.size())
        #x = x.view(-1, 86528)
        #print(x.size())
        x = F.relu(self.dropout1(self.fc1(x)))
        #print(x.size())
        x = F.relu(self.dropout2(self.fc2(x)))
        x = F.relu(self.dropout3(self.fc3(x)))
        x = F.relu(self.dropout4(self.fc4(x)))
        x = F.relu(self.dropout5(self.fc5(x)))
        x = self.fc6(x)
        #x = x.squeeze(1)
        #print(x.size(),"\n\n\n")
        return x


# -*- coding: utf-8 -*-
"""
Created on Thur Mar 21 08:20:07 2019

@author: marc
"""
#----------------------Imports------------------------------
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
import scipy.ndimage as ndimage

                
#from resnet import *
def HPFilterNP(imgs):
    filteredimgs=[]
    for img in imgs:
        img = np.transpose(img, [1,2,0])
        weights = torch.tensor([[-1.,2.,-2.,2.,-1.],
                           [2.,-6.,8.,-6.,2.],
                           [-2.,8.,-12.,8.,-2.],
                           [2.,-6.,8.,-6.,2.],
                           [-1.,2.,-2.,2.,1.]])
        result=ndimage.convolve(img, np.atleast_3d(weights))
        result=np.transpose(result,[2,0,1])
        filteredimgs.append(torch.from_numpy(result))
    filteredimgs = torch.stack(filteredimgs)
    return filteredimgs
        
    
def LPFilterNP(imgs):
    filteredimgs=[]
    for img in imgs:
        img = np.transpose(img, [1,2,0])
        weights2_m1 = np.array([[0.,0.,0.,0.,0.],
                        [0.,-1.,2.,-1.,0.], 
                        [0.,2.,-4.,2.,0],
                        [0.,-1.,2.,-1.,0.], 
                        [0.,0.,0.,0.,0.]])
        weights2_m1=1/4*weights2_m1

        weights2_m2 = np.array([[-1.,2.,-2.,2.,-1],
                            [2.,-6.,8.,-6.,2.], 
                            [-2.,8.,-12.,8.,-2.],
                            [2.,-6.,8.,-6.,2.], 
                            [-1.,2.,-2.,2.,-1],])
        weights2_m2=1/12*weights2_m2

        weights2_m3 = np.array([[0.,0.,0.,0.,0.],
                            [0.,0.,0.,0.,0.],
                            [0.,1.,-2.,1.,0],
                            [0.,0.,0.,0.,0.],
                            [0.,0.,0.,0.,0.]])
        weights2_m3=1/2*weights2_m3
        weights=np.dot(weights2_m1, weights2_m2, weights2_m3)
        weights=torch.from_numpy(weights)
        result=ndimage.convolve(img, np.atleast_3d(weights))
        result=np.transpose(result,[2,0,1])
        filteredimgs.append(torch.from_numpy(result))
    filteredimgs = torch.stack(filteredimgs)
    return filteredimgs
        
def LPFilter(img):
    weights2_m1 = np.array([[0.,0.,0.,0.,0.],
                        [0.,-1.,2.,-1.,0.], 
                        [0.,2.,-4.,2.,0],
                        [0.,-1.,2.,-1.,0.], 
                        [0.,0.,0.,0.,0.]])
    weights2_m1=1/4*weights2_m1

    weights2_m2 = np.array([[-1.,2.,-2.,2.,-1],
                        [2.,-6.,8.,-6.,2.], 
                        [-2.,8.,-12.,8.,-2.],
                        [2.,-6.,8.,-6.,2.], 
                        [-1.,2.,-2.,2.,-1],])
    weights2_m2=1/12*weights2_m2

    weights2_m3 = np.array([[0.,0.,0.,0.,0.],
                        [0.,0.,0.,0.,0.],
                        [0.,1.,-2.,1.,0],
                        [0.,0.,0.,0.,0.],
                        [0.,0.,0.,0.,0.]])
    weights2_m3=1/2*weights2_m3
    weights=np.dot(weights2_m1, weights2_m2, weights2_m3)
    weights=torch.from_numpy(weights)
    weights3=[weights,weights,weights]
    weights3=torch.stack(weights3).float()
    weights3=weights3.unsqueeze(dim=0)
    filteredimgs = F.conv2d(img, weights3, padding=2)
    return filteredimgs

def HPFilter(img):
    weights = torch.tensor([[[-1.,2.,-2.,2.,-1.],
                       [2.,-6.,8.,-6.,2.],
                       [-2.,8.,-12.,8.,-2.],
                       [2.,-6.,8.,-6.,2.],
                       [-1.,2.,-2.,2.,1.]], 
                       [[-1.,2.,-2.,2.,-1.],
                       [2.,-6.,8.,-6.,2.],
                       [-2.,8.,-12.,8.,-2.],
                       [2.,-6.,8.,-6.,2.],
                       [-1.,2.,-2.,2.,1.]],
                       [[-1.,2.,-2.,2.,-1.],
                       [2.,-6.,8.,-6.,2.],
                       [-2.,8.,-12.,8.,-2.],
                       [2.,-6.,8.,-6.,2.],
                       [-1.,2.,-2.,2.,1.]]])
    weights=weights.unsqueeze(dim=0)
    filteredimgs = F.conv2d(img, weights, padding=2)
    return filteredimgs


import torchvision.models as models
def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

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



class ResNet4(nn.Module):
    def __init__(self,):
        super(ResNet4, self).__init__()
        self.name = "ResNet4"
        self.fc1 = nn.Linear( 86528,300)
        self.fc2 = nn.Linear( 300,100)
        self.fc3 = nn.Linear( 100,32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class ResNet3(nn.Module):
    def __init__(self,):
        super(ResNet3, self).__init__()
        self.name = "ResNet3"
        self.fc1 = nn.Linear( 86528,300)
        self.fc2 = nn.Linear( 300,32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc4(x)
        return x

class ResNet2(nn.Module):
    def __init__(self,):
        super(ResNet2, self).__init__()
        self.name = "ResNet2"
        self.fc1 = nn.Linear( 86528,100)
        self.fc4 = nn.Linear( 100,2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc4(x)
        return x

class ResNet5(nn.Module):
    def __init__(self):
        super(ResNet5, self).__init__()
        self.name = "ResNet5"
        self.fc1 = nn.Linear(86528,500)
        self.fc2 = nn.Linear(500,200)
        self.fc3 = nn.Linear(200,100)
        self.fc4 = nn.Linear(100, 32)
        self.fc6 = nn.Linear(32, 2)

    def forward(self, x):
        #print(x.size())
        #x = x.view(-1, 86528)
        #print(x.size())
        x = F.relu(self.fc1(x))
        #print(x.size())
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc6(x)
        #x = x.squeeze(1)
        #print(x.size(),"\n\n\n")
        return x

class ResNet6(nn.Module):
    def __init__(self):
        super(ResNet6, self).__init__()
        self.name = "ResNet6"
        self.fc1 = nn.Linear(86528,500)
        self.fc2 = nn.Linear(500,200)
        self.fc3 = nn.Linear(200,120)
        self.fc4 = nn.Linear(120, 90)
        self.fc5 = nn.Linear(90, 32)
        self.fc6 = nn.Linear(32, 2)

    def forward(self, x):
        #print(x.size())
        #x = x.view(-1, 86528)
        #print(x.size())
        x = F.relu(self.fc1(x))
        #print(x.size())
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        #x = x.squeeze(1)
        #print(x.size(),"\n\n\n")
        return x

#-------------------Train Loop (Ft. Get Accuracy & Plotting)----------------------------------------



def get_accuracy(model,set_,batch_size):
    label_ = [0]*(batch_size*2)
    for i in range(1,batch_size*2,2):
        label_[i] = 1
        
    label = torch.tensor(label_)
    
    trainSet_,valSet_ = get_RN_data_loader(batch_size)
    if set_ == "train":
        data_ = trainSet_
    elif set_ == "val":
        data_ = valSet_
    
    
    correct = 0
    total = 0
    for res, batch in data_:
        if len(batch)==batch_size:
         #   b = torch.split(img,600,dim=3) 
          #  img = torch.cat(b, 0)

            res = res.view(-1, 86528)
            output = model(res) 
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(label.view_as(pred)).sum().item() #compute how many predictions were correct
            total += res.shape[0] #get the total ammount of predictions

    return correct / total
#
from sklearn.utils import shuffle

def train(mdl,epochs= 20,batch_size = 32,learning_rate =0.002, weight_decayval=0.001, schedulertype=None, factorval=0.05):
    #criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mdl.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decayval)
    #trainSet,valSet = get_data_loader(batch_size)
    if schedulertype==0:
        scheduler=ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=factorval)
    elif schedulertype==1:
        scheduler=LinearLR(optimizer, gamma=0.95)
    trainSet,valSet = get_RN_data_loader(batch_size)
    train_acc, val_acc = [], []
    n = 0 # the number of iterations

#    label_ = [0]*(batch_size)
#    label_.extend([1]*(batch_size))
#    
    label_ = [0]*(batch_size*2)
    for i in range(1,batch_size*2,2):
        label_[i] = 1


    label = torch.tensor(label_)#.type(torch.FloatTensor)
    #mdl = mdl.cuda()

    for epoch in range(epochs):  # loop over the dataset multiple times

        t1 = t.time()

        for res,batch in iter(trainSet):
            
            if len(batch)==batch_size:

                res = res.view(-1, 86528)
                
                res = res.cpu().detach().numpy()
                label = label.cpu().detach().numpy()
                #print(res,label)
         #       res, label = shuffle(res, label, random_state=0)
                #print(res,label)
                res = torch.tensor(res)
                label = torch.tensor(label)
                
                #print(res.size(),batch.size())
                #x = torch.squeeze(res,1)
                #print(res)
               
                #res = torch.cat(b,0)
                #print(x.shape)
                #res = resnet18(img)
                
                out = mdl(res)
    
                #print(out.size())
                loss = criterion(out, label)
                loss.backward()
    
                optimizer.step()
                
                optimizer.zero_grad()
            
                #print("Iteration Done")
            
        # Calculate the statistics
        train_acc.append(get_accuracy(mdl,"train",batch_size = 150))
        val_accuracy=get_accuracy(mdl,"val",batch_size = 150)
        scheduler.step(val_accuracy)
    
        val_acc.append(val_accuracy)  # compute validation accuracy
        n += 1
    
    
        #print("Epoch",n,"Done in:",t.time() - t1, "With Training Accuracy:",train_acc[-1], "And Validation Accuracy:",val_acc[-1])
    
    
            # Save the current model (checkpoint) to a file
           # model_path = "model_{0}_bs{1}_lr{2}_epoch{3}".format(mdl.name,batch_size,learning_rate,epoch)
           # torch.save(mdl.state_dict(), model_path)
    
    iterations = list(range(1,epochs + 1))
    
    
    return iterations,train_acc, val_acc



def plot(iterations, train_acc, val_acc,learning_rate, weight_decay, factor, mdl, batch_size):
    plt.title("Training Curve")
    print("lr=", learning_rate, " bs=", batch_size, " wd=", weight_decay," factor=", factor, " model=", mdl.name)
    plt.plot(iterations, train_acc, label="Train")
    plt.plot(iterations, val_acc, label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.legend(loc='best')
    plt.show()

    print("Final Training Accuracy: {}".format(train_acc[-1]))
    print("Final Validation Accuracy: {}".format(val_acc[-1]))



def RNFeatures(dataSet,type_):
    data_ = dataSet
    i = 0
    j= 0

    for img, label in data_:
        b = torch.split(img,600,dim=3)

        img = torch.cat(b, 0)
        filteredimgs=LPFilterNP(img)
        output = resnet18(filteredimgs)
        if type_ == 'train':
            tensor_path = "tensor_set{0}_number{1}".format(type_,i)

            torch.save(output,r'C:/Users/itaza/APS360/Project/LPtrainData/'+tensor_path)
            i+=1
        elif type_ == 'val':
            tensor_path = "tensor_set{0}_number{1}".format(type_,j)

            torch.save(output,r'C:/Users/itaza/APS360/Project/LPvalData/'+tensor_path)
            j+=1
#LPtrainSet,LPvalSet = get_data_loader(1)
#print("Train")
#RNFeatures(LPtrainSet,'train')
#print("Validation")
#RNFeatures(LPvalSet,'val')
def get_RN_data_loader(batch_size):

    train_path = r'RtrainData'
    val_path = r'RvalData'

    trainSet = torchvision.datasets.DatasetFolder(root=train_path,loader = torch.load,extensions = list(['']))
    train_data_loader = torch.utils.data.DataLoader(trainSet, batch_size=batch_size, shuffle=True)

    valSet = torchvision.datasets.DatasetFolder(root=val_path,loader = torch.load,extensions = list(['']))
    val_data_loader = torch.utils.data.DataLoader(valSet, batch_size=batch_size, shuffle=True)

    #testSet = torchvision.datasets.DatasetFolder(root=test_path,loader = torch.load,extensions = list(['']))
    #test_data_loader  = torch.utils.data.DataLoader(testSet, batch_size=batch_size, shuffle=True)
   
    return train_data_loader ,val_data_loader

#RtrainSet,RvalSet = get_RN_data_loader(16)
    
import matplotlib.pyplot as plt
import numpy as np
import pickle


#modelres1 = ResNet1()
mdl_list=[4,3,2,5,6]
bs_list=[32,64,128]
weight_list=[0.01,0.001,0.0001]
factor_list=[0.1,0.05,0.005,0.001]
lr_list=[0.001,0.01,0.0001,0.00005]

for bs in bs_list:
    for w in weight_list:
        for f in factor_list:
            for l in lr_list:
                for m in mdl_list:
                    if m==2: 
                        model=ResNet2()
                    elif m==3: 
                        model=ResNet3()
                    elif m==4: 
                        model=ResNet4()
                    elif m==5: 
                        model=ResNet5()
                    elif m==6: 
                        model=ResNet6()
                    iterations,train_acc,val_acc = train(mdl = model,epochs = 20,batch_size = bs, weight_decayval=w, factorval=f, learning_rate=l, schedulertype=0) #,val_acc
                    plot(iterations,train_acc, val_acc,learning_rate=l, weight_decay=w, factor=f, mdl=model, batch_size=bs)
