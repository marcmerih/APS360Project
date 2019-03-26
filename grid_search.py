# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 20:28:36 2019

@author: chris
"""
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self,dropout):
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