#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 14:42:54 2019

@author: docear
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import BatchLoader as BL
import DatasetHaemo as DH
import torchvision



trainPath = Path('../trainDataKaggle/')
trainpath = '../trainDataKaggle/'

# Defining the neural network structure
class Net(nn.Module):
    def __init__(self): 
        # Defining layer types
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,3,3)
        self.pool = nn.MaxPool2d(2,2) # Defines a type of maxpooling to be used
        self.conv2 = nn.Conv2d(3, 8, 4)
        self.fc1 = nn.Linear(8*26*26, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 5)
        
    def forward(self,x):
        # Defines the forward pass using the layer types / operations defined in __init__()
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1,8*26*26)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return F.log_softmax(x,dim=1)
        
net = Net()
 

#df = pd.read_csv('/media/docear/My Passport/Kaggle/Hemorrhage/train_labels_as_strings.csv')
#df.labels.fillna('/media/docear/My Passport/Kaggle/Hemorrhage/train_labels_as_strings.csv',inplace=True)


'''

# Extracting Training Data (below)
dfLabels = pd.read_csv('/media/docear/My Passport/Kaggle/Hemorrhage/train_pivot.csv') # csv relating to the training data
index = list(trainPath.iterdir()) # lists paths to all train images
posn = len(trainpath) # the length of the prefix to the image name as found in the .csv file

filenames = [] # initialising array to hold all image names (without prefixes)
for idx in range(len(index)): # filling array with image names
    filenames.append(str(index[idx])[posn:])
    
dfLabels.set_index("fn",inplace=True) # sets the relevant data entry for look-up to be the filename (fn)


EPOCHS = 3
possibleBatches = round(len(index)/batchsize)

'''

trainDataset = DH.DatasetHeamo('/media/docear/My Passport/Kaggle/Hemorrhage/train_pivot.csv',filenames,trainpath,[1,112,112])
trainLoader = DataLoader(trainDataset, batch_size=32, shuffle=True)
trainIter = iter(trainLoader)
criterion = torch.nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

print("Starting Training")

for epoch in range(EPOCHS):
    running_loss = 0.0  # tracks the loss over a number of batches
    for i, data in enumerate(trainLoader,0):
        # get next batch
        inputs, labels = data
        #labels = labels.type(torch.LongTensor)
        
        # set cumulative gradients to 0
        optimizer.zero_grad()
        
        # forward pass ; backward pass; optimization
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # statistics
        running_loss += loss.item()
        if i % 500 == 499:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

print('Finished Training')
        

    
