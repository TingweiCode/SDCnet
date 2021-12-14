"""
Created on AUG 24 11:02:11 2021

@author: yang
"""
from SDCnet import *
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
import torch.nn as nn
from tqdm import *
import torch
import sys


from torch import nn
import torch.optim as optim
import torch.utils.data
from sklearn import preprocessing
from collections import Counter

from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import torch.autograd
from torch.autograd import Variable
from torchvision import transforms
from torchvision import datasets
from torchvision.utils import save_image

# set GPU 
#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

## load train data
amp = np.load('./train_data/amp_syn.npy').reshape((-1,1,512,512))
group = np.load('./train_data/group_syn.npy').reshape((-1,1,512,512))
amp = torch.tensor(amp).type(torch.FloatTensor)
group = torch.tensor(group).type(torch.FloatTensor)

## load test data
test = np.load('./test_data/amp_syn.npy').reshape((-1,1,512,512))
test_label = np.load('./test_data/group_syn.npy').reshape((-1,1,512,512))
test = torch.tensor(test).type(torch.FloatTensor)
test_label = torch.tensor(test_label).type(torch.FloatTensor)

## set data batch_size
train_loader = torch.utils.data.DataLoader(amp,batch_size=5)
label_loader = torch.utils.data.DataLoader(group,batch_size=5)
test_loader = torch.utils.data.DataLoader(test,batch_size=1)
test_label_loader = torch.utils.data.DataLoader(test_label,batch_size=1)

## set train parameter
device = torch.device('cuda:0')
net = torch.nn.DataParallel(Resunet_base()).to(device)
biloss = nn.BCELoss().to(device)
optimizer = optim.Adam(net.parameters(),lr=0.0002,betas=(0.5,0.999))
Train_Loss_list = []
Test_Loss_list = []
Accuracy_list = []
epochs = 500

for epoch in range(epochs):
    i = 0
    j = 0
    net.train()
    train_loss = 0
    train_acc = 0
    test_loss = 0

    for train,label in zip(train_loader,label_loader):
        i += 1
        train = train.cuda()
        label = label.cuda()
        score = net(train)
        loss = biloss(score,label)
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        train_loss += float(loss.item())
        sys.stdout.write("[EPoch %d/%d] [Batch:%d/%d] [loss: %f] \n" %(epoch,epochs,i,len(train_loader),loss.item()))

    with torch.no_grad():
        for  test_data, test_data_label in zip(test_loader, test_label_loader):
            j += 1
            res_test = net(test_data.cuda())
            loss_test = biloss(res_test, test_data_label.cuda())
            test_loss += float(loss_test.item())
            sys.stdout.write("[EPoch %d/%d] [Batch:%d/%d] [loss_test: %f]\n" % (epoch,epochs,j,len(test_loader),loss_test.item()))

    ## save photo
    if epoch % 5 == 0 and epoch != 0:
        save_image(score.cpu().data, 'photo/res_{}.png'.format(epoch))
        save_image(label.cpu().data, 'photo/label_{}.png'.format(epoch))
        save_image(train.cpu().data, 'photo/data_{}.png'.format(epoch))
        
        save_image(res_test.cpu().data, 'photo/res_test{}.png'.format(epoch))
        save_image(test_data.cpu().data, 'photo/test_data{}.png'.format(epoch))
        save_image(test_data_label.cpu().data, 'photo/test_data_label{}.png'.format(epoch))
    
    Test_Loss_list.append(test_loss / len(test_loader))
    Train_Loss_list.append(train_loss / len(train_loader))

## save loss
np.savetxt("Test_Loss_syn.csv", np.array(Test_Loss_list))
np.savetxt("Train_Loss_syn.csv", np.array(Train_Loss_list))
torch.save(net.state_dict(),"SDCnet_syn_model.pth")





