"""
Created on March 9 11:02:11 2022

@author: yang
"""
## Import module
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

# set GPU channels
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

## load training data
amp = np.load('./train_data/amp_syn.npy').reshape((-1,1,512,512)) ## Dimension of input training data
group = np.load('./train_data/group_syn.npy').reshape((-1,1,512,512)) ## Dimension of label training data
amp = torch.tensor(amp).type(torch.FloatTensor) ## Input data transform to tensor
group = torch.tensor(group).type(torch.FloatTensor)  ## Labels data transform to tensor

## load test data
test = np.load('./test_data/amp_syn.npy').reshape((-1,1,512,512)) ## Dimension of input test data
test_label = np.load('./test_data/group_syn.npy').reshape((-1,1,512,512)) ## Dimension of label test data
test = torch.tensor(test).type(torch.FloatTensor)  ## Input of test data transform to tensor
test_label = torch.tensor(test_label).type(torch.FloatTensor) ## Labels of test data transform to tensor 

## set train data and test data batch_size
train_loader = torch.utils.data.DataLoader(amp,batch_size=5)
label_loader = torch.utils.data.DataLoader(group,batch_size=5)
test_loader = torch.utils.data.DataLoader(test,batch_size=1)
test_label_loader = torch.utils.data.DataLoader(test_label,batch_size=1)

## set train parameter
device = torch.device('cuda:0') 
net = torch.nn.DataParallel(Resunet_base()).to(device)
biloss = nn.BCELoss().to(device) ## set loss 
optimizer = optim.Adam(net.parameters(),lr=0.0002,betas=(0.5,0.999)) ## first-order and second-order Moment Estimations
Train_Loss_list = []
Test_Loss_list = []
Accuracy_list = []
epochs = 500   ## Training epochs

os.system('mkdir -p photo')

for epoch in range(epochs):
    i = 0
    j = 0
    net.train()
    train_loss = 0
    train_acc = 0
    test_loss = 0

    for train,label in zip(train_loader,label_loader):
        i += 1
        train = train.cuda() ## put data into GPU
        label = label.cuda() ## put data into GPU
        score = net(train)
        loss = biloss(score,label)
        optimizer.zero_grad()
        loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        train_loss += float(loss.item())
        sys.stdout.write("[EPoch %d/%d] [Batch:%d/%d] [loss: %f] \n" %(epoch,epochs,i,len(train_loader),loss.item())) ## Print the EPoch, Batch and loss train to screen

    with torch.no_grad():
        for  test_data, test_data_label in zip(test_loader, test_label_loader):
            j += 1
            res_test = net(test_data.cuda())
            loss_test = biloss(res_test, test_data_label.cuda())
            test_loss += float(loss_test.item())
            sys.stdout.write("[EPoch %d/%d] [Batch:%d/%d] [loss_test: %f]\n" % (epoch,epochs,j,len(test_loader),loss_test.item())) ## Print the EPoch, Batch and loss test to screen

    ## save photo
    if epoch % 5 == 0 and epoch != 0:
        save_image(score.cpu().data, 'photo/res_{}.png'.format(epoch)) ## save prediction of train data
        save_image(label.cpu().data, 'photo/label_{}.png'.format(epoch)) ## save labels of train data  
        save_image(train.cpu().data, 'photo/data_{}.png'.format(epoch)) ## save inputs of train data
        
        save_image(res_test.cpu().data, 'photo/res_test{}.png'.format(epoch)) ## save prediction of test data
        save_image(test_data.cpu().data, 'photo/test_data{}.png'.format(epoch)) ## save labels of test data  
        save_image(test_data_label.cpu().data, 'photo/test_data_label{}.png'.format(epoch)) ## save inputs of test data
     
    Test_Loss_list.append(test_loss / len(test_loader)) ## calc the test loss value 
    Train_Loss_list.append(train_loss / len(train_loader)) ## calc the train loss value 

## save loss
os.system('mkdir -p new_loss')
os.system('mkdir -p new_model')
np.savetxt("./new_loss/Test_Loss_syn.csv", np.array(Test_Loss_list)) ## save test loss
np.savetxt("./new_loss/Train_Loss_syn.csv", np.array(Train_Loss_list))  ## save train loss

## save SDCnet model
torch.save(net.state_dict(),"./new_model/SDCnet_syn_model.pth")





