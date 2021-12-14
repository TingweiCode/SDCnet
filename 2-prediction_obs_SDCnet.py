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
from sklearn import preprocessing
import matplotlib as mpl

font1 = {'family' : 'serif',
'weight' : 'normal',
'size'   : 30,
}
font2 = {'family' : 'serif',
'weight' : 'normal',
'size'   : 20,
}

## load data
datas = np.load('test_data/amp.npy').reshape((-1,1,512,512))
labels = np.load('test_data/group.npy')

## put data in to network
datas = torch.tensor(datas).type(torch.FloatTensor).cuda()
net = torch.nn.DataParallel(Resunet_base())
torch.backends.cudnn.benchmark = True
net.load_state_dict(torch.load('SDCnet_model.pth'),False)
net.cuda()
net.eval()
outputs = torch.zeros(len(datas),1,512,512).cuda()

with torch.no_grad():
    for i in tqdm(range(len(datas))):
        outputs[i] = net(datas[i].view(1,1,512,512)).view(1,512,512)
        
        output = outputs[i].cpu().numpy().reshape((512,512))
        data = datas[i].cpu().numpy().reshape((512,512))
        label = labels[i].reshape((512,512))
  

        ## plot prediction
        fig = plt.figure(1,figsize=(10,7))
        plt.imshow(output,origin='lower',cmap = 'gray')
        plt.title("Predict",font1)
        xticks=[49,149,249,349,469]
        xticklabes=['10','20','30','40','50']
        yticks=[99,199,299,399,499]
        yticklabes=['2','3','4','5','6']
        plt.xticks(xticks, xticklabes,size=20)
        plt.yticks(yticks, yticklabes,size=20)
        plt.xlabel('Period(s)',font2)
        plt.ylabel('Group Velocity(km/s)',font2)
        plt.savefig('./test_photo/predicted_%d.png' %i,dpi = 600)

        fig = plt.figure(2,figsize=(10,7))
        plt.imshow(label,origin='lower',cmap = 'gray')
        plt.title("Label",font1)
        xticks=[49,149,249,349,469]
        xticklabes=['10','20','30','40','50']
        yticks=[99,199,299,399,499]
        yticklabes=['2','3','4','5','6']
        plt.xticks(xticks, xticklabes,size=20)
        plt.yticks(yticks, yticklabes,size=20)
        plt.xlabel('Period(s)',font2)
        plt.ylabel('Group Velocity(km/s)',font2)
        plt.savefig('./test_photo/label_%d.png' %i,dpi = 600)

        fig = plt.figure(3,figsize=(10,7))
        plt.imshow(data,origin='lower',vmin=0, vmax=1,cmap=plt.cm.jet)
        plt.title("The dispersion image",font1)
        xticks=[49,149,249,349,469]
        xticklabes=['10','20','30','40','50']
        yticks=[99,199,299,399,499]
        yticklabes=['2','3','4','5','6']
        plt.xticks(xticks, xticklabes,size=20)
        plt.yticks(yticks, yticklabes,size=20)
        plt.xlabel('Period(s)',font2)
        plt.ylabel('Group Velocity(km/s)',font2)
        plt.savefig('./test_photo/data_%d.png' %i,dpi = 600)


