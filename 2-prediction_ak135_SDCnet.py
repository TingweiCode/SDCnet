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


## load data
data= np.array(pd.read_csv("./test_data_ak135/Amp_AK13-AK13.csv", header=None)).reshape((1,1,512,512))#(10,1,512,512)->(10,512,512)
label = np.array(pd.read_csv('./test_data_ak135/Group_AK13-AK13.csv', header=None))

## min_max_scaler
min_max_scaler = preprocessing.MinMaxScaler()
for i in tqdm(range(data.shape[0])):
    mm_scaler = min_max_scaler.fit(data[ i, 0, :, :])
    data[ i, 0, :, :] = min_max_scaler.transform(data[ i, 0, :, :])

## put data in to network
data = torch.tensor(data).type(torch.FloatTensor)
net = torch.nn.DataParallel(Resunet_base())
torch.backends.cudnn.benchmark = True
net.load_state_dict(torch.load('SDCnet_model.pth'),False)
net.cuda()
net.eval()
with torch.no_grad():
    outputs = net(data.cuda())
outputs = outputs.data.cpu().numpy().reshape((512,512))

## plot prediction 
font1 = {'family' : 'serif',
'weight' : 'normal',
'size'   : 30,
}
font2 = {'family' : 'serif',
'weight' : 'normal',
'size'   : 20,
}

fig = plt.figure(1,figsize=(10,7))
plt.imshow(outputs,origin='lower',cmap = 'gray')
plt.title("Predict",font1)
xticks=[49,149,249,349,469]
xticklabes=['10','20','30','40','50']
yticks=[99,199,299,399,499]
yticklabes=['2','3','4','5','6']
plt.xticks(xticks, xticklabes,size=20)
plt.yticks(yticks, yticklabes,size=20)
plt.xlabel('Period(s)',font2)
plt.ylabel('Group Velocity(km/s)',font2)
plt.savefig('./test_ak135_photo/predicted.png',dpi = 600)

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
plt.savefig('./test_ak135_photo/label.png',dpi = 600)

fig = plt.figure(3,figsize=(10,7))
plt.imshow(data[0,0,:,:],origin='lower',vmin=0, vmax=1,cmap=plt.cm.jet)
plt.title("The dispersion image",font1)
plt.colorbar()
xticks=[49,149,249,349,469]
xticklabes=['10','20','30','40','50']
yticks=[99,199,299,399,499]
yticklabes=['2','3','4','5','6']
plt.xticks(xticks, xticklabes,size=20)
plt.yticks(yticks, yticklabes,size=20)
plt.xlabel('Period(s)',font2)
plt.ylabel('Group Velocity(km/s)',font2)
plt.savefig('./test_ak135_photo/data.png',dpi = 600)


