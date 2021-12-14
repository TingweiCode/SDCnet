"""
Created on AUG 24 11:02:11 2021

@author: yang
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from tqdm import *

## load data 
rn_Train_loss = np.array(pd.read_csv('Train_Loss.csv', header=None))
rn_Test_loss = np.array(pd.read_csv('Test_Loss.csv', header=None))
u_Train_loss = np.array(pd.read_csv('UTrain_Loss.csv', header=None))
u_Test_loss = np.array(pd.read_csv('UTest_Loss.csv', header=None))

## plot
fig = plt.figure(1,figsize=(9,7))
plt.plot(rn_Test_loss,'darkred',label='SDCnet Test')
plt.plot(u_Test_loss,'k',label='Unet Test')
plt.plot(rn_Train_loss,'k',lw=2.5,label='SDCnet Train')
plt.plot(u_Train_loss,'darkblue',lw=2.5,label='Unet Train')

## set figure parameter
font1 = {'family' : 'serif',
'weight' : 'normal',
'size'   : 30,
}
font2 = {'family' : 'serif',
'weight' : 'normal',
'size'   : 20,
}
plt.legend(fontsize=16)
plt.title("Loss",font1)
plt.tick_params(labelsize=16)
plt.xlabel('Epoch',font2)
plt.ylabel('Binary Cross Entropy',font2)
plt.xlim(0,500)
plt.savefig('loss.png',dpi = 1200)

