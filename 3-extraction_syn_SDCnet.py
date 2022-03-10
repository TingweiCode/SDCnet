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
from scipy import interpolate

#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

font1 = {'family' : 'serif',
'weight' : 'normal',
'size'   : 30,
}
font2 = {'family' : 'serif',
'weight' : 'normal',
'size'   : 20,
}

os.system('rm -r  SDCnet_Dispersion_syn')
os.system('mkdir -p SDCnet_Dispersion_syn')

## load data
datas = np.load('test_data/amp_syn.npy').reshape((-1,1,512,512))
labels = np.load('test_data/group_syn.npy')

## put data in to network
datas = torch.tensor(datas).type(torch.FloatTensor).cuda()
net = torch.nn.DataParallel(Resunet_base())
torch.backends.cudnn.benchmark = True
net.load_state_dict(torch.load('SDCnet_syn_model.pth'),False)
net.cuda()
net.eval()
outputs = torch.zeros(len(datas),1,512,512).cuda()

with torch.no_grad():
    for i in tqdm(range(len(datas))):
        outputs[i] = net(datas[i].view(1,1,512,512)).view(1,512,512)
        
        output = outputs[i].cpu().numpy().reshape((512,512))
        data = datas[i].cpu().numpy().reshape((512,512))
        label = labels[i].reshape((512,512))


        all_h_pre=[]
        all_dop_pre=[]
        all_dop_pre_new = []
        c = 0
        for l_pre in range(0,510):
            all_h_pre=[]
            for h_pre in range(output.shape[0]):
                if output[h_pre,l_pre]>0.8:    ##convince>=2.0
                    break
                all_h_pre.append(h_pre)
            if all_h_pre is None or len(all_h_pre)==0 : 
                c=c+1
            elif max(all_h_pre)>=10 : ##group>=2.0
                all_dop_pre.append([l_pre,max(all_h_pre)])
        all_dop_pre =  np.array(all_dop_pre)           
        for num in range(len(all_dop_pre)):
            if num < len(all_dop_pre)-1:
                x1 = np.float(all_dop_pre[num][0])
                x2 =  np.float(all_dop_pre[num+1][0])
                y1 =  np.float(all_dop_pre[num][1])
                y2 =  np.float(all_dop_pre[num+1][1])
                grad = (y2-y1) / (x2-x1)
            else:
                grad = 2
            all_dop_pre_new.append(grad)
        all_dop_pre_new=np.array(all_dop_pre_new)  

        index_grad = []
        inis = []
        ends = []
        
        for k in range(len(all_dop_pre_new)):
            if all_dop_pre_new[k]>2 or all_dop_pre_new[k]<-2:
                index_grad.append(k)

        for g in index_grad:
            if g < 50 :
               inis.append(g)
            else:
               ends.append(g)

        if  len(ends) == 0:
            ends = [-1]

        if  len(inis) == 0:
            inis = [0]

        select2= all_dop_pre[inis[-1]:ends[0]]
        select = all_dop[inis[-1]:ends[0]]

        # interp1d
        if select2.shape[0]>=5:               
            raw_t_min=np.round(select2[0,0]+5,2)
            raw_t_max=np.round(select2[-1,0]-5,2)
            npts = int(((raw_t_max-raw_t_min)/5) + 1)
            new_t2=np.linspace(raw_t_min,raw_t_max,npts)
            f=interpolate.interp1d(select2[1:-1,0],select2[1:-1,1],kind="cubic")
            new_g2=f(new_t2)
   

        f=open('./SDCnet_Dispersion_syn/Dispersion_%d_syn.txt' %i,'w')
        for l in range(new_t2[2:-2].shape[0]):
            f.write('%.2f %.2f\n' %(new_t2[l]*0.1+5.1,new_g2[l]*0.01+1.01))
        f.close()




