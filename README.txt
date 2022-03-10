SDCnet Network Package

===========================

###########Environment
Linux or Windows
Python3.8.5
PyTorch1.7.1

This Package can auto-extract dispersion from FTAN diagram.

All codes outline
1-train_syn.py  //  running the code for the main program
2-prediction_ak135_SDCnet.py    //  synthetic test for AK135 dispersion curves
2-prediction_syn_SDCnet.py  //    synthetic test for 3D synthetic data extract dispersion curves
2-prediction_obs_SDCnet.py  //  Implementation of extract dispersion curves for observed data in photo
2-prediction_obs_Unet.py    //  Implementation SDCnet of extract dispersion curves for observed data in photo
3-extraction_syn_SDCnet.py  //    synthetic test for 3D synthetic data extract dispersion curves in ASCII
3-extraction_obs_SDCnet.py  //  Implementation of extract dispersion curves for observed data in ASCII
4-plot_loss_obs_Unet_and_SDCnet.py  //  Plot SDCnet and Unet loss

All codes detailed imformation

1-train_syn.py 
############# 
Description of data:
We set the input and output to 512*512 two-dimensional data, to get the mapping relation from the spectrum energy diagram 
to the dispersion curve, making the network more generalizable.
(You can also adjust the training scale according to the needs of different research areas.)
The input data is 512*512 two-dimensional data of spectrum energy diagram (Different positions have different energy levels).
The label data is 512*512 two-dimensional data of segmentation result (Manually pick up dispersion curve, and set it to 0 below and 1 above).

2-prediction_xx_SDCnet.py ('xx' in the filename refers to different data; The demo is 2-prediction_obs_SDCnet.py represents observation data)
#############
Firstly,we need to use time-frequency analysis method to get the spectrum energy diagram.
Secondly,we need to make the spectrum diagram into 512*512 two-dimensional data.
And then, we can obtain the predicted segmentation results to test the effectiveness of the network by 2-prediction_xx_SDCnet.py
Finally, we can obtain the dispersion curves in ASCII by 3-extraction_xx_SDCnet.py

3-extraction_xx_SDCnet.py ('xx' in the filename refers to different data; The demo is 2-extraction_syn_SDCnet.py represents observation data)
#############
extract dispersion curves from the spectrum energy diagram in ASCII (Dispersion curves data will save in /SDCnet_Dispersion/)
## Period(s) Group(km/s)
10.00 2.88
10.50 2.89
11.01 2.90
11.51 2.92
12.01 2.92
12.51 2.93
13.02 2.94
13.52 2.93

All code can be run on computer server with four Nvidia GEFORCE RTX 2080Ti in Ubuntu 18.04.4 environment.
Contact information: Tingwei Yang, Email:s19010004@s.upc.edu.cn, Address: No. 19, Beitucheng Western Road, Chaoyang District, 100029, Beijing, P.R.China, Telephone: 82998104. If you have some problem about code, please contact me for priority email, many thanks.

===========================

All model
SDCnet.py                // Network structure of SDCnet (The network parameters are introduced in this). 
Unet.py                // Network structure of Unet (The network parameters are introduced in this). 
SDCnet_model_syn.pth    // SDCnet model trained with synthetic data.
SDCnet_model.pth    // SDCnet model trained with observed data and synthetic data.
Unet_model.pth    // Unet model trained with observed data and synthetic data.

===========================

###########Running
python 1-train_syn.py  // Loading data and training
python 2-prediction_syn_SDCnet.py   // Loading data, predicting and drawing segmented image.
python 2-prediction_obs_SDCnet.py   // Loading data, predicting and drawing segmented image.
python 3-extraction_syn_SDCnet.py   // Loading data and extracting  dispersion curves from synthetic data.
python 3-extraction_obs_SDCnet.py   // Loading data and extracting  dispersion curves from observed  data.
python plot_loss/4-plot_loss_obs_Unet_and_SDCnet.py   // Poltting the loss of SDCnet and Unet.
