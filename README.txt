SDCnet Network Package
===========================

###########Environment
Linux or Windows
Python3.8.5
PyTorch1.7.1

###########Directory
Readme.txt                  // help
train_data                  // Train data file included  FTAN diagram and dispersion curves data for training network.
test_data                  // Test data file included  FTAN diagram and dispersion curves data for testing network.
test_data_ak135                  // Test data file included  Ak135 FTAN diagram and dispersion curves data for testing network.
plot_loss                  // Loss data file.
test_syn_photo_SDCnet                  // prediction photo of SDCnet for synthetic data.
test_obs_photo_SDCnet                  // prediction photo of SDCnet for observed data.
test_obs_photo_Unet                  // prediction photo of Unet for observed data.
1-train_syn.py            //Running the code for the main program.
2-prediction_ak135_SDCnet.py    // Synthetic test for AK135 dispersion curves.
2-prediction_syn_SDCnet.py  // Synthetic test for 3D synthetic data extract dispersion curves.
2-prediction_obs_SDCnet.py  // Implementation of extract dispersion curves for observed data.
2-prediction_obs_Unet.py    // Implementation SDCnet of extract dispersion curves for observed data.
3-plot_loss_obs_Unet_and_SDCnet.py  // Plot SDCnet and Unet loss.
SDCnet.py                // Network structure of SDCnet.
Unet.py                // Network structure of Unet.
SDCnet_model_syn.pth    // SDCnet model trained with synthetic data.
SDCnet_model.pth    // SDCnet model trained with observed data and synthetic data.
Unet_model.pth    // Unet model trained with observed data and synthetic data.

###########V1.0.0
Extracting dispersion curves from FTAN diagram