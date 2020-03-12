#!/bin/bash

## tensorboard --logdir /home/xin/OneDrive/Working_directory/Continuous_cGAN/CellCounting/Output/saved_logs

SEED=2020
EPOCH_GAN=2000
BATCH_SIZE=128
START_COUNT=1
END_COUNT=200
STEPSIZE_COUNT=2
N_IMGS_PER_CELLCOUNT=5
SIGMA=0.01
LR_G=1e-4
LR_D=1e-4
NFAKE=50000



# CUDA_VISIBLE_DEVICES=0 python3 main.py --show_real_imgs



echo "########################################################################################"
echo "                         Pre-train a CNN for evaluation                                 "
echo "########################################################################################"
# CUDA_VISIBLE_DEVICES=0 python3 pretrain_CNN.py --start_count $START_COUNT --end_count $END_COUNT --CNN ResNet34 --epochs 100 --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --transform --CVMode
# CUDA_VISIBLE_DEVICES=0 python3 pretrain_CNN.py --start_count $START_COUNT --end_count $END_COUNT --CNN ResNet34 --epochs 100 --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --transform

# CUDA_VISIBLE_DEVICES=0 python3 pretrain_CNN.py --start_count $START_COUNT --end_count $END_COUNT --CNN ResNet50 --epochs 100 --batch_size_train 128 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --transform --CVMode
# CUDA_VISIBLE_DEVICES=0 python3 pretrain_CNN.py --start_count $START_COUNT --end_count $END_COUNT --CNN ResNet50 --epochs 100 --batch_size_train 128 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --transform

# nvidia-settings -a "[gpu:0]/GPUFanControlState=0"
# nvidia-settings -a "[gpu:1]/GPUFanControlState=0"

# echo "########################################################################################"
# echo "                                  Baseline GANs                                         "
# echo "########################################################################################"

# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN"
# CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN DCGAN --start_count $START_COUNT --end_count $END_COUNT --stepsize_count $STEPSIZE_COUNT --num_imgs_per_count $N_IMGS_PER_CELLCOUNT --transform --epoch_gan $EPOCH_GAN --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED --nfake $NFAKE --samp_batch_size 100 --comp_FID
#
# echo "-------------------------------------------------------------------------------------------------"
# echo "cDCGAN"
# CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN cDCGAN --start_count $START_COUNT --end_count $END_COUNT --stepsize_count $STEPSIZE_COUNT --num_imgs_per_count $N_IMGS_PER_CELLCOUNT --transform --epoch_gan $EPOCH_GAN --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED --nfake $NFAKE --samp_batch_size 100 --comp_LS --comp_FID



echo "########################################################################################"
echo "                                 Continuous_cDCGAN                                      "
echo "########################################################################################"


### Hard

for kappa in 0.005 0.01 0.02 0.03 0.04
do
  kernel_sigma=$SIGMA
  # kappa=$(echo "$factor*$kernel_sigma" | bc)

  echo "-------------------------------------------------------------------------------------------------"
  echo "Continuous_cDCGAN, normalize, HARD, kernel_sigma=$kernel_sigma, kappa=$kappa SEED=$SEED"
  CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN Continuous_cDCGAN --start_count $START_COUNT --end_count $END_COUNT --stepsize_count $STEPSIZE_COUNT --num_imgs_per_count $N_IMGS_PER_CELLCOUNT --transform --kernel_sigma $kernel_sigma --threshold_type hard --kappa $kappa --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED --nfake $NFAKE --samp_batch_size 100 #--comp_LS --comp_FID
done

### Soft

for kappa in 0.8 1.0 2.0 5.0 10.0 20.0
do
  kernel_sigma=$SIGMA

  echo "-------------------------------------------------------------------------------------------------"
  echo "Continuous_cDCGAN, normalize, Soft, kernel_sigma=$kernel_sigma, kappa=$kappa SEED=$SEED"
  CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN Continuous_cDCGAN --start_count $START_COUNT --end_count $END_COUNT --stepsize_count $STEPSIZE_COUNT --num_imgs_per_count $N_IMGS_PER_CELLCOUNT --transform --kernel_sigma $kernel_sigma --threshold_type soft --kappa $kappa --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED --nfake $NFAKE --samp_batch_size 100 #--comp_LS --comp_FID
done


# for factor in 1.0 2.0 3.0 4.0
# do
#   kernel_sigma=$SIGMA
#   kappa=$(echo "$factor*$kernel_sigma" | bc)
#
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "Continuous_cDCGAN, normalize, Soft, kernel_sigma=$kernel_sigma, kappa=$kappa SEED=$SEED"
#   CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN Continuous_cDCGAN --start_count $START_COUNT --end_count $END_COUNT --stepsize_count $STEPSIZE_COUNT --num_imgs_per_count $N_IMGS_PER_CELLCOUNT --transform --kernel_sigma $kernel_sigma --threshold_type soft --kappa $kappa --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED --nfake $NFAKE --samp_batch_size 100 --comp_LS --comp_FID
# done


### The limit to cGAN: sigma-->0, kappa-->0 Hard
CUDA_VISIBLE_DEVICES=0 python3 main.py --start_count $START_COUNT --end_count $END_COUNT --stepsize_count $STEPSIZE_COUNT --num_imgs_per_count $N_IMGS_PER_CELLCOUNT --GAN Continuous_cDCGAN --transform --kernel_sigma 1e-30 --threshold_type hard --kappa 1e-30 --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED --nfake $NFAKE --samp_batch_size 100 --comp_LS --comp_FID

### The limit to GAN: sigma-->1, kappa-->1e+30 Hard
SIGMA=1.0
CUDA_VISIBLE_DEVICES=0 python3 main.py --start_count $START_COUNT --end_count $END_COUNT --stepsize_count $STEPSIZE_COUNT --num_imgs_per_count $N_IMGS_PER_CELLCOUNT --GAN Continuous_cDCGAN --transform --kernel_sigma $SIGMA --threshold_type hard --kappa 1e+30 --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED --nfake $NFAKE --samp_batch_size 100 --comp_LS --comp_FID



### The limit to cGAN: sigma-->0, kappa-->1e+30 Soft
CUDA_VISIBLE_DEVICES=0 python3 main.py --start_count $START_COUNT --end_count $END_COUNT --stepsize_count $STEPSIZE_COUNT --num_imgs_per_count $N_IMGS_PER_CELLCOUNT --GAN Continuous_cDCGAN --transform --kernel_sigma 1e-30 --threshold_type soft --kappa 1e+30 --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED --nfake $NFAKE --samp_batch_size 100 --comp_LS --comp_FID

# ### The limit to GAN: sigma-->1, kappa-->0 Soft
SIGMA=1.0
CUDA_VISIBLE_DEVICES=0 python3 main.py --start_count $START_COUNT --end_count $END_COUNT --stepsize_count $STEPSIZE_COUNT --num_imgs_per_count $N_IMGS_PER_CELLCOUNT --GAN Continuous_cDCGAN --transform --kernel_sigma $SIGMA --threshold_type soft --kappa 1e-30 --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED --nfake $NFAKE --samp_batch_size 100 --comp_LS --comp_FID









# ## reset fan speed
# nvidia-settings -a "[gpu:0]/GPUFanControlState=0"
# nvidia-settings -a "[gpu:1]/GPUFanControlState=0"
