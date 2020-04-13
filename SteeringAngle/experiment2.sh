#!/bin/bash

## tensorboard --logdir /home/xin/OneDrive/Working_directory/Continuous_cGAN/SteeringAngle/Output/saved_logs

SEED=2020
MAX_N_IMG_PER_LABEL=100
EPOCH_GAN=500
EPOCH_CNN=200
BATCH_SIZE=512
SIGMA=-1.0
LR_G=1e-4
LR_D=1e-4
NFAKE=200000
IMG_SIZE=64






echo "########################################################################################"
echo "                         Pre-train a CNN for evaluation                                 "
echo "########################################################################################"
# CUDA_VISIBLE_DEVICES=1,0 python3 pretrain_CNN_regre.py --CNN ResNet34_regre --epochs $EPOCH_CNN --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --img_size $IMG_SIZE
CUDA_VISIBLE_DEVICES=1,0 python3 pretrain_CNN_regre.py --CNN ResNet34_regre --epochs $EPOCH_CNN --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --img_size $IMG_SIZE --CVMode
#
# CUDA_VISIBLE_DEVICES=1,0 python3 pretrain_CNN_class.py --CNN ResNet34_class --epochs $EPOCH_CNN --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --num_classes 100 --img_size $IMG_SIZE
CUDA_VISIBLE_DEVICES=1,0 python3 pretrain_CNN_class.py --CNN ResNet34_class --epochs $EPOCH_CNN --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --num_classes 20 --img_size $IMG_SIZE --CVMode





# echo "########################################################################################"
# echo "                                  Baseline GANs                                         "
# echo "########################################################################################"

# echo "-------------------------------------------------------------------------------------------------"
# echo "cDCGAN"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN cDCGAN --seed $SEED --img_size $IMG_SIZE --max_num_img_per_label $MAX_N_IMG_PER_LABEL --epoch_gan $EPOCH_GAN --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_gan $BATCH_SIZE --nfake $NFAKE --visualize_fake_images #--comp_LS --comp_FID --epoch_FID_CNN $EPOCH_CNN



echo "########################################################################################"
echo "                                 ContSAGAN                                      "
echo "########################################################################################"

echo "-------------------------------------------------------------------------------------------------"
echo "Continuous cDCGAN"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN ContcDCGAN --seed $SEED --img_size $IMG_SIZE --max_num_img_per_label $MAX_N_IMG_PER_LABEL --threshold_type hard --kernel_sigma -1.0 --kappa -1.0 --epoch_gan $EPOCH_GAN --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_gan $BATCH_SIZE --nfake $NFAKE --visualize_fake_images #--comp_LS --comp_FID --epoch_FID_CNN $EPOCH_CNN

# echo "-------------------------------------------------------------------------------------------------"
# echo "Continuous cDCGAN"
# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN ContcDCGAN --seed $SEED --img_size $IMG_SIZE --max_num_img_per_label $MAX_N_IMG_PER_LABEL --threshold_type soft --kernel_sigma -1.0 --kappa -1.0 --epoch_gan $EPOCH_GAN --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_gan $BATCH_SIZE --nfake $NFAKE --visualize_fake_images #--comp_LS --comp_FID --epoch_FID_CNN $EPOCH_CNN


## reset fan speed
# nvidia-settings -a "[gpu:0]/GPUFanControlState=0"
# nvidia-settings -a "[gpu:1]/GPUFanControlState=0"
