#!/bin/bash



SEED=2020
NSIM=1
EPOCH=500
N_GANSSIANS=120
N_SAMP_PER_GAUSSIAN=10
STD_GAUSSIAN=0.02
RADIUS=1
BATCH_SIZE=128
LR_GAN=1e-4
SIGMA=-1.0

# CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN GAN --nsim $NSIM --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --sigma_gaussian $STD_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan $LR_GAN --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0
#
#
CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN cGAN --nsim $NSIM --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --sigma_gaussian $STD_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan $LR_GAN --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --eval

CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN CcGAN --nsim $NSIM --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --sigma_gaussian $STD_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan $LR_GAN --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0  --kernel_sigma $SIGMA --threshold_type hard --kappa -1.0 --eval

CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN CcGAN --nsim $NSIM --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --sigma_gaussian $STD_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan $LR_GAN --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0  --kernel_sigma $SIGMA --threshold_type soft --kappa -1.0 --eval


### Hard

# for kappa in 0.01
# do
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "CcGAN, normalize, HARD, kernel_sigma=$SIGMA, kappa=$kappa SEED=$SEED"
#   CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN CcGAN --nsim $NSIM --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --sigma_gaussian $STD_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan $LR_GAN --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0  --kernel_sigma $SIGMA --threshold_type hard --kappa $kappa --eval
# done

### Soft

# for kappa in 500.0 1000.0 1500.0 1800.0 2000.0 3600.0
# do
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "CcGAN, normalize, Soft, kernel_sigma=$SIGMA, kappa=$kappa SEED=$SEED"
#   CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN CcGAN --nsim $NSIM --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --sigma_gaussian $STD_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan $LR_GAN --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0  --kernel_sigma $SIGMA --threshold_type soft --kappa $kappa --eval
# done

















# CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN CcGAN --nsim 1 --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan $LR_GAN --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0  --kernel_sigma 1e-30 --threshold_type hard --kappa 1e-30 --eval
#
# CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN CcGAN --nsim 1 --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan $LR_GAN --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0  --kernel_sigma 1e-30 --threshold_type soft --kappa 1e+30 --eval
