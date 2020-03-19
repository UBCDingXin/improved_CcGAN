#!/bin/bash



SEED=2020
NSIM=3
EPOCH=10000
N_GANSSIANS=120
N_SAMP_PER_GAUSSIAN=10
RADIUS=1
BATCH_SIZE=128
LR_GAN=1e-4
SIGMA=-1

# python3 main.py --GAN GAN --nsim $NSIM --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan $LR_GAN --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0
#
#
# python3 main.py --GAN cGAN --nsim $NSIM --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan $LR_GAN --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --eval

# python3 main.py --GAN CcGAN --nsim $NSIM --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan $LR_GAN --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0  --kernel_sigma $SIGMA --threshold_type hard --kappa -1.0 --eval #--eval_label 1.570796


# python3 main.py --GAN CcGAN --nsim $NSIM --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan $LR_GAN --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0  --kernel_sigma $SIGMA --threshold_type soft --kappa 5 --eval


### Hard

# for kappa in 0.02 #0.001 0.005 0.01 0.1
# do
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "CcGAN, normalize, HARD, kernel_sigma=$SIGMA, kappa=$kappa SEED=$SEED"
#   python3 main.py --GAN CcGAN --nsim $NSIM --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan $LR_GAN --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0  --kernel_sigma $SIGMA --threshold_type hard --kappa $kappa --eval
# done

## Soft

# for kappa in 0.1 0.5 0.8 1.0 2.0 5.0 10.0 20.0
# do
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "CcGAN, normalize, Soft, kernel_sigma=$SIGMA, kappa=$kappa SEED=$SEED"
#   python3 main.py --GAN CcGAN --nsim $NSIM --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan $LR_GAN --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0  --kernel_sigma $SIGMA --threshold_type soft --kappa $kappa --eval
# done

















# python3 main.py --GAN CcGAN --nsim 1 --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan $LR_GAN --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0  --kernel_sigma 1e-30 --threshold_type hard --kappa 1e-30 --eval
#
# python3 main.py --GAN CcGAN --nsim 1 --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan $LR_GAN --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0  --kernel_sigma 1e-30 --threshold_type soft --kappa 1e+30 --eval
