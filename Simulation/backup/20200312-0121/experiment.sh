#!/bin/bash



SEED=2020
EPOCH=2000
N_GANSSIANS=50
N_SAMP_PER_GAUSSIAN=10
RADIUS=5

# python3 main.py --GAN GAN --nsim 1 --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan 1e-3 --batch_size_gan 128 --resumeTrain_gan 0
#
#
# python3 main.py --GAN cGAN --nsim 1 --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan 1e-3 --batch_size_gan 128 --resumeTrain_gan 0


python3 main.py --GAN CcGAN --nsim 1 --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --radius $RADIUS --epoch_gan $EPOCH --lr_gan 1e-3 --batch_size_gan 256 --resumeTrain_gan 0  --kernel_sigma 0.01 --threshold_type soft --kappa 0.5
