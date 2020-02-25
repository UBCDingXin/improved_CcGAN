#!/bin/bash


echo "-------------------------------------------------------------------------------------------------"
echo "cDCGAN"
python3 main.py --GAN cDCGAN --transform --normalize_count --epoch_gan 1000 --lr_g_gan 1e-4 --lr_d_gan 1e-4 --batch_size_gan 64 --resumeTrain_gan 0
