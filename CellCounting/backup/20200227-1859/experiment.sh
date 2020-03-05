#!/bin/bash

# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN"
# python3 main.py --GAN DCGAN --transform --epoch_gan 1000 --lr_g_gan 1e-4 --lr_d_gan 1e-4 --batch_size_gan 64 --resumeTrain_gan 0


# echo "-------------------------------------------------------------------------------------------------"
# echo "cDCGAN"
# python3 main.py --GAN cDCGAN --transform --epoch_gan 1000 --lr_g_gan 1e-4 --lr_d_gan 1e-4 --batch_size_gan 64 --resumeTrain_gan 0


# echo "-------------------------------------------------------------------------------------------------"
# echo "WGANGP"
# python3 main.py --GAN WGANGP --transform --epoch_gan 2000 --lr_g_gan 1e-4 --lr_d_gan 1e-4 --batch_size_gan 64 --resumeTrain_gan 0


# echo "-------------------------------------------------------------------------------------------------"
# echo "cWGANGP"
# python3 main.py --GAN cWGANGP --transform --epoch_gan 1000 --lr_g_gan 1e-4 --lr_d_gan 1e-4 --batch_size_gan 64 --resumeTrain_gan 0


echo "-------------------------------------------------------------------------------------------------"
echo "Continuous_cDCGAN"
python3 main.py --GAN Continuous_cDCGAN --normalize_count --transform --epoch_gan 1000 --lr_g_gan 1e-4 --lr_d_gan 1e-4 --batch_size_gan 32 --resumeTrain_gan 0
