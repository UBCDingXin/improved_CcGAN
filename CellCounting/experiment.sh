#!/bin/bash

# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN"
# python3 main.py --GAN DCGAN --transform --epoch_gan 1000 --lr_g_gan 1e-4 --lr_d_gan 1e-4 --batch_size_gan 64 --resumeTrain_gan 0


# echo "-------------------------------------------------------------------------------------------------"
# echo "cDCGAN"
# python3 main.py --GAN cDCGAN --transform --epoch_gan 1000 --lr_g_gan 1e-4 --lr_d_gan 1e-4 --batch_size_gan 64 --resumeTrain_gan 0


# echo "-------------------------------------------------------------------------------------------------"
# echo "cDCGAN normalize count"
# python3 main.py --GAN cDCGAN --transform --normalize_count --epoch_gan 1000 --lr_g_gan 1e-4 --lr_d_gan 1e-4 --batch_size_gan 64 --resumeTrain_gan 0


echo "-------------------------------------------------------------------------------------------------"
echo "WGANGP"
python3 main.py --GAN WGANGP --transform --normalize_count --epoch_gan 2000 --lr_g_gan 1e-4 --lr_d_gan 1e-4 --batch_size_gan 64 --resumeTrain_gan 0


echo "-------------------------------------------------------------------------------------------------"
echo "cWGANGP normalize count"
python3 main.py --GAN cWGANGP --transform --normalize_count --epoch_gan 2000 --lr_g_gan 1e-4 --lr_d_gan 1e-4 --batch_size_gan 64 --resumeTrain_gan 0
