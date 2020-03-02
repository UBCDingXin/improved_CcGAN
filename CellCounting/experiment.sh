#!/bin/bash

## tensorboard --logdir /home/xin/OneDrive/Working_directory/Continuous_cGAN/CellCounting/Output/saved_logs



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

### HARD

# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN Continuous_cDCGAN --normalize_count --transform --kernel_sigma 0.1 --threshold_type hard --kappa 0.5 --b_int_digits 4 --b_dec_digits 28 --dim_gan 128 --epoch_gan 2000 --lr_g_gan 1e-4 --lr_d_gan 1e-4 --batch_size_gan 128 --resumeTrain_gan 0

CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN Continuous_cDCGAN --transform --kernel_sigma 2 --threshold_type hard --kappa 5 --b_int_digits 16 --b_dec_digits 0 --dim_gan 128 --epoch_gan 2000 --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan 128 --resumeTrain_gan 0

### Soft

# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN Continuous_cDCGAN --normalize_count --transform --kernel_sigma 0.1 --threshold_type soft --b_int_digits 4 --b_dec_digits 28 --dim_gan 128 --epoch_gan 2000 --lr_g_gan 1e-4 --lr_d_gan 1e-4 --batch_size_gan 128 --resumeTrain_gan 0

# CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN Continuous_cDCGAN --transform --kernel_sigma 3 --threshold_type soft --b_int_digits 16 --b_dec_digits 0 --dim_gan 128 --epoch_gan 2000 --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan 128 --resumeTrain_gan 0




# python3 main.py --GAN Continuous_cDCGAN --normalize_count --transform --kernel_sigma 0.1 --threshold_type hard --b_int_digits 4 --b_dec_digits 28 --dim_gan 128 --epoch_gan 10000 --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan 256 --resumeTrain_gan 0

# python3 main.py --GAN Continuous_cDCGAN --transform --kernel_sigma 2 --threshold_type hard --b_int_digits 10 --b_dec_digits 0 --dim_gan 128 --epoch_gan 10000 --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan 128 --resumeTrain_gan 0



# python3 main.py --GAN Continuous_cDCGAN --normalize_count --transform --kernel_sigma 0.1 --threshold_type global_soft --b_int_digits 4 --b_dec_digits 28 --dim_gan 128 --epoch_gan 10000 --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan 128 --resumeTrain_gan 0

# python3 main.py --GAN Continuous_cDCGAN --transform --kernel_sigma 2 --threshold_type global_soft --b_int_digits 16 --b_dec_digits 0 --dim_gan 128 --epoch_gan 10000 --lr_g_gan 1e-4 --lr_d_gan 1e-4 --batch_size_gan 128 --resumeTrain_gan 0
