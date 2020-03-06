#!/bin/bash

## tensorboard --logdir /home/xin/OneDrive/Working_directory/Continuous_cGAN/CellCounting/Output/saved_logs

echo "########################################################################################"
echo "                         Run script with ONE GPU !!!!                                   "
echo "########################################################################################"



SEED=2020


echo "########################################################################################"
echo "                         Pre-train a CNN for evaluation                                 "
echo "########################################################################################"
<<<<<<< HEAD
# CUDA_VISIBLE_DEVICES=0 python3 pretrain_CNN.py --CNN ResNet34 --epochs 200 --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --transform --CVMode
# CUDA_VISIBLE_DEVICES=0 python3 pretrain_CNN.py --CNN ResNet34 --epochs 200 --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --transform
=======
<<<<<<< HEAD
# CUDA_VISIBLE_DEVICES=0 python3 pretrain_CNN.py --CNN ResNet34 --epochs 200 --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --transform --CVMode
# CUDA_VISIBLE_DEVICES=0 python3 pretrain_CNN.py --CNN ResNet34 --epochs 200 --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --transform
=======
# CUDA_VISIBLE_DEVICES=1 python3 pretrain_CNN.py --CNN ResNet34 --epochs 200 --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --transform --CVMode
# CUDA_VISIBLE_DEVICES=1 python3 pretrain_CNN.py --CNN ResNet34 --epochs 200 --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --transform
>>>>>>> 7ac9937253c320ac79df75051df2f3f4848e2534
>>>>>>> 740531af0a182b1d40d461b5b996b744d21e1182


# echo "########################################################################################"
# echo "                                  Baseline GANs                                         "
# echo "########################################################################################"

# echo "-------------------------------------------------------------------------------------------------"
# echo "DCGAN"
<<<<<<< HEAD
# CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN DCGAN --transform --epoch_gan 1000 --lr_g_gan 1e-4 --lr_d_gan 1e-4 --batch_size_gan 64 --resumeTrain_gan 0 --seed $SEED

echo "-------------------------------------------------------------------------------------------------"
echo "cDCGAN"
CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN cDCGAN --transform --epoch_gan 1000 --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan 256 --resumeTrain_gan 0 --seed $SEED
=======
<<<<<<< HEAD
# CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN DCGAN --transform --epoch_gan 1000 --lr_g_gan 1e-4 --lr_d_gan 1e-4 --batch_size_gan 64 --resumeTrain_gan 0 --seed $SEED

echo "-------------------------------------------------------------------------------------------------"
echo "cDCGAN"
CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN cDCGAN --transform --epoch_gan 1000 --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan 256 --resumeTrain_gan 0 --seed $SEED



echo "########################################################################################"
echo "                                 Continuous_cDCGAN                                      "
echo "########################################################################################"
EPOCH_GAN=1000
BATCH_SIZE=256

### Hard

# for factor in 1.0 2.0 3.0 4.0
# do
#   kernel_sigma=0.01
#   kappa=$(echo "$factor*$kernel_sigma" | bc)
#
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "Continuous_cDCGAN, normalize, HARD, kernel_sigma=$kernel_sigma, kappa=$kappa SEED=$SEED"
#   CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN Continuous_cDCGAN --normalize_count --transform --kernel_sigma $kernel_sigma --threshold_type hard --kappa $kappa --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED
# done
#
# for factor in 1.0 2.0 3.0 4.0
# do
#   kernel_sigma=0.02
#   kappa=$(echo "$factor*$kernel_sigma" | bc)
#
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "Continuous_cDCGAN, normalize, HARD, kernel_sigma=$kernel_sigma, kappa=$kappa SEED=$SEED"
#   CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN Continuous_cDCGAN --normalize_count --transform --kernel_sigma $kernel_sigma --threshold_type hard --kappa $kappa --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED
# done


### Soft

# for factor in 1.0 2.0 3.0 4.0
# do
#   kernel_sigma=0.01
#   kappa=$(echo "$factor*$kernel_sigma" | bc)
#
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "Continuous_cDCGAN, normalize, Soft, kernel_sigma=$kernel_sigma, kappa=$kappa SEED=$SEED"
#   CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN Continuous_cDCGAN --normalize_count --transform --kernel_sigma $kernel_sigma --threshold_type soft --kappa $kappa --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED
# done
#
# for factor in 1.0 2.0 3.0 4.0
# do
#   kernel_sigma=0.02
#   kappa=$(echo "$factor*$kernel_sigma" | bc)
#
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "Continuous_cDCGAN, normalize, Soft, kernel_sigma=$kernel_sigma, kappa=$kappa SEED=$SEED"
#   CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN Continuous_cDCGAN --normalize_count --transform --kernel_sigma $kernel_sigma --threshold_type soft --kappa $kappa --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED
# done
=======
# CUDA_VISIBLE_DEVICES=1 python3 main.py --GAN DCGAN --transform --epoch_gan 1000 --lr_g_gan 1e-4 --lr_d_gan 1e-4 --batch_size_gan 64 --resumeTrain_gan 0 --seed $SEED

# echo "-------------------------------------------------------------------------------------------------"
# echo "cDCGAN"
# CUDA_VISIBLE_DEVICES=1 python3 main.py --GAN cDCGAN --transform --epoch_gan 1000 --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan 256 --resumeTrain_gan 0 --seed $SEED
>>>>>>> 7ac9937253c320ac79df75051df2f3f4848e2534
>>>>>>> 740531af0a182b1d40d461b5b996b744d21e1182



echo "########################################################################################"
echo "                                 Continuous_cDCGAN                                      "
echo "########################################################################################"
EPOCH_GAN=1000
BATCH_SIZE=256


<<<<<<< HEAD
# for factor in 1.0 2.0 3.0 4.0
# do
#   kernel_sigma=0.01
#   kappa=$(echo "$factor*$kernel_sigma" | bc)
#
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "Continuous_cDCGAN, normalize, HARD, kernel_sigma=$kernel_sigma, kappa=$kappa SEED=$SEED"
#   CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN Continuous_cDCGAN --normalize_count --transform --kernel_sigma $kernel_sigma --threshold_type hard --kappa $kappa --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED
# done
#
# for factor in 1.0 2.0 3.0 4.0
# do
#   kernel_sigma=0.02
#   kappa=$(echo "$factor*$kernel_sigma" | bc)
#
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "Continuous_cDCGAN, normalize, HARD, kernel_sigma=$kernel_sigma, kappa=$kappa SEED=$SEED"
#   CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN Continuous_cDCGAN --normalize_count --transform --kernel_sigma $kernel_sigma --threshold_type hard --kappa $kappa --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED
# done


### Soft

# for factor in 1.0 2.0 3.0 4.0
# do
#   kernel_sigma=0.01
#   kappa=$(echo "$factor*$kernel_sigma" | bc)
#
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "Continuous_cDCGAN, normalize, Soft, kernel_sigma=$kernel_sigma, kappa=$kappa SEED=$SEED"
#   CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN Continuous_cDCGAN --normalize_count --transform --kernel_sigma $kernel_sigma --threshold_type soft --kappa $kappa --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED
# done
#
# for factor in 1.0 2.0 3.0 4.0
# do
#   kernel_sigma=0.02
#   kappa=$(echo "$factor*$kernel_sigma" | bc)
#
#   echo "-------------------------------------------------------------------------------------------------"
#   echo "Continuous_cDCGAN, normalize, Soft, kernel_sigma=$kernel_sigma, kappa=$kappa SEED=$SEED"
#   CUDA_VISIBLE_DEVICES=0 python3 main.py --GAN Continuous_cDCGAN --normalize_count --transform --kernel_sigma $kernel_sigma --threshold_type soft --kappa $kappa --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED
# done


=======
<<<<<<< HEAD

=======
for factor in 1.0 2.0 3.0 4.0
do
  kernel_sigma=0.01
  kappa=$(echo "$factor*$kernel_sigma" | bc)

  echo "-------------------------------------------------------------------------------------------------"
  echo "Continuous_cDCGAN, normalize, HARD, kernel_sigma=$kernel_sigma, kappa=$kappa SEED=$SEED"
  CUDA_VISIBLE_DEVICES=1 python3 main.py --GAN Continuous_cDCGAN --normalize_count --transform --kernel_sigma $kernel_sigma --threshold_type hard --kappa $kappa --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED
done
>>>>>>> 7ac9937253c320ac79df75051df2f3f4848e2534
>>>>>>> 740531af0a182b1d40d461b5b996b744d21e1182


<<<<<<< HEAD
=======
<<<<<<< HEAD




# ###############################
# ## VGG dataset
# kernel_sigma=0.05
# kappa=0.1
# BATCH_SIZE=64
# echo "-------------------------------------------------------------------------------------------------"
# echo "Continuous_cDCGAN, normalize, HARD, kernel_sigma=$kernel_sigma, kappa=$kappa SEED=$SEED"
# CUDA_VISIBLE_DEVICES=0 python3 main.py --Dataset VGG --GAN Continuous_cDCGAN --normalize_count --transform --kernel_sigma $kernel_sigma --threshold_type hard --kappa $kappa --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED
=======
  echo "-------------------------------------------------------------------------------------------------"
  echo "Continuous_cDCGAN, normalize, HARD, kernel_sigma=$kernel_sigma, kappa=$kappa SEED=$SEED"
  CUDA_VISIBLE_DEVICES=1 python3 main.py --GAN Continuous_cDCGAN --normalize_count --transform --kernel_sigma $kernel_sigma --threshold_type hard --kappa $kappa --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED
done
>>>>>>> 740531af0a182b1d40d461b5b996b744d21e1182



<<<<<<< HEAD





# ###############################
# ## VGG dataset
# kernel_sigma=0.05
# kappa=0.1
# BATCH_SIZE=64
# echo "-------------------------------------------------------------------------------------------------"
# echo "Continuous_cDCGAN, normalize, HARD, kernel_sigma=$kernel_sigma, kappa=$kappa SEED=$SEED"
# CUDA_VISIBLE_DEVICES=0 python3 main.py --Dataset VGG --GAN Continuous_cDCGAN --normalize_count --transform --kernel_sigma $kernel_sigma --threshold_type hard --kappa $kappa --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED
=======
for factor in 1.0 2.0 3.0 4.0
do
  kernel_sigma=0.01
  kappa=$(echo "$factor*$kernel_sigma" | bc)

  echo "-------------------------------------------------------------------------------------------------"
  echo "Continuous_cDCGAN, normalize, Soft, kernel_sigma=$kernel_sigma, kappa=$kappa SEED=$SEED"
  CUDA_VISIBLE_DEVICES=1 python3 main.py --GAN Continuous_cDCGAN --normalize_count --transform --kernel_sigma $kernel_sigma --threshold_type soft --kappa $kappa --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED
done

for factor in 1.0 2.0 3.0 4.0
do
  kernel_sigma=0.02
  kappa=$(echo "$factor*$kernel_sigma" | bc)

  echo "-------------------------------------------------------------------------------------------------"
  echo "Continuous_cDCGAN, normalize, Soft, kernel_sigma=$kernel_sigma, kappa=$kappa SEED=$SEED"
  CUDA_VISIBLE_DEVICES=1 python3 main.py --GAN Continuous_cDCGAN --normalize_count --transform --kernel_sigma $kernel_sigma --threshold_type soft --kappa $kappa --dim_gan 128 --epoch_gan $EPOCH_GAN --lr_g_gan 2e-4 --lr_d_gan 1e-4 --batch_size_gan $BATCH_SIZE --resumeTrain_gan 0 --seed $SEED
done
>>>>>>> 7ac9937253c320ac79df75051df2f3f4848e2534
>>>>>>> 740531af0a182b1d40d461b5b996b744d21e1182
