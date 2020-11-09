
ROOT_PATH="YOUR WORKING DIR"
SEED=2020
NSIM=3
NITERS=6000
N_GANSSIANS=120
N_SAMP_PER_GAUSSIAN=10
STD_GAUSSIAN=0.02
RADIUS=1
BATCH_SIZE_D=128
BATCH_SIZE_G=128
LR_GAN=5e-5
SIGMA=-1.0
KAPPA=-2.0
DIM_GAN=2


echo "-------------------------------------------------------------------------------------------------"
echo "cGAN"
CUDA_VISIBLE_DEVICES=0 python3 main.py --root_path $ROOT_PATH --GAN cGAN --nsim $NSIM --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --sigma_gaussian $STD_GAUSSIAN --radius $RADIUS --niters_gan $NITERS --resume_niters_gan 0 --lr_gan $LR_GAN --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --eval --dim_gan $DIM_GAN 2>&1 | tee output_cGAN.txt

echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN Hard"
CUDA_VISIBLE_DEVICES=0 python3 main.py --root_path $ROOT_PATH --GAN CcGAN --nsim $NSIM --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --sigma_gaussian $STD_GAUSSIAN --radius $RADIUS --niters_gan $NITERS --resume_niters_gan 0 --lr_gan $LR_GAN --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --kernel_sigma $SIGMA --threshold_type hard --kappa $KAPPA --eval --dim_gan $DIM_GAN # 2>&1 | tee output_hard.txt

echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN Soft"
CUDA_VISIBLE_DEVICES=0 python3 main.py --root_path $ROOT_PATH --GAN CcGAN --nsim $NSIM --seed $SEED --n_gaussians $N_GANSSIANS --n_samp_per_gaussian_train $N_SAMP_PER_GAUSSIAN --sigma_gaussian $STD_GAUSSIAN --radius $RADIUS --niters_gan $NITERS --resume_niters_gan 0 --lr_gan $LR_GAN --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --kernel_sigma $SIGMA --threshold_type soft --kappa $KAPPA --eval --dim_gan $DIM_GAN # 2>&1 | tee output_soft.txt
