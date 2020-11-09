
ROOT_PATH="./improved_CcGAN/Cell200"
DATA_PATH="./improved_CcGAN/dataset/Cell200"

SEED=2020
NITERS=5000
BATCH_SIZE_D=32
BATCH_SIZE_G=512
START_COUNT=1
END_COUNT=200
STEPSIZE_COUNT=2
N_IMGS_PER_CELLCOUNT=10
SIGMA=-1.0
KAPPA=-2.0
LR_G=1e-4
LR_D=1e-4
NFAKE_PER_LABEL=1000
FID_RADIUS=0
DIM_GAN=128



echo "-------------------------------------------------------------------------------------------------"
echo "AE for Intra-FID"
CUDA_VISIBLE_DEVICES=1,0 python3 pretrain_AE.py --root_path $ROOT_PATH --data_path $DATA_PATH --dim_bottleneck 512 --epochs 200 --resume_epoch 0 --batch_size_train 256 --batch_size_valid 64 --base_lr 1e-3 --seed $SEED --min_label $START_COUNT --max_label $END_COUNT


echo "-------------------------------------------------------------------------------------------------"
echo "regression CNN for Label Score"
CUDA_VISIBLE_DEVICES=1,0 python3 pretrain_CNN_regre.py --root_path $ROOT_PATH --data_path $DATA_PATH --start_count $START_COUNT --end_count $END_COUNT --CNN ResNet34_regre --epochs 200 --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --transform


echo "-------------------------------------------------------------------------------------------------"
echo "cGAN"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py --root_path $ROOT_PATH --data_path $DATA_PATH --GAN cGAN --cGAN_num_classes 100 --start_count $START_COUNT --end_count $END_COUNT --stepsize_count $STEPSIZE_COUNT --num_imgs_per_count $N_IMGS_PER_CELLCOUNT --transform --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --seed $SEED --visualize_fake_images --comp_FID --nfake_per_label $NFAKE_PER_LABEL --samp_batch_size 1000 --FID_radius

echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN: Hard"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py --root_path $ROOT_PATH --data_path $DATA_PATH --GAN CcGAN --start_count $START_COUNT --end_count $END_COUNT --stepsize_count $STEPSIZE_COUNT --num_imgs_per_count $N_IMGS_PER_CELLCOUNT --transform --kernel_sigma $SIGMA --threshold_type hard --kappa $KAPPA --dim_gan $DIM_GAN --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --seed $SEED --visualize_fake_images --comp_FID --nfake_per_label $NFAKE_PER_LABEL --samp_batch_size 1000 --FID_radius $FID_RADIUS --dump_fake_for_NIQE


echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN: Soft"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py --root_path $ROOT_PATH --data_path $DATA_PATH --GAN CcGAN --start_count $START_COUNT --end_count $END_COUNT --stepsize_count $STEPSIZE_COUNT --num_imgs_per_count $N_IMGS_PER_CELLCOUNT --transform --kernel_sigma $SIGMA --threshold_type soft --kappa $KAPPA --dim_gan $DIM_GAN --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --seed $SEED --visualize_fake_images --comp_FID --nfake_per_label $NFAKE_PER_LABEL --samp_batch_size 1000 --FID_radius $FID_RADIUS --dump_fake_for_NIQE


echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN: Hard limit"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py --root_path $ROOT_PATH --data_path $DATA_PATH --GAN CcGAN --start_count $START_COUNT --end_count $END_COUNT --stepsize_count $STEPSIZE_COUNT --num_imgs_per_count $N_IMGS_PER_CELLCOUNT --transform --kernel_sigma 1e-30 --threshold_type hard --kappa 1e-30 --dim_gan $DIM_GAN --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --seed $SEED --visualize_fake_images --comp_FID --nfake_per_label $NFAKE_PER_LABEL --samp_batch_size 1000 --FID_radius $FID_RADIUS
