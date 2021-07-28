#!/bin/bash

## Path
ROOT_PATH="./SteeringAngle/SteeringAngle_128x128/CcGAN-improved"
DATA_PATH="./datasets/SteeringAngle"
EVAL_PATH="./SteeringAngle/SteeringAngle_128x128/CcGAN-improved/output/eval_models"

SEED=2020
NUM_WORKERS=0
MIN_LABEL=-80.0
MAX_LABEL=80.0
IMG_SIZE=128
MAX_N_IMG_PER_LABEL=9999
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0

NITERS=20000
BATCH_SIZE_G=256
BATCH_SIZE_D=256
NUM_D_STEPS=2
SIGMA=-1.0
KAPPA=-5.0
LR_G=1e-4
LR_D=1e-4

GAN_ARCH="SAGAN"
LOSS_TYPE="hinge"

NUM_EVAL_LABELS=2000
NFAKE_PER_LABEL=50
SAMP_BATCH_SIZE=500
FID_RADIUS=2
FID_NUM_CENTERS=1000
FID_EPOCH_CNN=200


python pretrain_AE.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --dim_bottleneck 512 --epochs 200 --resume_epoch 0 \
    --batch_size_train 256 --batch_size_valid 10 \
    --base_lr 1e-3 --lr_decay_epochs 50 --lr_decay_factor 0.1 \
    --lambda_sparsity 0 --weight_dacay 1e-4 \
    --img_size $IMG_SIZE --min_label $MIN_LABEL --max_label $MAX_LABEL \
    2>&1 | tee output_AE.txt

python pretrain_CNN_class.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --CNN ResNet34_class \
    --epochs 200 --batch_size_train 256 --batch_size_valid 10  \
    --base_lr 0.01 --weight_dacay 1e-4 \
    --img_size $IMG_SIZE \
    2>&1 | tee output_CNN_class.txt

python pretrain_CNN_regre.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --CNN ResNet34_regre \
    --epochs 200 --batch_size_train 256 --batch_size_valid 10  \
    --base_lr 0.01 --weight_dacay 1e-4 \
    --img_size $IMG_SIZE --min_label $MIN_LABEL --max_label $MAX_LABEL \
    2>&1 | tee output_CNN_regre.txt


GAN="cGAN"
DIM_GAN=128
cGAN_NUM_CLASSES=210
resume_niters_gan=0
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN $GAN --GAN_arch $GAN_ARCH --cGAN_num_classes $cGAN_NUM_CLASSES --loss_type_gan $LOSS_TYPE \
    --niters_gan $NITERS --resume_niters_gan $resume_niters_gan --num_D_steps $NUM_D_STEPS \
    --save_niters_freq 2000 --visualize_freq 1000 \
    --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
    --lr_g $LR_G --lr_d $LR_D --dim_gan $DIM_GAN \
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout \
    --visualize_fake_images \
    --comp_FID --FID_radius $FID_RADIUS --nfake_per_label $NFAKE_PER_LABEL \
    --dump_fake_for_NIQE \
    2>&1 | tee output_${GAN}_${NITERS}.txt


GAN="cGAN-concat"
DIM_GAN=128
resume_niters_gan=0
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN $GAN --GAN_arch $GAN_ARCH --loss_type_gan $LOSS_TYPE \
    --niters_gan $NITERS --resume_niters_gan $resume_niters_gan --num_D_steps $NUM_D_STEPS \
    --save_niters_freq 2000 --visualize_freq 1000 \
    --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
    --lr_g $LR_G --lr_d $LR_D --dim_gan $DIM_GAN \
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout \
    --visualize_fake_images \
    --comp_FID --FID_radius $FID_RADIUS --nfake_per_label $NFAKE_PER_LABEL \
    --dump_fake_for_NIQE \
    2>&1 | tee output_${GAN}_${NITERS}.txt


GAN="CcGAN"
DIM_GAN=256
DIM_EMBED=128
resume_niters_gan=0
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN $GAN --GAN_arch $GAN_ARCH --niters_gan $NITERS --resume_niters_gan $resume_niters_gan --loss_type_gan $LOSS_TYPE \
    --save_niters_freq 2500 --visualize_freq 1000 \
    --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --num_D_steps $NUM_D_STEPS \
    --lr_g $LR_G --lr_d $LR_D --dim_gan $DIM_GAN --dim_embed $DIM_EMBED \
    --kernel_sigma $SIGMA --threshold_type soft --kappa $KAPPA \
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout \
    --visualize_fake_images \
    --comp_FID --num_eval_labels $NUM_EVAL_LABELS \
    --samp_batch_size $SAMP_BATCH_SIZE --nfake_per_label $NFAKE_PER_LABEL \
    --epoch_FID_CNN $FID_EPOCH_CNN --FID_radius $FID_RADIUS --FID_num_centers $FID_NUM_CENTERS \
    --dump_fake_for_NIQE \
    2>&1 | tee output_${GAN}_${NITERS}.txt
