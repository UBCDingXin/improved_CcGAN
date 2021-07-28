#!/bin/bash

## Path
ROOT_PATH="./SteeringAngle/SteeringAngle_64x64/cGAN-concat"
DATA_PATH="./datasets/SteeringAngle"
EVAL_PATH="./SteeringAngle/SteeringAngle_64x64/CcGAN-improved/output/saved_models"

SEED=2020
NUM_WORKERS=0
MIN_LABEL=-80.0
MAX_LABEL=80.0
IMG_SIZE=64
MAX_N_IMG_PER_LABEL=9999
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0

NUM_EVAL_LABELS=2000
NFAKE_PER_LABEL=50
SAMP_BATCH_SIZE=500
FID_RADIUS=2
FID_NUM_CENTERS=1000
FID_EPOCH_CNN=200

NITERS=20000
BATCH_SIZE=512
LR_G=1e-4
LR_D=1e-4
DIM_Z=128
GAN_ARCH="SNGAN"


resume_niters_gan=0
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH \
    --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN_arch $GAN_ARCH --niters_gan $NITERS --resume_niters_gan $resume_niters_gan --save_niters_freq 2000 \
    --lr_g $LR_G --lr_d $LR_D --dim_z $DIM_Z --batch_size $BATCH_SIZE \
    --visualize_freq 2000 --visualize_fake_images \
    --comp_FID \
    --num_eval_labels $NUM_EVAL_LABELS --nfake_per_label $NFAKE_PER_LABEL \
    --epoch_FID_CNN $FID_EPOCH_CNN --samp_batch_size $SAMP_BATCH_SIZE \
    --FID_num_centers $FID_NUM_CENTERS --FID_radius $FID_RADIUS \
    --dump_fake_for_NIQE \
    2>&1 | tee output_${NITERS}.txt