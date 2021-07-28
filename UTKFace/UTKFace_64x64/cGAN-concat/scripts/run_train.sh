#!/bin/bash
## Path
ROOT_PATH="./UTKFace/UTKFace_64x64/cGAN-concat"
DATA_PATH="./datasets/UTKFace"
EVAL_PATH="./UTKFace/UTKFace_64x64/CcGAN/output/saved_models"

SEED=2021
NUM_WORKERS=0
MIN_LABEL=1
MAX_LABEL=60
IMG_SIZE=64
MAX_N_IMG_PER_LABEL=2000
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=200

NITERS=50000
BATCH_SIZE=256
LR_G=1e-4
LR_D=1e-4
DIM_Z=128
GAN_ARCH="SNGAN"

python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN_arch $GAN_ARCH --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 5000 \
    --lr_g $LR_G --lr_d $LR_D --dim_z $DIM_Z --batch_size $BATCH_SIZE \
    --visualize_freq 2000 --visualize_fake_images --comp_FID --dump_fake_for_NIQE \
    2>&1 | tee output_cGAN-concat.txt