## Path
ROOT_PATH="./RC-49/RC-49_256x256/CcGAN-improved"
DATA_PATH="./datasets/RC-49"
EVAL_PATH="./RC-49/RC-49_256x256/CcGAN-improved/output/eval_models"

SEED=2020
NUM_WORKERS=0
MIN_LABEL=0
MAX_LABEL=90
IMG_SIZE=256
MAX_N_IMG_PER_LABEL=25
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0

NITERS=30000
BATCH_SIZE_G=256
BATCH_SIZE_D=256
NUM_D_STEPS=2
SIGMA=-1.0
KAPPA=-2.0
LR_G=1e-4
LR_D=1e-4
GAN_ARCH="SAGAN"
LOSS_TYPE="hinge"

NUM_EVAL_LABELS=-1
NFAKE_PER_LABEL=200
SAMP_BATCH_SIZE=1000
FID_RADIUS=0
FID_NUM_CENTERS=-1



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
    --img_size $IMG_SIZE --min_label $MIN_LABEL --max_label $MAX_LABEL \
    2>&1 | tee output_CNN_class.txt


python pretrain_CNN_regre.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --CNN ResNet34_regre \
    --epochs 200 --batch_size_train 256 --batch_size_valid 10  \
    --base_lr 0.01 --weight_dacay 1e-4 \
    --img_size $IMG_SIZE --min_label $MIN_LABEL --max_label $MAX_LABEL \
    2>&1 | tee output_CNN_regre.txt


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
    --comp_FID --samp_batch_size $SAMP_BATCH_SIZE --FID_radius $FID_RADIUS --FID_num_centers $FID_NUM_CENTERS \
    --num_eval_labels $NUM_EVAL_LABELS --nfake_per_label $NFAKE_PER_LABEL \
    --dump_fake_for_NIQE \
    2>&1 | tee output_CcGAN_30K.txt


GAN="cGAN"
DIM_GAN=128
resume_niters_gan=0
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN $GAN --GAN_arch $GAN_ARCH --cGAN_num_classes 150 --niters_gan $NITERS --resume_niters_gan $resume_niters_gan --loss_type_gan $LOSS_TYPE \
    --save_niters_freq 2000 --visualize_freq 1000 \
    --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --num_D_steps $NUM_D_STEPS \
    --lr_g $LR_G --lr_d $LR_D --dim_gan $DIM_GAN \
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout \
    --visualize_fake_images \
    --comp_FID --samp_batch_size $SAMP_BATCH_SIZE --FID_radius $FID_RADIUS --FID_num_centers $FID_NUM_CENTERS \
    --num_eval_labels $NUM_EVAL_LABELS --nfake_per_label $NFAKE_PER_LABEL \
    --dump_fake_for_NIQE \
    2>&1 | tee output_cGAN_150classes_30K.txt


GAN="cGAN-concat"
DIM_GAN=128
resume_niters_gan=0
python main.py \
    --root_path $ROOT_PATH --data_path $DATA_PATH --eval_ckpt_path $EVAL_PATH --seed $SEED --num_workers $NUM_WORKERS \
    --min_label $MIN_LABEL --max_label $MAX_LABEL --img_size $IMG_SIZE \
    --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
    --GAN $GAN --GAN_arch $GAN_ARCH --niters_gan $NITERS --resume_niters_gan $resume_niters_gan --loss_type_gan $LOSS_TYPE \
    --save_niters_freq 2000 --visualize_freq 1000 \
    --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --num_D_steps $NUM_D_STEPS \
    --lr_g $LR_G --lr_d $LR_D --dim_gan $DIM_GAN \
    --gan_DiffAugment --gan_DiffAugment_policy color,translation,cutout \
    --visualize_fake_images \
    --comp_FID --samp_batch_size $SAMP_BATCH_SIZE --FID_radius $FID_RADIUS --FID_num_centers $FID_NUM_CENTERS \
    --num_eval_labels $NUM_EVAL_LABELS --nfake_per_label $NFAKE_PER_LABEL \
    --dump_fake_for_NIQE \
    2>&1 | tee output_cGAN-concat_30K.txt