ROOT_PATH="./improved_CcGAN/RC-49-improved"
DATA_PATH="./improved_CcGAN/dataset/RC-49"

SEED=2020
MIN_LABEL=0.0
MAX_LABEL=90.0
MAX_N_IMG_PER_LABEL=25
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0
NITERS=30000
BATCH_SIZE_D=256
BATCH_SIZE_G=256
SIGMA=-1.0
KAPPA=-2.0
LR_G=1e-4
LR_D=1e-4

DIM_GAN=256
DIM_EMBED=128
LOSS_TYPE='vanilla'

FID_EPOCH_CNN=200
NUM_EVAL_LABELS=-1
NFAKE_PER_LABEL=200
SAMP_BATCH_SIZE=1000
NREAL_PER_LABEL=10 #number of real images for each label




FID_RADIUS=0
FID_NUM_CENTERS=-1
## Eval mode 1: Only use real images for GAN training to compute the SFID and only evaluate on the seen labels
echo "-------------------------------------------------------------------------------------------------"
echo "Radius: ${FID_RADIUS}; # Centers: ${FID_NUM_CENTERS} "
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--root_path $ROOT_PATH --data_path $DATA_PATH \
--GAN CcGAN --seed $SEED \
--min_label $MIN_LABEL --max_label $MAX_LABEL --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
--threshold_type hard --kernel_sigma $SIGMA --kappa $KAPPA \
--dim_gan $DIM_GAN \
--dim_embed $DIM_EMBED \
--niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 \
--lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
--visualize_fake_images \
--comp_FID --eval_mode 1 \
--num_eval_labels $NUM_EVAL_LABELS --nfake_per_label $NFAKE_PER_LABEL --nreal_per_label $NREAL_PER_LABEL \
--samp_batch_size $SAMP_BATCH_SIZE --FID_radius $FID_RADIUS --FID_num_centers $FID_NUM_CENTERS \
2>&1 | tee output_test_SFID_radius_${FID_RADIUS}_numcenter_${FID_NUM_CENTERS}.txt




FID_RADIUS=0.5
FID_NUM_CENTERS=400
## Eval mode 3: Only use real images for GAN training to compute the SFID.
echo "-------------------------------------------------------------------------------------------------"
echo "Radius: ${FID_RADIUS}; # Centers: ${FID_NUM_CENTERS} "
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--root_path $ROOT_PATH --data_path $DATA_PATH \
--GAN CcGAN --seed $SEED \
--min_label $MIN_LABEL --max_label $MAX_LABEL --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
--threshold_type hard --kernel_sigma $SIGMA --kappa $KAPPA \
--dim_gan $DIM_GAN \
--dim_embed $DIM_EMBED \
--niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 \
--lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
--visualize_fake_images \
--comp_FID --eval_mode 3 \
--num_eval_labels $NUM_EVAL_LABELS --nfake_per_label $NFAKE_PER_LABEL \
--samp_batch_size $SAMP_BATCH_SIZE --FID_radius $FID_RADIUS --FID_num_centers $FID_NUM_CENTERS \

FID_RADIUS=1
FID_NUM_CENTERS=400
## Eval mode 3: Only use real images for GAN training to compute the SFID.
echo "-------------------------------------------------------------------------------------------------"
echo "Radius: ${FID_RADIUS}; # Centers: ${FID_NUM_CENTERS} "
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--root_path $ROOT_PATH --data_path $DATA_PATH \
--GAN CcGAN --seed $SEED \
--min_label $MIN_LABEL --max_label $MAX_LABEL --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
--threshold_type hard --kernel_sigma $SIGMA --kappa $KAPPA \
--dim_gan $DIM_GAN \
--dim_embed $DIM_EMBED \
--niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 \
--lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
--visualize_fake_images \
--comp_FID --eval_mode 3 \
--num_eval_labels $NUM_EVAL_LABELS --nfake_per_label $NFAKE_PER_LABEL \
--samp_batch_size $SAMP_BATCH_SIZE --FID_radius $FID_RADIUS --FID_num_centers $FID_NUM_CENTERS \

FID_RADIUS=2
FID_NUM_CENTERS=400
## Eval mode 3: Only use real images for GAN training to compute the SFID.
echo "-------------------------------------------------------------------------------------------------"
echo "Radius: ${FID_RADIUS}; # Centers: ${FID_NUM_CENTERS} "
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--root_path $ROOT_PATH --data_path $DATA_PATH \
--GAN CcGAN --seed $SEED \
--min_label $MIN_LABEL --max_label $MAX_LABEL --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
--threshold_type hard --kernel_sigma $SIGMA --kappa $KAPPA \
--dim_gan $DIM_GAN \
--dim_embed $DIM_EMBED \
--niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 \
--lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
--visualize_fake_images \
--comp_FID --eval_mode 3 \
--num_eval_labels $NUM_EVAL_LABELS --nfake_per_label $NFAKE_PER_LABEL \
--samp_batch_size $SAMP_BATCH_SIZE --FID_radius $FID_RADIUS --FID_num_centers $FID_NUM_CENTERS \



FID_RADIUS=0.5
FID_NUM_CENTERS=600
echo "-------------------------------------------------------------------------------------------------"
echo "Radius: ${FID_RADIUS}; # Centers: ${FID_NUM_CENTERS} "
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--root_path $ROOT_PATH --data_path $DATA_PATH \
--GAN CcGAN --seed $SEED \
--min_label $MIN_LABEL --max_label $MAX_LABEL --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
--threshold_type hard --kernel_sigma $SIGMA --kappa $KAPPA \
--dim_gan $DIM_GAN \
--dim_embed $DIM_EMBED \
--niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 \
--lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
--visualize_fake_images \
--comp_FID --eval_mode 3 \
--num_eval_labels $NUM_EVAL_LABELS --nfake_per_label $NFAKE_PER_LABEL \
--samp_batch_size $SAMP_BATCH_SIZE --FID_radius $FID_RADIUS --FID_num_centers $FID_NUM_CENTERS \

FID_RADIUS=1
FID_NUM_CENTERS=600
echo "-------------------------------------------------------------------------------------------------"
echo "Radius: ${FID_RADIUS}; # Centers: ${FID_NUM_CENTERS} "
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--root_path $ROOT_PATH --data_path $DATA_PATH \
--GAN CcGAN --seed $SEED \
--min_label $MIN_LABEL --max_label $MAX_LABEL --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
--threshold_type hard --kernel_sigma $SIGMA --kappa $KAPPA \
--dim_gan $DIM_GAN \
--dim_embed $DIM_EMBED \
--niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 \
--lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
--visualize_fake_images \
--comp_FID --eval_mode 3 \
--num_eval_labels $NUM_EVAL_LABELS --nfake_per_label $NFAKE_PER_LABEL \
--samp_batch_size $SAMP_BATCH_SIZE --FID_radius $FID_RADIUS --FID_num_centers $FID_NUM_CENTERS \

FID_RADIUS=2
FID_NUM_CENTERS=600
echo "-------------------------------------------------------------------------------------------------"
echo "Radius: ${FID_RADIUS}; # Centers: ${FID_NUM_CENTERS} "
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--root_path $ROOT_PATH --data_path $DATA_PATH \
--GAN CcGAN --seed $SEED \
--min_label $MIN_LABEL --max_label $MAX_LABEL --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
--threshold_type hard --kernel_sigma $SIGMA --kappa $KAPPA \
--dim_gan $DIM_GAN \
--dim_embed $DIM_EMBED \
--niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 \
--lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
--visualize_fake_images \
--comp_FID --eval_mode 3 \
--num_eval_labels $NUM_EVAL_LABELS --nfake_per_label $NFAKE_PER_LABEL \
--samp_batch_size $SAMP_BATCH_SIZE --FID_radius $FID_RADIUS --FID_num_centers $FID_NUM_CENTERS \






FID_RADIUS=0.5
FID_NUM_CENTERS=800
echo "-------------------------------------------------------------------------------------------------"
echo "Radius: ${FID_RADIUS}; # Centers: ${FID_NUM_CENTERS} "
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--root_path $ROOT_PATH --data_path $DATA_PATH \
--GAN CcGAN --seed $SEED \
--min_label $MIN_LABEL --max_label $MAX_LABEL --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
--threshold_type hard --kernel_sigma $SIGMA --kappa $KAPPA \
--dim_gan $DIM_GAN \
--dim_embed $DIM_EMBED \
--niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 \
--lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
--visualize_fake_images \
--comp_FID --eval_mode 3 \
--num_eval_labels $NUM_EVAL_LABELS --nfake_per_label $NFAKE_PER_LABEL \
--samp_batch_size $SAMP_BATCH_SIZE --FID_radius $FID_RADIUS --FID_num_centers $FID_NUM_CENTERS \

FID_RADIUS=1
FID_NUM_CENTERS=800
echo "-------------------------------------------------------------------------------------------------"
echo "Radius: ${FID_RADIUS}; # Centers: ${FID_NUM_CENTERS} "
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--root_path $ROOT_PATH --data_path $DATA_PATH \
--GAN CcGAN --seed $SEED \
--min_label $MIN_LABEL --max_label $MAX_LABEL --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
--threshold_type hard --kernel_sigma $SIGMA --kappa $KAPPA \
--dim_gan $DIM_GAN \
--dim_embed $DIM_EMBED \
--niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 \
--lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
--visualize_fake_images \
--comp_FID --eval_mode 3 \
--num_eval_labels $NUM_EVAL_LABELS --nfake_per_label $NFAKE_PER_LABEL \
--samp_batch_size $SAMP_BATCH_SIZE --FID_radius $FID_RADIUS --FID_num_centers $FID_NUM_CENTERS \

FID_RADIUS=2
FID_NUM_CENTERS=800
echo "-------------------------------------------------------------------------------------------------"
echo "Radius: ${FID_RADIUS}; # Centers: ${FID_NUM_CENTERS} "
CUDA_VISIBLE_DEVICES=1,0 python3 main.py \
--root_path $ROOT_PATH --data_path $DATA_PATH \
--GAN CcGAN --seed $SEED \
--min_label $MIN_LABEL --max_label $MAX_LABEL --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA \
--threshold_type hard --kernel_sigma $SIGMA --kappa $KAPPA \
--dim_gan $DIM_GAN \
--dim_embed $DIM_EMBED \
--niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 \
--lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G \
--visualize_fake_images \
--comp_FID --eval_mode 3 \
--num_eval_labels $NUM_EVAL_LABELS --nfake_per_label $NFAKE_PER_LABEL \
--samp_batch_size $SAMP_BATCH_SIZE --FID_radius $FID_RADIUS --FID_num_centers $FID_NUM_CENTERS \
