ROOT_PATH="./improved_CcGAN/SteeringAngle"
DATA_PATH="./improved_CcGAN/dataset/SteeringAngle"

SEED=2020
MIN_LABEL=-80.0
MAX_LABEL=80.0
MAX_N_IMG_PER_LABEL=9999
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=0
NITERS=20000
BATCH_SIZE_D_HARD=64
BATCH_SIZE_D_SOFT=64
BATCH_SIZE_D_ELSE=512
BATCH_SIZE_G=512
SIGMA=-1.0
KAPPA=-5.0
DIM_GAN=128
LR_G=1e-4
LR_D=1e-4

NUM_EVAL_LABELS=2000
NFAKE_PER_LABEL=50
SAMP_BATCH_SIZE=1000
FID_RADIUS=2
FID_NUM_CENTERS=1000
FID_EPOCH_CNN=200


echo "-------------------------------------------------------------------------------------------------"
echo "AE for SFID"
CUDA_VISIBLE_DEVICES=1,0 python3 pretrain_AE.py --root_path $ROOT_PATH --data_path $DATA_PATH --dim_bottleneck 512 --epochs 200 --resume_epoch 0 --batch_size_train 256 --batch_size_valid 64 --base_lr 1e-3 --seed $SEED --min_label $MIN_LABEL --max_label $MAX_LABEL


echo "-------------------------------------------------------------------------------------------------"
echo "Pre-trained CNN for computing Label Score"
CUDA_VISIBLE_DEVICES=1,0 python3 pretrain_CNN_regre.py --root_path $ROOT_PATH --data_path $DATA_PATH --CNN ResNet34_regre --epochs $FID_EPOCH_CNN --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --min_label $MIN_LABEL --max_label $MAX_LABEL

echo "-------------------------------------------------------------------------------------------------"
echo "classification CNN for computing Diversity"
CUDA_VISIBLE_DEVICES=1,0 python3 pretrain_CNN_class.py --root_path $ROOT_PATH --data_path $DATA_PATH --CNN ResNet34_class --epochs 20 --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED


echo "-------------------------------------------------------------------------------------------------"
echo "cGAN"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py --root_path $ROOT_PATH --data_path $DATA_PATH --GAN cGAN --seed $SEED --min_label $MIN_LABEL --max_label $MAX_LABEL --cGAN_num_classes 30 --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D_ELSE --batch_size_gene $BATCH_SIZE_G --num_eval_labels $NUM_EVAL_LABELS --nfake_per_label $NFAKE_PER_LABEL --visualize_fake_images --comp_FID --epoch_FID_CNN $FID_EPOCH_CNN --samp_batch_size $SAMP_BATCH_SIZE --FID_radius $FID_RADIUS --FID_num_centers $FID_NUM_CENTERS


echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN Hard"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN CcGAN --root_path $ROOT_PATH --data_path $DATA_PATH --seed $SEED --min_label $MIN_LABEL --max_label $MAX_LABEL --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA --threshold_type hard --kernel_sigma $SIGMA --kappa $KAPPA --dim_gan $DIM_GAN --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D_HARD --batch_size_gene $BATCH_SIZE_G --num_eval_labels $NUM_EVAL_LABELS --nfake_per_label $NFAKE_PER_LABEL --visualize_fake_images --comp_FID --epoch_FID_CNN $FID_EPOCH_CNN --samp_batch_size $SAMP_BATCH_SIZE --FID_radius $FID_RADIUS --FID_num_centers $FID_NUM_CENTERS #2>&1 | tee output_hard.txt


echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN Soft"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py --GAN CcGAN --root_path $ROOT_PATH --data_path $DATA_PATH --seed $SEED --min_label $MIN_LABEL --max_label $MAX_LABEL --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA --threshold_type soft --kernel_sigma $SIGMA --kappa $KAPPA --dim_gan $DIM_GAN --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D_SOFT --batch_size_gene $BATCH_SIZE_G --num_eval_labels $NUM_EVAL_LABELS --nfake_per_label $NFAKE_PER_LABEL --visualize_fake_images --comp_FID --epoch_FID_CNN $FID_EPOCH_CNN --samp_batch_size $SAMP_BATCH_SIZE --FID_radius $FID_RADIUS --FID_num_centers $FID_NUM_CENTERS #2>&1 | tee output_soft.txt
