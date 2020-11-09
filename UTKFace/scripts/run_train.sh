ROOT_PATH="./improved_CcGAN/UTKFace"
DATA_PATH="./improved_CcGAN/dataset/UTKFace"


SEED=2020
NITERS=40000
BATCH_SIZE_D=512
BATCH_SIZE_G=512
MIN_AGE=1
MAX_AGE=60
MAX_N_IMG_PER_LABEL=99999
MAX_N_IMG_PER_LABEL_AFTER_REPLICA=200
SIGMA=-1.0
KAPPA=-2.0
DIM_GAN=128
LR_G=1e-4
LR_D=1e-4
NFAKE_PER_LABEL=1000
FID_EPOCH_CNN=200
FID_RADIUS=0


echo "-------------------------------------------------------------------------------------------------"
echo "AE for Intra-FID"
CUDA_VISIBLE_DEVICES=1,0 python3 pretrain_AE.py --root_path $ROOT_PATH --data_path $DATA_PATH --dim_bottleneck 512 --epochs 200 --resume_epoch 0 --batch_size_train 256 --batch_size_valid 64 --base_lr 1e-3 --seed $SEED --min_label $MIN_AGE --max_label $MAX_AGE


echo "-------------------------------------------------------------------------------------------------"
echo "regression CNN for label score"
CUDA_VISIBLE_DEVICES=1,0 python3 pretrain_CNN_regre.py --root_path $ROOT_PATH --data_path $DATA_PATH --CNN ResNet34_regre --epochs $FID_EPOCH_CNN --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --min_age $MIN_AGE --max_age $MAX_AGE


echo "-------------------------------------------------------------------------------------------------"
echo "classification CNN for diversity"
CUDA_VISIBLE_DEVICES=1,0 python3 pretrain_CNN_class.py --root_path $ROOT_PATH --data_path $DATA_PATH --CNN ResNet34_class --epochs $FID_EPOCH_CNN --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --min_age $MIN_AGE --max_age $MAX_AGE --valid_proport 0.1


N_CLASS=40
echo "-------------------------------------------------------------------------------------------------"
echo "cGAN $N_CLASS classes"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py --root_path $ROOT_PATH --data_path $DATA_PATH --GAN cGAN --cGAN_num_classes $N_CLASS --seed $SEED --min_age $MIN_AGE --max_age $MAX_AGE --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --nfake_per_label $NFAKE_PER_LABEL --visualize_fake_images --comp_FID --epoch_FID_CNN $FID_EPOCH_CNN --FID_radius $FID_RADIUS 2>&1 | tee output_cGAN_40class.txt

N_CLASS=60
echo "-------------------------------------------------------------------------------------------------"
echo "cGAN $N_CLASS classes"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py --root_path $ROOT_PATH --data_path $DATA_PATH --GAN cGAN --cGAN_num_classes $N_CLASS --seed $SEED --min_age $MIN_AGE --max_age $MAX_AGE --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --nfake_per_label $NFAKE_PER_LABEL --visualize_fake_images --comp_FID --epoch_FID_CNN $FID_EPOCH_CNN --FID_radius $FID_RADIUS 2>&1 | tee output_cGAN_60class.txt


RESUME_NITERS=0
echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN Hard"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py --root_path $ROOT_PATH --data_path $DATA_PATH --GAN CcGAN --seed $SEED --min_age $MIN_AGE --max_age $MAX_AGE --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA --threshold_type hard --kernel_sigma $SIGMA --kappa $KAPPA --dim_gan $DIM_GAN --niters_gan $NITERS --resume_niters_gan $RESUME_NITERS --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --nfake_per_label $NFAKE_PER_LABEL --visualize_fake_images --comp_FID --epoch_FID_CNN $FID_EPOCH_CNN --FID_radius $FID_RADIUS 2>&1 | tee output_hard.txt

RESUME_NITERS=0
echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN Soft"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py --root_path $ROOT_PATH --data_path $DATA_PATH --GAN CcGAN --seed $SEED --min_age $MIN_AGE --max_age $MAX_AGE --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA --threshold_type soft --kernel_sigma $SIGMA --kappa $KAPPA --dim_gan $DIM_GAN --niters_gan $NITERS --resume_niters_gan $RESUME_NITERS --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --nfake_per_label $NFAKE_PER_LABEL --visualize_fake_images --comp_FID --epoch_FID_CNN $FID_EPOCH_CNN --FID_radius $FID_RADIUS 2>&1 | tee output_soft.txt


echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN Hard Limit"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py --root_path $ROOT_PATH --data_path $DATA_PATH --GAN CcGAN --seed $SEED --min_age $MIN_AGE --max_age $MAX_AGE --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA --threshold_type hard --kernel_sigma 1e-30 --kappa 1e-30 --dim_gan $DIM_GAN --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --nfake_per_label $NFAKE_PER_LABEL --visualize_fake_images --comp_FID --epoch_FID_CNN $FID_EPOCH_CNN --FID_radius $FID_RADIUS 2>&1 | tee output_limit.txt
