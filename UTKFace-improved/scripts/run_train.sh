ROOT_PATH="./improved_CcGAN/UTKFace-improved"
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
LR_G=1e-4
LR_D=1e-4
NFAKE_PER_LABEL=1000
FID_EPOCH_CNN=200
FID_RADIUS=0

DIM_CcGAN=256
DIM_cGAN=128
DIM_EMBED=128
LOSS_TYPE='vanilla'


echo "-------------------------------------------------------------------------------------------------"
echo "regression CNN for Label Score"
CUDA_VISIBLE_DEVICES=1,0 python3 pretrain_CNN_regre.py --root_path $ROOT_PATH --data_path $DATA_PATH --CNN ResNet34_regre --epochs $FID_EPOCH_CNN --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --min_age $MIN_AGE --max_age $MAX_AGE

echo "-------------------------------------------------------------------------------------------------"
echo "classification CNN for diversity"
CUDA_VISIBLE_DEVICES=1,0 python3 pretrain_CNN_class.py --root_path $ROOT_PATH --data_path $DATA_PATH --CNN ResNet34_class --epochs $FID_EPOCH_CNN --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --min_age $MIN_AGE --max_age $MAX_AGE --valid_proport 0.1

N_CLASS=40
echo "-------------------------------------------------------------------------------------------------"
echo "cGAN $N_CLASS classes"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py --root_path $ROOT_PATH --data_path $DATA_PATH --GAN cGAN --cGAN_num_classes $N_CLASS --dim_gan $DIM_cGAN --loss_type_gan $LOSS_TYPE --seed $SEED --min_age $MIN_AGE --max_age $MAX_AGE --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --nfake_per_label $NFAKE_PER_LABEL --visualize_fake_images --comp_FID --epoch_FID_CNN $FID_EPOCH_CNN --FID_radius $FID_RADIUS 2>&1 | tee output_cGAN40.txt

N_CLASS=60
echo "-------------------------------------------------------------------------------------------------"
echo "cGAN $N_CLASS classes"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py --root_path $ROOT_PATH --data_path $DATA_PATH --GAN cGAN --cGAN_num_classes $N_CLASS --dim_gan $DIM_cGAN --loss_type_gan $LOSS_TYPE --seed $SEED --min_age $MIN_AGE --max_age $MAX_AGE --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --nfake_per_label $NFAKE_PER_LABEL --visualize_fake_images --comp_FID --epoch_FID_CNN $FID_EPOCH_CNN --FID_radius $FID_RADIUS 2>&1 | tee output_cGAN40.txt


echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN Hard"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py --root_path $ROOT_PATH --data_path $DATA_PATH --GAN CcGAN --dim_gan $DIM_CcGAN --loss_type_gan $LOSS_TYPE --seed $SEED --min_age $MIN_AGE --max_age $MAX_AGE --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA --threshold_type hard --kernel_sigma $SIGMA --kappa $KAPPA --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --nfake_per_label $NFAKE_PER_LABEL --visualize_fake_images --dim_embed $DIM_EMBED --comp_FID --epoch_FID_CNN $FID_EPOCH_CNN --FID_radius $FID_RADIUS 2>&1 | tee output_hard.txt


echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN Soft"
CUDA_VISIBLE_DEVICES=1,0 python3 main.py --root_path $ROOT_PATH --data_path $DATA_PATH --GAN CcGAN --dim_gan $DIM_CcGAN --loss_type_gan $LOSS_TYPE --seed $SEED --min_age $MIN_AGE --max_age $MAX_AGE --max_num_img_per_label $MAX_N_IMG_PER_LABEL --max_num_img_per_label_after_replica $MAX_N_IMG_PER_LABEL_AFTER_REPLICA --threshold_type soft --kernel_sigma $SIGMA --kappa $KAPPA --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --nfake_per_label $NFAKE_PER_LABEL --visualize_fake_images --dim_embed $DIM_EMBED --comp_FID --epoch_FID_CNN $FID_EPOCH_CNN --FID_radius $FID_RADIUS 2>&1 | tee output_soft.txt
