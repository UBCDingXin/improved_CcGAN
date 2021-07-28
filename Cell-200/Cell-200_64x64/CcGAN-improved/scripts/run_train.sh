
ROOT_PATH="./Cell-200/Cell-200_64x64/CcGAN-improved"
DATA_PATH="./datasets/Cell200"


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

DIM_CcGAN=256
DIM_cGAN=128
DIM_EMBED=128
LOSS_TYPE='vanilla'


echo "-------------------------------------------------------------------------------------------------"
echo "AE"
python pretrain_AE.py --root_path $ROOT_PATH --data_path $DATA_PATH --dim_bottleneck 512 --epochs 50 --resume_epoch 0 --save_ckpt_freq 25 --batch_size_train 256 --batch_size_valid 128 --base_lr 1e-4 --lr_decay_epochs 25 --lr_decay_factor 0.5 --weight_dacay 1e-5 --seed $SEED --min_label $START_COUNT --max_label $END_COUNT

echo "-------------------------------------------------------------------------------------------------"
echo "regression CNN for evaluation"
python pretrain_CNN_regre.py --root_path $ROOT_PATH --data_path $DATA_PATH --start_count $START_COUNT --end_count $END_COUNT --CNN ResNet34_regre --epochs 200 --batch_size_train 256 --batch_size_valid 64 --base_lr 0.01 --seed $SEED --transform


N_CLASS=100
echo "-------------------------------------------------------------------------------------------------"
echo "cGAN $N_CLASS"
python main.py --root_path $ROOT_PATH --data_path $DATA_PATH --GAN cGAN --cGAN_num_classes $N_CLASS --start_count $START_COUNT --end_count $END_COUNT --stepsize_count $STEPSIZE_COUNT --num_imgs_per_count $N_IMGS_PER_CELLCOUNT --transform --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 --dim_gan $DIM_cGAN --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --seed $SEED --visualize_fake_images --comp_FID --nfake_per_label $NFAKE_PER_LABEL --samp_batch_size 1000 --FID_radius $FID_RADIUS --dump_fake_for_NIQE 2>&1 | tee output_cGAN100.txt


RESUME_NITER=0
echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN: Hard"
python main.py --root_path $ROOT_PATH --data_path $DATA_PATH --GAN CcGAN --start_count $START_COUNT --end_count $END_COUNT --stepsize_count $STEPSIZE_COUNT --num_imgs_per_count $N_IMGS_PER_CELLCOUNT --transform --kernel_sigma $SIGMA --threshold_type hard --kappa $KAPPA --dim_gan $DIM_CcGAN --loss_type_gan $LOSS_TYPE --niters_gan $NITERS --resume_niters_gan $RESUME_NITER --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --seed $SEED --visualize_fake_images --dim_embed $DIM_EMBED --comp_FID --nfake_per_label $NFAKE_PER_LABEL --samp_batch_size 1000 --FID_radius $FID_RADIUS --dump_fake_for_NIQE 2>&1 | tee output_hard.txt


RESUME_NITER=0
echo "-------------------------------------------------------------------------------------------------"
echo "CcGAN: Soft"
python main.py --root_path $ROOT_PATH --data_path $DATA_PATH --GAN CcGAN --start_count $START_COUNT --end_count $END_COUNT --stepsize_count $STEPSIZE_COUNT --num_imgs_per_count $N_IMGS_PER_CELLCOUNT --transform --kernel_sigma $SIGMA --threshold_type soft --kappa $KAPPA --dim_gan $DIM_CcGAN --loss_type_gan $LOSS_TYPE --niters_gan $NITERS --resume_niters_gan $RESUME_NITER --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --seed $SEED --visualize_fake_images --dim_embed $DIM_EMBED --comp_FID --nfake_per_label $NFAKE_PER_LABEL --samp_batch_size 1000 --FID_radius $FID_RADIUS --dump_fake_for_NIQE 2>&1 | tee output_soft.txt
