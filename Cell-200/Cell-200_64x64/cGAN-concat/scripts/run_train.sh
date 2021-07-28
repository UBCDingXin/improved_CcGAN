ROOT_PATH="./Cell-200/Cell-200_64x64/cGAN-concat"
DATA_PATH="./datasets/Cell200"


SEED=2020
NITERS=5000
BATCH_SIZE_D=32
BATCH_SIZE_G=512
START_COUNT=1
END_COUNT=200
STEPSIZE_COUNT=2
N_IMGS_PER_CELLCOUNT=10
LR_G=1e-4
LR_D=1e-4
NFAKE_PER_LABEL=1000
FID_RADIUS=0
DIM_GAN=128


python main.py --root_path $ROOT_PATH --data_path $DATA_PATH --GAN cGAN-concat --start_count $START_COUNT --end_count $END_COUNT --stepsize_count $STEPSIZE_COUNT --num_imgs_per_count $N_IMGS_PER_CELLCOUNT --transform --dim_gan $DIM_GAN --niters_gan $NITERS --resume_niters_gan 0 --save_niters_freq 2000 --lr_g_gan $LR_G --lr_d_gan $LR_D --batch_size_disc $BATCH_SIZE_D --batch_size_gene $BATCH_SIZE_G --seed $SEED --visualize_fake_images --comp_FID --nfake_per_label $NFAKE_PER_LABEL --samp_batch_size 500 --FID_radius $FID_RADIUS --dump_fake_for_NIQE