# cd fake_data
# unzip fake_images_CcGAN_hard_nsamp_100000.zip -d fake_images_CcGAN_hard_nsamp_100000/
# unzip fake_images_CcGAN_soft_nsamp_100000.zip -d fake_images_CcGAN_soft_nsamp_100000/
# unzip fake_images_cGAN_nclass_30_nsamp_100000.zip -d fake_images_cGAN_nclass_30_nsamp_100000/
# unzip fake_images_cGAN_nclass_90_nsamp_100000.zip -d fake_images_cGAN_nclass_90_nsamp_100000/
# unzip fake_images_cGAN_nclass_150_nsamp_100000.zip -d fake_images_cGAN_nclass_150_nsamp_100000/
# unzip fake_images_cGAN_nclass_210_nsamp_100000.zip -d fake_images_cGAN_nclass_210_nsamp_100000/
# unzip fake_images_improved_CcGAN_hard_nsamp_100000.zip -d fake_images_improved_CcGAN_hard_nsamp_100000/
# unzip fake_images_improved_CcGAN_soft_nsamp_100000.zip -d fake_images_improved_CcGAN_soft_nsamp_100000/
# cd ..
#
# bash ./imgs_to_groups_fake.sh

matlab -nodisplay -nodesktop -r "run intra_niqe_test_steeringangle.m"

cd fake_data
rm -rf fake_images*
