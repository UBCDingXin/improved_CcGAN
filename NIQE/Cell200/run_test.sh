cd fake_data
unzip fake_images_CcGAN_hard_nsamp_200000.zip -d fake_images_CcGAN_hard_nsamp_200000/
unzip fake_images_CcGAN_limit_nsamp_200000.zip -d fake_images_CcGAN_limit_nsamp_200000/
unzip fake_images_CcGAN_soft_nsamp_200000.zip -d fake_images_CcGAN_soft_nsamp_200000/
unzip fake_images_cGAN_nclass_50_nsamp_200000.zip -d fake_images_cGAN_nclass_50_nsamp_200000/
unzip fake_images_cGAN_nclass_100_nsamp_200000.zip -d fake_images_cGAN_nclass_100_nsamp_200000/
unzip fake_images_improved_CcGAN_hard_nsamp_200000.zip -d fake_images_improved_CcGAN_hard_nsamp_200000/
unzip fake_images_improved_CcGAN_soft_nsamp_200000.zip -d fake_images_improved_CcGAN_soft_nsamp_200000/
cd ..

bash ./imgs_to_groups_fake.sh

matlab -nodisplay -nodesktop -r "run Intra_niqe_test_cell200.m"

cd fake_data
rm -rf fake_images*
