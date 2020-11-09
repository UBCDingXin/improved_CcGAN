cd fake_data
unzip fake_images_CcGAN_hard_nsamp_60000.zip -d fake_images_CcGAN_hard_nsamp_60000/
unzip fake_images_CcGAN_limit_nsamp_60000.zip -d fake_images_CcGAN_limit_nsamp_60000/
unzip fake_images_CcGAN_soft_nsamp_60000.zip -d fake_images_CcGAN_soft_nsamp_60000/
unzip fake_images_cGAN_nclass_40_nsamp_60000.zip -d fake_images_cGAN_nclass_40_nsamp_60000/
unzip fake_images_cGAN_nclass_60_nsamp_60000.zip -d fake_images_cGAN_nclass_60_nsamp_60000/
unzip fake_images_improved_CcGAN_hard_nsamp_60000.zip -d fake_images_improved_CcGAN_hard_nsamp_60000/
unzip fake_images_improved_CcGAN_soft_nsamp_60000.zip -d fake_images_improved_CcGAN_soft_nsamp_60000/
cd ..

bash ./imgs_to_groups_fake.sh

matlab -nodisplay -nodesktop -r "run Intra_niqe_test_utkface.m"

cd fake_data
rm -rf fake_images*
