cd real_data
unzip RC-49_images_all_0_90.zip -d RC-49_images_all_0_90/
cd ..

bash ./imgs_to_groups_real.sh

matlab -nodisplay -nodesktop -r "run Intra_niqe_train_rc49.m"

cd real_data
rm -rf RC-49_images_all_0_90
