
cd real_data
unzip real_data_steeringangle_-80_80.zip -d real_data_steeringangle_-80_80/
cd ..


python3 imgs_to_groups_real.py --imgs_dir 'real_data/real_data_steeringangle_-80_80/real_data_steeringangle_-80_80/' --center_file './fake_data/steering_angle_centers_loc_for_NIQE.txt'

matlab -nodisplay -nodesktop -r "run intra_niqe_train_steeringangle.m"

cd real_data
rm -rf real_data_steeringangle_-80_80
