@REM 不需要
@REM python imgs_to_groups_real.py --imgs_dir "G:\\Local_WD\\CcGAN_TPAMI_NIQE\\SteeringAngle\\NIQE_128x128\\real_data\\real_data_steeringangle_-80_80"  --center_file "G:\\Local_WD\\CcGAN_TPAMI_NIQE\\SteeringAngle\\NIQE_128x128\\fake_data\\steering_angle_centers_loc_for_NIQE.txt" --out_dir_base "G:\\Local_WD\\CcGAN_TPAMI_NIQE\\SteeringAngle\\NIQE_128x128\\real_data"

matlab -nodisplay -nodesktop -r "run niqe_train_steeringangle.m" %*
