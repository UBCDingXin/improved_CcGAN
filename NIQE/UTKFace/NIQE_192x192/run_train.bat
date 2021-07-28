
python imgs_to_groups_real.py --imgs_dir .\real_data\UTKFace_images_all --out_dir_base .\real_data

matlab -noFigureWindows -nodesktop -logfile output.txt -r "run Intra_niqe_train_utkface.m" %*
