python imgs_to_groups_fake.py --imgs_dir .\fake_data\fake_images --out_dir_base .\fake_data

mkdir results

matlab -noFigureWindows -nodesktop -logfile output.txt -r "run Intra_niqe_test_utkface.m" %*