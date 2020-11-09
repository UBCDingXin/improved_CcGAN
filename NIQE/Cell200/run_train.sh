cd real_data
unzip Cell200.zip -d Cell200
cd ..

bash ./imgs_to_groups_real.sh

matlab -nodisplay -nodesktop -r "run Intra_niqe_train_cell200.m"


cd real_data
rm -rf Cell200
