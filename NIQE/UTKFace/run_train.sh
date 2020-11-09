cd real_data
unzip UTKFace.zip -d UTKFace
cd ..

matlab -nodisplay -nodesktop -r "run Intra_niqe_train_utkface.m"


cd real_data
rm -rf UTKFace
