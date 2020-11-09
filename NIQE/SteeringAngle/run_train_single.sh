
cd real_data
unzip real_data_steeringangle_-80_80.zip
cd ..

# train NIQE model
matlab -nodisplay -nodesktop -r "run niqe_train_steeringangle.m"

cd real_data
rm -rf real_data_steeringangle_-80_80
