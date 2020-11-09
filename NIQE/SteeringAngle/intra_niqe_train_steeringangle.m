close all;
clear; clc

block_sz = 8;
dataset_name = 'steeringangle';  datadir_base = 'real_data/real_data_steeringangle_-80_80/centers/'; train_type = 'all'; %('all', '10')


N = 1000;

parfor center = 1: N

    fprintf('center=%d \n', center);

    model_name = ['model_center_', num2str(center), '_', dataset_name, '_', train_type, '_', num2str(block_sz), 'x', num2str(block_sz), '.mat'];
    model_path = ['models/Intra_niqe/', model_name];

    datadir = [datadir_base, num2str(center), '/']
    imds = imageDatastore(datadir,'FileExtensions',{'.png'});
    model = fitniqe(imds,'BlockSize',[block_sz block_sz], 'SharpnessThreshold', 0.1);
    parsave(model_path, model);

end


quit()

function parsave(model_path, model)
  save(model_path, 'model')
end
