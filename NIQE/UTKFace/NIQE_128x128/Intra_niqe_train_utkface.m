% % Test on an Intra-NIQE model on fake UTKFace
% % Aug.18.2020

close all;
clear; clc

block_sz = 16;
dataset_name = 'utkface';  datadir_base = 'real_data/real_images_by_ages/'; train_type = 'all'; %('all', '10')

N = 60;

parfor age = 1: N
    model_name = ['model_age_', num2str(age), '_', dataset_name, '_', train_type, '_', num2str(block_sz), 'x', num2str(block_sz), '.mat'];
    model_path = ['models/', model_name];

    datadir = [datadir_base, num2str(age), '/']
    imds = imageDatastore(datadir,'FileExtensions',{'.png'});
    model = fitniqe(imds,'BlockSize',[block_sz block_sz], 'SharpnessThreshold', 0.1);
    % save(model_path, 'model');
    parsave(model_path, model);
end

quit()

function parsave(model_path, model)
  save(model_path, 'model')
end
