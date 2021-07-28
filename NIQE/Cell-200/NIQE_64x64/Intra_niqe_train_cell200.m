% % Test on an Intra-NIQE model on fake UTKFace
% % Aug.18.2020

close all;
clear; clc

block_sz = 8;
dataset_name = 'cell200';  datadir_base = 'real_data/Cell200/counts/'; train_type = 'all'; %('all', '10')

N = 200;

parfor count = 1: N
    model_name = ['model_count_', num2str(count), '_', dataset_name, '_', train_type, '_', num2str(block_sz), 'x', num2str(block_sz), '.mat'];
    model_path = ['models/', model_name];

    datadir = [datadir_base, num2str(count), '/']
    imds = imageDatastore(datadir,'FileExtensions',{'.png'});
    model = fitniqe(imds,'BlockSize',[block_sz block_sz], 'SharpnessThreshold', 0.1);
    % save(model_path, 'model');
    parsave(model_path, model);
end

quit()

function parsave(model_path, model)
  save(model_path, 'model')
end
