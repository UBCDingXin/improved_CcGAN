close all;
clear; clc

block_sz = 16;
dataset_name = 'steeringangle';  datadir_base = 'real_data/'; train_type = 'all'; %('all', '10')


% % model training on whole dataset
datadir = fullfile(datadir_base, 'real_data_steeringangle_-80_80/');
model_name = ['model_whole_', dataset_name, '_', train_type, '_', num2str(block_sz), 'x', num2str(block_sz), '.mat'];
model_path = ['models/', model_name];
if ~exist('models/', 'dir')
   mkdir('models/')
end
imds = imageDatastore(datadir,'FileExtensions',{'.png'});
model = fitniqe(imds,'BlockSize',[block_sz block_sz], 'SharpnessThreshold', 0.1);
save(model_path, 'model');

quit()
