% % Test on an Intra-NIQE model on fake UTKFace
% % Aug.18.2020

close all;
clear; clc

block_sz = 16;
dataset_name = 'rc49';  datadir_base = 'real_data/RC-49_images_all_0_90/real_images_by_angles/'; train_type = 'all'; %('all', '10')


angles = 0.1: 0.1: 89.9;
parfor i = 1: length(angles)
    angle = angles(i)

    fprintf('i=%d, angle=%.1f \n', i, angle);

    datadir = [fullfile(datadir_base, num2str(angle,'%.1f')), '/'];
    model_name = ['model_angle_', num2str(angle,'%.1f'), '_', dataset_name, '_', train_type, '_', num2str(block_sz), 'x', num2str(block_sz), '.mat'];
    model_path = ['models/', model_name];

    datadir = [datadir_base, num2str(angle,'%.1f'), '/']
    imds = imageDatastore(datadir,'FileExtensions',{'.png'});
    model = fitniqe(imds,'BlockSize',[block_sz block_sz], 'SharpnessThreshold', 0.1);
    % save(model_path, 'model');
    parsave(model_path, model);
end

quit()

function parsave(model_path, model)
  save(model_path, 'model')
end
