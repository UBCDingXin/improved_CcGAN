% % Test on an Intra-NIQE model on fake UTKFace
% % Aug.18.2020

close all;
clear; clc

block_sz = 8;
dataset_name = 'cell200';  datadir_base = 'fake_data/'; train_type = 'all'; %('all', '10')



N = 200;
IQA = zeros(N, 6);

tic;

for count = 1: N


    model_name = ['model_count_', num2str(count), '_', dataset_name, '_', train_type, '_', num2str(block_sz), 'x', num2str(block_sz), '.mat'];
    model_path = ['models/Intra_niqe/', model_name];

    % imds = imageDatastore(datadir,'FileExtensions',{'.jpg'});
    % model = fitniqe(imds,'BlockSize',[block_sz block_sz]); save(model_path, 'model');
    load(model_path);

    % delete(gcp('nocreate'));
    IQA_count = zeros(1,6);
    parfor k = 1:6
        if k==1
            img_dir_base = 'fake_images_cGAN_nclass_50_nsamp_200000/';
        elseif k==2
            img_dir_base = 'fake_images_cGAN_nclass_100_nsamp_200000/';
        elseif k==3
            img_dir_base = 'fake_images_CcGAN_hard_nsamp_200000/';
        elseif k==4
            img_dir_base = 'fake_images_CcGAN_soft_nsamp_200000/';
        elseif k==5
            img_dir_base = 'fake_images_improved_CcGAN_hard_nsamp_200000/';
        elseif k==6
            img_dir_base = 'fake_images_improved_CcGAN_soft_nsamp_200000/';
        end

        img_dir = [datadir_base, img_dir_base, 'counts/', num2str(count), '/'];
        imgs = dir(img_dir);
        imgs = imgs(3:end);

        iq_niqe = 0;
        for img_idx = 1: length(imgs)
            img_name = imgs(img_idx).name;
            img = imread(fullfile(img_dir, img_name));
            tmp = niqe(img, model); %compute NIQE by pre-trained model
            iq_niqe = iq_niqe + tmp;
        end

        iq_niqe = iq_niqe / length(imgs);
        IQA_count(k) = iq_niqe;
    end
    IQA(count, :) = IQA_count;

    toc
    for k = 1:6
      fprintf('count=%d, k=%d, NIQE=%.3f \n', count, k, IQA(count, k));
    end

end
toc

csvwrite('results/intra_niqe_cell200.csv', IQA);

stats_cell200 = zeros(6,2);
for i = 1: 6
    stats_cell200(i,1) = mean(IQA(:,i));
    stats_cell200(i,2) = std(IQA(:,i));
end

disp(stats_cell200)
quit()
