% % Test on an Intra-NIQE model on fake UTKFace
% % Aug.18.2020

close all;
clear; clc

block_sz = 8;
dataset_name = 'utkface';  datadir_base = 'fake_data/'; train_type = 'all'; %('all', '10')



N = 60;
IQA = zeros(N, 7);

tic;

for age = 1: N


    model_name = ['model_age_', num2str(age), '_', dataset_name, '_', train_type, '_', num2str(block_sz), 'x', num2str(block_sz), '.mat'];
    model_path = ['models/Intra_niqe/', model_name];

    % imds = imageDatastore(datadir,'FileExtensions',{'.jpg'});
    % model = fitniqe(imds,'BlockSize',[block_sz block_sz]); save(model_path, 'model');
    load(model_path);

    % delete(gcp('nocreate'));
    IQA_age = zeros(1,7);
    parfor k = 1:7
        if k==1
            img_dir_base = 'fake_images_cGAN_nclass_40_nsamp_60000/';
        elseif k==2
            img_dir_base = 'fake_images_cGAN_nclass_60_nsamp_60000/';
        elseif k==3
            img_dir_base = 'fake_images_CcGAN_hard_nsamp_60000/';
        elseif k==4
            img_dir_base = 'fake_images_CcGAN_soft_nsamp_60000/';
        elseif k==5
            img_dir_base = 'fake_images_CcGAN_limit_nsamp_60000/';
        elseif k==6
            img_dir_base = 'fake_images_improved_CcGAN_hard_nsamp_60000/';
        elseif k==7
            img_dir_base = 'fake_images_improved_CcGAN_soft_nsamp_60000/';
        end

        img_dir = [datadir_base, img_dir_base, 'ages/', num2str(age), '/'];
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
        % fprintf('age=%d, k=%d, NIQE=%.3f \n', age, k, iq_niqe);
        IQA_age(k) = iq_niqe;
    end
    IQA(age, :) = IQA_age;

    toc
    for k = 1:7
      fprintf('age=%d, k=%d, NIQE=%.3f \n', age, k, IQA(age, k));
    end

end
toc

csvwrite('results/intra_niqe_utkface.csv', IQA);

stats_utkface = zeros(7,2);
for i = 1: 7
    stats_utkface(i,1) = mean(IQA(:,i));
    stats_utkface(i,2) = std(IQA(:,i));
end

disp(stats_utkface)
quit()
