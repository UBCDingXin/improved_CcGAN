% % Test on an Intra-NIQE model on fake UTKFace
% % Aug.18.2020

close all;
clear; clc

block_sz = 8;
dataset_name = 'rc49';  datadir_base = 'fake_data/'; train_type = 'all'; %('all', '10')



N = 899;
IQA = zeros(N, 9);


tic;

for angle = 0.1: 0.1: 89.9

    model_name = ['model_angle_', num2str(angle,'%.1f'), '_', dataset_name, '_', train_type, '_', num2str(block_sz), 'x', num2str(block_sz), '.mat'];
    model_path = ['models/Intra_niqe/', model_name];
    load(model_path);

    % delete(gcp('nocreate'));
    IQA_angle = zeros(1,9);
    parfor k = 1:9
        if k==1
            img_dir_base = 'fake_images_cGAN_nclass_60_nsamp_179800/';
        elseif k==2
            img_dir_base = 'fake_images_cGAN_nclass_90_nsamp_179800/';
        elseif k==3
            img_dir_base = 'fake_images_cGAN_nclass_150_nsamp_179800/';
        elseif k==4
            img_dir_base = 'fake_images_cGAN_nclass_210_nsamp_179800/';
        elseif k==5
            img_dir_base = 'fake_images_CcGAN_hard_nsamp_179800/';
        elseif k==6
            img_dir_base = 'fake_images_CcGAN_soft_nsamp_179800/';
        elseif k==7
            img_dir_base = 'fake_images_CcGAN_limit_nsamp_179800/';
        elseif k==8
            img_dir_base = 'fake_images_improved_CcGAN_hard_nsamp_179800/';
        elseif k==9
            img_dir_base = 'fake_images_improved_CcGAN_soft_nsamp_179800/';
        end


        img_dir = [datadir_base, img_dir_base, 'angles/', num2str(angle,'%.1f'), '/'];
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
        IQA_angle(k) = iq_niqe;
    end
    IQA(round(angle*10), :) = IQA_angle;

    toc
    for k = 1:9
      fprintf('angle=%.1f, k=%d, NIQE=%.3f \n', angle, k, IQA(round(angle*10), k));
    end

end
toc

csvwrite('results/intra_niqe_rc49.csv', IQA);

stats_rc49 = zeros(9,2);
for i = 1: 9
    stats_rc49(i,1) = mean(IQA(:,i));
    stats_rc49(i,2) = std(IQA(:,i));
end

disp(stats_rc49)
quit()
