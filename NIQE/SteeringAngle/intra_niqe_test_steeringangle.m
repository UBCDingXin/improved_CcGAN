close all;
clear; clc


block_sz = 8;
dataset_name = 'steeringangle';  datadir_base = 'fake_data/'; train_type = 'all'; %('all', '10')


img_dir_all = {'fake_images_cGAN_nclass_30_nsamp_100000/', 'fake_images_cGAN_nclass_90_nsamp_100000/', 'fake_images_cGAN_nclass_150_nsamp_100000/', 'fake_images_cGAN_nclass_210_nsamp_100000/', 'fake_images_CcGAN_hard_nsamp_100000/', 'fake_images_CcGAN_soft_nsamp_100000/', 'fake_images_improved_CcGAN_hard_nsamp_100000/', 'fake_images_improved_CcGAN_soft_nsamp_100000/'};



K = length(img_dir_all); %number of img dirs
N = 1000; %num of centers
IQA = zeros(N, K);


tic;
for i = 1:1000

    % load model
    model_name = ['model_center_', num2str(i), '_', dataset_name, '_', train_type, '_', num2str(block_sz), 'x', num2str(block_sz), '.mat'];
    model_path = ['models/Intra_niqe/', model_name];
    load(model_path);

    IQA_i = zeros(1,K);

    parfor k = 1:K

        img_dir_base = img_dir_all{k};

        img_dir = [datadir_base, img_dir_base, 'centers/', num2str(i), '/'];
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
        IQA_i(k) = iq_niqe;
    end
    IQA(i, :) = IQA_i;

    toc
    for k = 1:K
      fprintf('center=%d, k=%d, NIQE=%.3f \n', i, k, IQA(i, k));
    end

end
toc
