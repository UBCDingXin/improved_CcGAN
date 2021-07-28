% % Test on an Intra-NIQE model on fake UTKFace
% % Aug.18.2020

close all;
clear; clc

block_sz = 16;
dataset_name = 'rc49';  datadir_base = 'fake_data/fake_images_by_angles/'; train_type = 'all'; %('all', '10')


angles = 0.1: 0.1: 89.9;
N = length(angles);
intra_niqe = zeros(N,1);

tic;

for i = 1: N

    angle = angles(i);

    model_name = ['model_angle_', num2str(angle,'%.1f'), '_', dataset_name, '_', train_type, '_', num2str(block_sz), 'x', num2str(block_sz), '.mat'];
    model_path = ['models/', model_name];
    load(model_path);

    img_dir = [datadir_base, num2str(angle,'%.1f'), '/'];
    imgs = dir(img_dir);
    imgs = imgs(3:end);

    niqe_of_each_img = zeros(length(imgs),1);
    parfor img_idx = 1: length(imgs)
        img_name = imgs(img_idx).name;
        img = imread(fullfile(img_dir, img_name));
        niqe_of_each_img(img_idx) = niqe(img, model); %compute NIQE by pre-trained model
    end
    intra_niqe(i) = mean(niqe_of_each_img);

    toc
    fprintf('angle=%.1f, NIQE=%.3f \n', angle, intra_niqe(i));
end
toc

avg_niqe=mean(intra_niqe, 'omitnan');
std_niqe=std(intra_niqe, 'omitnan');

fprintf('NIQE, mean(std): %.3f (%.3f) \n', avg_niqe, std_niqe);

csvwrite('results/intra_niqe_rc49.csv', intra_niqe);

% disp([avg_niqe, std_niqe])

quit()
