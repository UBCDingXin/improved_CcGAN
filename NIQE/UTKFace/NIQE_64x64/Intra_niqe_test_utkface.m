% % Test on an Intra-NIQE model on fake UTKFace
% % Aug.18.2020

close all;
clear; clc

block_sz = 8;
dataset_name = 'utkface';  datadir_base = 'fake_data/fake_images_by_ages/'; train_type = 'all'; %('all', '10')


ages = 1:60;
N = length(ages);
intra_niqe = zeros(N,1);


tic;

for i = 1: N

    age = ages(i);

    model_name = ['model_age_', num2str(age), '_', dataset_name, '_', train_type, '_', num2str(block_sz), 'x', num2str(block_sz), '.mat'];
    model_path = ['models/', model_name];

    load(model_path);

    img_dir = [datadir_base, num2str(age), '/'];
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
    fprintf('age=%d, nfake=%d, NIQE=%.3f \n', age, length(imgs), intra_niqe(i));
end
toc

avg_niqe=mean(intra_niqe);
std_niqe=std(intra_niqe);

fprintf('NIQE, mean(std): %.3f (%.3f) \n', avg_niqe, std_niqe);

csvwrite('results/intra_niqe_utkface.csv', intra_niqe);

quit()
