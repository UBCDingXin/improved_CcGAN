% % Test on an Intra-NIQE model on fake UTKFace
% % Aug.18.2020

close all;
clear; clc

block_sz = 8;
dataset_name = 'cell200';  datadir_base = 'fake_data/counts/'; train_type = 'all'; %('all', '10')

N = 200;
counts = 1:N;
N = length(counts);
intra_niqe = zeros(N,1);

tic;

for i = 1: N

    count = counts(i);

    model_name = ['model_count_', num2str(count), '_', dataset_name, '_', train_type, '_', num2str(block_sz), 'x', num2str(block_sz), '.mat'];
    model_path = ['models/', model_name];
    load(model_path);

    img_dir = [datadir_base, num2str(count,'%d'), '/'];
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
    fprintf('count=%d, NIQE=%.3f \n', count, intra_niqe(i));
end
toc

avg_niqe=mean(intra_niqe, 'omitnan');
std_niqe=std(intra_niqe, 'omitnan');

fprintf('NIQE, mean(std): %.3f (%.3f) \n', avg_niqe, std_niqe);

csvwrite('results/intra_niqe_rc49.csv', intra_niqe);

% disp([avg_niqe, std_niqe])

quit()