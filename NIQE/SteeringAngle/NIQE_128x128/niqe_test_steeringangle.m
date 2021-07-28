close all;
clear; clc


block_sz = 16;
dataset_name = 'steeringangle';  datadir_base = 'fake_data/'; train_type = 'all'; %('all', '10')

N = 1000; %num of centers
IQA = zeros(N, 1);


% load model
model_name = ['model_whole_', dataset_name, '_', train_type, '_', num2str(block_sz), 'x', num2str(block_sz), '.mat'];
model_path = ['models/', model_name];
load(model_path);

tic;
parfor i = 1:N

    img_dir = [datadir_base, 'centers/', num2str(i), '/'];
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
    IQA(i) = iq_niqe;

    toc
    fprintf('center=%d, NIQE=%.3f \n', i, IQA(i));

end
toc


avg_niqe=mean(IQA);
std_niqe=std(IQA);

fprintf('NIQE, mean(std): %.3f (%.3f) \n', avg_niqe, std_niqe);

csvwrite('results/intra_niqe_steering_angle.csv', IQA);

quit()