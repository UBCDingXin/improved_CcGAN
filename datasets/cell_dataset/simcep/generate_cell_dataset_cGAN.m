% This code generates cell dataset with a given cell number range to test
% the effectiveness of the ccGAN. 
% We use the tool from: http://www.cs.tut.fi/sgn/csb/simcep/tool.html
% Input: 
%       min_cell_num: minimum cell number, e.g.,1 ; 
%       max_cell_num: maximum cell number, e.g., 300;
%       cell_num_each: number of images for given cell number, e.g.,1000;
%       dataset_root: base path to save cell image dataset, '../cell_dataset'. 
% Date: Feb.27,2020

dataset_root = '../cell_dataset/';

min_cell_num = 1;
max_cell_num = 300;
cell_num_each = 1000;

GenDataset(min_cell_num, max_cell_num, cell_num_each, dataset_root); 


function GenDataset(min_cell_num, max_cell_num, cell_num_each, dataset_root)
%% This code generates a batch of cell images with given range of cell number.
% parameters given. 

for cell_num = min_cell_num: max_cell_num
    disp(cell_num); 
    folder_name = [dataset_root, int2str(cell_num), '/'];
    mkdir(folder_name); 
    
    for img_idx = 1:cell_num_each
        [image,~,~] = simcep(cell_num);
        img_name = [int2str(cell_num), '_', int2str(img_idx), '.png'];
        img_path = [folder_name, img_name];
        imwrite(image, img_path); 
    end
    
end
end

