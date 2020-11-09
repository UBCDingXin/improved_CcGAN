import os
from tqdm import tqdm

path_to_models = '/home/xin/OneDrive/Working_directory/CcGAN/NIQE/RC49/models/Intra_niqe'
os.chdir(path_to_models)


filenames_all = os.listdir(path_to_models)
N_all = len(filenames_all)

# for i in tqdm(range(N_all)):
#     filename_i = filenames_all[i]
#     angle_str = filename_i.split('_')[2]
#     angle = float(angle_str)
#     if len(angle_str.split('.'))==1:
#         new_filename_split_i = filename_i.split('_')
#         new_filename_split_i[2] = str(angle)
#         new_filename_i = '_'.join(new_filename_split_i)
#         os.rename(filename_i, new_filename_i)


for i in tqdm(range(N_all)):
    filename_i = filenames_all[i]
    if filename_i.split('_')[-1].split('.')[0] in ['10x10', '12x12','14x14','32x32']:
        os.remove(filename_i)
