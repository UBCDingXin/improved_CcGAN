#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First, run 'main.py' to dump several images to '/UTKFace/Output/saved_images' for each age (images for each age are stored in the same folder). 
Second, for each age only keep 3 images and run this script.
"""

import os
wd = '/home/xin/OneDrive/Working_directory/CcGAN/RC-49'

os.chdir(wd)
import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt



threshold_type = 'hard'
max_label=60
n_row = 10
n_col = 3

path_fake_images = wd + '/output/saved_images/CcGAN_{}'.format(threshold_type)
filename_fake_images = wd + '/output/saved_images/CcGAN_{}_fake_images_grid_{}x{}.png'.format(threshold_type, n_row, n_col)


#displayed_labels = (np.linspace(0.05, 0.95, n_row)*max_label).astype(np.int)
displayed_labels = np.array([3,9,15,21,26,32,39,45,51,57])


images_show = np.zeros((n_row*n_col, 3, 64, 64))
for i_row in range(n_row):
    curr_label = displayed_labels[i_row]
    curr_folder_i = path_fake_images + '/{}'.format(curr_label)
    curr_filenames = os.listdir(curr_folder_i)
    for j_col in range(n_col):
        curr_image_pil = Image.open(curr_folder_i+'/{}'.format(curr_filenames[j_col]))
        curr_image = (np.array(curr_image_pil)).transpose((2,0,1))
        curr_image = curr_image/255.0
        images_show[i_row*n_col+j_col,:,:,:] = curr_image
images_show = torch.from_numpy(images_show)
save_image(images_show.data, filename_fake_images, nrow=n_col, normalize=False)




'''

Draw FID versus eval center (number of samples)

'''

filename1 = wd + "/cGAN_nclass_60_fids_nrealimgs_centers.npz"
filename2 = wd + "/cGAN_nclass_90_fids_nrealimgs_centers.npz"
filename3 = wd + "/CcGAN_hard_fids_nrealimgs_centers.npz"
filename4 = wd + "/CcGAN_soft_fids_nrealimgs_centers.npz"

data1 = np.load(filename1)
data2 = np.load(filename2)
data3 = np.load(filename3)
data4 = np.load(filename4)

centers = data1['centers']
nrealimgs = data1['nrealimgs']
y_max = max(np.max(data1['fids']), np.max(data2['fids']), np.max(data3['fids']), np.max(data4['fids']))


filename_fid_versus_center = wd + "/comparison_fid_versus_center.pdf"


fig, ax = plt.subplots()
ax.plot(centers, data1['fids'], 'k--', label='cGAN (60 classes)')
ax.plot(centers, data2['fids'], 'k:', label='cGAN (90 classes)')
ax.plot(centers, data3['fids'], 'b-', label='CcGAN (HVDL)')
ax.plot(centers, data4['fids'], 'g-', label='CcGAN (SVDL)')
#ax.plot(centers, nrealimgs, 'r-.', label='\# images per label')
legend = ax.legend(loc='upper left', shadow=True, fontsize=10)
plt.xlabel("Angle of yaw rotation")
plt.ylabel("FID")
plt.show()
fig.savefig(filename_fid_versus_center, bbox_inches='tight')













