#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First, run 'main.py' to dump several images to './output/saved_images' for each age (images for each age are stored in the same folder). 
Second, for each age only keep 3 images and run this script.
"""

import os
wd = '/home/xin/OneDrive/Working_directory/CcGAN/SteeringAngle-improved'

os.chdir(wd)
import numpy as np
from PIL import Image
import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib as mpl



# load evaluation data
filename1 = wd + "/cGAN_nclass_30_fid_ls_entropy_over_centers.npz"
filename2 = wd + "/cGAN_nclass_90_fid_ls_entropy_over_centers.npz"
filename3 = wd + "/cGAN_nclass_150_fid_ls_entropy_over_centers.npz"
filename4 = wd + "/CcGAN_hard_fid_ls_entropy_over_centers.npz"
filename5 = wd + "/CcGAN_soft_fid_ls_entropy_over_centers.npz"
filename6 = wd + "/CcGAN-improved_hard_fid_ls_entropy_over_centers.npz"
filename7 = wd + "/CcGAN-improved_soft_fid_ls_entropy_over_centers.npz"

data1 = np.load(filename1)
data2 = np.load(filename2)
data3 = np.load(filename3)
data4 = np.load(filename4)
data5 = np.load(filename5)
data6 = np.load(filename6)
data7 = np.load(filename7)

'''

Draw FID versus eval center (number of samples)

'''
centers = data1['centers']
nrealimgs = data1['nrealimgs']

## plot all
filename = wd + "/comparison_fid_versus_center_all.pdf"
y_max = max(np.max(data1['fids']), np.max(data2['fids']), np.max(data3['fids']), np.max(data4['fids']), np.max(data5['fids']), np.max(data6['fids']), np.max(data7['fids']))*1.5
y_min = min(np.min(data1['fids']), np.min(data2['fids']), np.min(data3['fids']), np.min(data4['fids']), np.min(data5['fids']), np.min(data6['fids']), np.min(data7['fids']))-0.05
fig, ax = plt.subplots()
plt.ylim(y_min, y_max)
ax.plot(centers, data1['fids'], 'r-', label='cGAN (30 classes)')
ax.plot(centers, data2['fids'], 'r--', label='cGAN (90 classes)')
ax.plot(centers, data3['fids'], 'r:', label='cGAN (150 classes)')
ax.plot(centers, data4['fids'], 'g-', label='vanilla CcGAN (HVDL)')
ax.plot(centers, data5['fids'], 'g--', label='vanilla CcGAN (SVDL)')
ax.plot(centers, data6['fids'], 'b-', label='improved CcGAN (HVDL)')
ax.plot(centers, data7['fids'], 'b--', label='improved CcGAN (SVDL)')
#ax.plot(centers, nrealimgs, 'r-.', label='number of images')
legend = ax.legend(loc='upper right', shadow=True, fontsize=9)
plt.xlabel("Steering Angle")
plt.ylabel("FID")
plt.show()
fig.savefig(filename, bbox_inches='tight')


## plot exclude vanilla CcGAN
filename = wd + "/comparison_fid_versus_center_exclude_vanila_CcGAN.pdf"
y_max = max(np.max(data1['fids']), np.max(data2['fids']), np.max(data3['fids']), np.max(data6['fids']), np.max(data7['fids']))*1.2
y_min = min(np.min(data1['fids']), np.min(data2['fids']), np.min(data3['fids']), np.min(data6['fids']), np.min(data7['fids']))-0.05
fig, ax = plt.subplots()
plt.ylim(y_min, y_max)
ax.plot(centers, data1['fids'], 'r-', label='cGAN (30 classes)')
ax.plot(centers, data2['fids'], 'r--', label='cGAN (90 classes)')
ax.plot(centers, data3['fids'], 'r:', label='cGAN (150 classes)')
ax.plot(centers, data6['fids'], 'g-', label='improved CcGAN (HVDL)')
ax.plot(centers, data7['fids'], 'g--', label='improved CcGAN (SVDL)')
#ax.plot(centers, nrealimgs, 'r-.', label='number of images')
legend = ax.legend(loc='upper right', shadow=True, fontsize=9)
plt.xlabel("Steering Angle")
plt.ylabel("FID")
plt.show()
fig.savefig(filename, bbox_inches='tight')


## plot exclude cGAN
filename = wd + "/comparison_fid_versus_center_exclude_cGAN.pdf"
y_max = max(np.max(data4['fids']), np.max(data5['fids']), np.max(data6['fids']), np.max(data7['fids']))*1.2
y_min = min(np.min(data4['fids']), np.min(data5['fids']), np.min(data6['fids']), np.min(data7['fids']))-0.05
fig, ax = plt.subplots()
plt.ylim(y_min, y_max)
ax.plot(centers, data4['fids'], 'b-', label='vanilla CcGAN (HVDL)')
ax.plot(centers, data5['fids'], 'b--', label='vanilla CcGAN (SVDL)')
ax.plot(centers, data6['fids'], 'g-', label='improved CcGAN (HVDL)')
ax.plot(centers, data7['fids'], 'g--', label='improved CcGAN (SVDL)')
#ax.plot(centers, nrealimgs, 'r-.', label='number of images')
legend = ax.legend(loc='upper right', shadow=True, fontsize=9)
plt.xlabel("Steering Angle")
plt.ylabel("FID")
plt.show()
fig.savefig(filename, bbox_inches='tight')




'''

Draw LS versus eval center

'''

## plot all
filename = wd + "/comparison_labelscore_versus_center_all.pdf"
y_max = max(np.max(data1['labelscores']), np.max(data2['labelscores']), np.max(data3['labelscores']), np.max(data4['labelscores']), np.max(data5['labelscores']), np.max(data6['labelscores']), np.max(data7['labelscores']))*2
y_min = min(np.min(data1['labelscores']), np.min(data2['labelscores']), np.min(data3['labelscores']), np.min(data4['labelscores']), np.min(data5['labelscores']), np.min(data6['labelscores']), np.min(data7['labelscores']))-0.05
fig, ax = plt.subplots()
plt.ylim(y_min, y_max)
ax.plot(centers, data1['labelscores'], 'r-', label='cGAN (30 classes)')
ax.plot(centers, data2['labelscores'], 'r--', label='cGAN (90 classes)')
ax.plot(centers, data3['labelscores'], 'r:', label='cGAN (150 classes)')
ax.plot(centers, data4['labelscores'], 'g-', label='vanilla CcGAN (HVDL)')
ax.plot(centers, data5['labelscores'], 'g--', label='vanilla CcGAN (SVDL)')
ax.plot(centers, data6['labelscores'], 'b-', label='improved CcGAN (HVDL)')
ax.plot(centers, data7['labelscores'], 'b--', label='improved CcGAN (SVDL)')
legend = ax.legend(loc='upper right', shadow=True, fontsize=9)
plt.xlabel("Steering Angle")
plt.ylabel("Label Score")
plt.show()
fig.savefig(filename, bbox_inches='tight')



## plot exclude vanilla CcGAN
filename = wd + "/comparison_labelscore_versus_center_exclude_vanila_CcGAN.pdf"
y_max = max(np.max(data1['labelscores']), np.max(data2['labelscores']), np.max(data3['labelscores']),  np.max(data6['labelscores']), np.max(data7['labelscores']))*3
y_min = min(np.min(data1['labelscores']), np.min(data2['labelscores']), np.min(data3['labelscores']), np.min(data6['labelscores']), np.min(data7['labelscores']))-0.05
fig, ax = plt.subplots()
plt.ylim(y_min, y_max)
ax.plot(centers, data1['labelscores'], 'r-', label='cGAN (30 classes)')
ax.plot(centers, data2['labelscores'], 'r--', label='cGAN (90 classes)')
ax.plot(centers, data3['labelscores'], 'r:', label='cGAN (150 classes)')
ax.plot(centers, data4['labelscores'], 'g-', label='vanilla CcGAN (HVDL)')
ax.plot(centers, data5['labelscores'], 'g--', label='vanilla CcGAN (SVDL)')
legend = ax.legend(loc='upper right', shadow=True, fontsize=9)
plt.xlabel("Steering Angle")
plt.ylabel("Label Score")
plt.show()
fig.savefig(filename, bbox_inches='tight')


## plot exclude cGAN
filename = wd + "/comparison_labelscore_versus_center_exclude_cGAN.pdf"
y_max = max(np.max(data4['labelscores']), np.max(data5['labelscores']), np.max(data6['labelscores']), np.max(data7['labelscores']))*1.5
y_min = min(np.min(data4['labelscores']), np.min(data5['labelscores']), np.min(data6['labelscores']), np.min(data7['labelscores']))-0.05
fig, ax = plt.subplots()
plt.ylim(y_min, y_max)
ax.plot(centers, data4['labelscores'], 'g-', label='vanilla CcGAN (HVDL)')
ax.plot(centers, data5['labelscores'], 'g--', label='vanilla CcGAN (SVDL)')
ax.plot(centers, data6['labelscores'], 'b-', label='improved CcGAN (HVDL)')
ax.plot(centers, data7['labelscores'], 'b--', label='improved CcGAN (SVDL)')
legend = ax.legend(loc='upper right', shadow=True, fontsize=9)
plt.xlabel("Steering Angle")
plt.ylabel("Label Score")
plt.show()
fig.savefig(filename, bbox_inches='tight')



'''

Draw entropy versus eval center

'''

## plot all
filename = wd + "/comparison_diversity_versus_center_all.pdf"
y_max = max(np.max(data1['entropies']), np.max(data2['entropies']), np.max(data3['entropies']), np.max(data4['entropies']), np.max(data5['entropies']), np.max(data6['entropies']))*1.2
y_min = min(np.min(data1['entropies']), np.min(data2['entropies']), np.min(data3['entropies']), np.min(data4['entropies']), np.min(data5['entropies']), np.min(data6['entropies']))-2
fig, ax = plt.subplots()
plt.ylim(y_min, y_max)
ax.plot(centers, data1['entropies'], 'r-', label='cGAN (30 classes)')
ax.plot(centers, data2['entropies'], 'r--', label='cGAN (90 classes)')
ax.plot(centers, data3['entropies'], 'r:', label='cGAN (150 classes)')
ax.plot(centers, data3['entropies'], 'b-', label='vanilla CcGAN (HVDL)')
ax.plot(centers, data4['entropies'], 'b--', label='vanilla CcGAN (SVDL)')
ax.plot(centers, data5['entropies'], 'g-', label='improved CcGAN (HVDL)')
ax.plot(centers, data6['entropies'], 'g--', label='improved CcGAN (SVDL)')
#ax.plot(centers, nrealimgs, 'r-.', label='number of images')
legend = ax.legend(loc='lower right', shadow=True, fontsize=9)
plt.xlabel("Steering Angle")
plt.ylabel("Diversity")
plt.show()
fig.savefig(filename, bbox_inches='tight')

## plot vanilla only
filename = wd + "/comparison_diversity_versus_center_vanilla.pdf"
y_max = max(np.max(data1['entropies']), np.max(data2['entropies']), np.max(data3['entropies']), np.max(data4['entropies']))*1.2
y_min = min(np.min(data1['entropies']), np.min(data2['entropies']), np.min(data3['entropies']), np.min(data4['entropies']))-1.2
fig, ax = plt.subplots()
plt.ylim(y_min, y_max)
ax.plot(centers, data1['entropies'], 'r-', label='cGAN (30 classes)')
ax.plot(centers, data2['entropies'], 'r--', label='cGAN (90 classes)')
ax.plot(centers, data3['entropies'], 'r:', label='cGAN (150 classes)')
ax.plot(centers, data3['entropies'], 'b-', label='vanilla CcGAN (HVDL)')
ax.plot(centers, data4['entropies'], 'b--', label='vanilla CcGAN (SVDL)')
legend = ax.legend(loc='lower right', shadow=True, fontsize=9)
plt.xlabel("Steering Angle")
plt.ylabel("Diversity")
plt.show()
fig.savefig(filename, bbox_inches='tight')

## plot exclude cGAN
filename = wd + "/comparison_diversity_versus_center_exclude_cGAN.pdf"
y_max = max(np.max(data3['entropies']), np.max(data4['entropies']), np.max(data5['entropies']), np.max(data6['entropies']))*1.2
y_min = min(np.min(data3['entropies']), np.min(data4['entropies']), np.min(data5['entropies']), np.min(data6['entropies']))-1
fig, ax = plt.subplots()
plt.ylim(y_min, y_max)
ax.plot(centers, data3['entropies'], 'b-', label='vanilla CcGAN (HVDL)')
ax.plot(centers, data4['entropies'], 'b--', label='vanilla CcGAN (SVDL)')
ax.plot(centers, data5['entropies'], 'g-', label='improved CcGAN (HVDL)')
ax.plot(centers, data6['entropies'], 'g--', label='improved CcGAN (SVDL)')
#ax.plot(centers, nrealimgs, 'r-.', label='number of images')
legend = ax.legend(loc='lower right', shadow=True, fontsize=9)
plt.xlabel("Steering Angle")
plt.ylabel("Diversity")
plt.show()
fig.savefig(filename, bbox_inches='tight')