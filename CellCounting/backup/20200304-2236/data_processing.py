'''

Split 200 data into 4 folds randomly. 
4-fold CV

'''


wd = '/home/xin/OneDrive/Working_directory/Continuous_cGAN/CellCounting'

import os
os.chdir(wd)
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import h5py
import random
from PIL import Image
import pickle


N_ALL = 200
FOLDS = 4
assert N_ALL%FOLDS==0
N_FOLD = int(N_ALL/FOLDS)
RESIZE = True
IMG_SIZE = 64

# random seed
SEED=2019
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

#test h5 file
h5py_file = "./data/VGG_dataset.h5"
hf = h5py.File(h5py_file, 'r')
IMGs_rgb = hf['IMGs_rgb'][:]
IMGs_grey = hf['IMGs_grey'][:]
CellCounts = hf['CellCounts'][:]
print(hf.keys())
hf.close()

#im1 = Image.fromarray(IMGs_rgb[0][0])
#im1.show()
#im2 = Image.fromarray(IMGs_rgb[0][1])
#im2.show()
#im3 = Image.fromarray(IMGs_rgb[0][2])
#im3.show()

IMGs_grey_resize = np.zeros((N_ALL, 1, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
IMGs_rgb_resize = np.zeros((N_ALL, 3, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
if RESIZE: #resize images
    for i in range(N_ALL):
        im_grey = Image.fromarray(IMGs_grey[i][0], mode='L')
        im_grey = im_grey.resize((IMG_SIZE, IMG_SIZE))
        # im_grey.show()
        im_grey = np.array(im_grey)
        IMGs_grey_resize[i,0,:,:] = im_grey
        
        im_rgb = Image.fromarray(np.transpose(IMGs_rgb[i], (1,2,0)), mode='RGB')
        im_rgb = im_rgb.resize((IMG_SIZE, IMG_SIZE))
        # im_grey.show()
        im_rgb = np.transpose(np.array(im_rgb), (2,0,1))
        IMGs_rgb_resize[i] = im_rgb

#dump to h5 file
h5py_file = './data/VGG_dataset_'+ str(IMG_SIZE) + 'x' + str(IMG_SIZE) +'.h5'
with h5py.File(h5py_file, "w") as f:
    f.create_dataset('IMGs_rgb', data = IMGs_rgb_resize)
    f.create_dataset('IMGs_grey', data = IMGs_grey_resize)
    f.create_dataset('CellCounts', data = CellCounts)


#dump grey scale images
dump_path = wd + '/data/grey_scale/'
os.makedirs(dump_path, exist_ok=True)
for i in range(len(IMGs_grey)):
    im = Image.fromarray(IMGs_grey[i][0], mode='L')
    im = im.save(dump_path+str(i)+".png") 


#dump grey scale resized images
dump_path = wd + '/data/grey_scale_resize/'
os.makedirs(dump_path, exist_ok=True)
for i in range(len(IMGs_grey)):
    im = Image.fromarray(IMGs_grey_resize[i][0], mode='L')
    im = im.save(dump_path+str(i)+".png") 

#split the whole dataset into four parts randomly
indx_all = np.arange(N_ALL)
np.random.shuffle(indx_all)
indx_fold_all = []
for i in range(FOLDS):
    indx_fold_all.append(indx_all[(i*N_FOLD):((i+1)*N_FOLD)])
## check
indx_fold_check = indx_fold_all[0]
for i in range(FOLDS-1):
    indx_fold_check = np.concatenate((indx_fold_check, indx_fold_all[i+1]))
assert (indx_all==indx_fold_check).all()

#store the data split of each round of CV in a pickle file
indx_train_CV = []
indx_valid_CV = []
for i in range(FOLDS):
    indx_valid = indx_fold_all[i]
    indx_train = np.array(list(set(indx_all).difference(set(indx_valid))))
    indx_train_CV.append(indx_train)
    indx_valid_CV.append(indx_valid)
    ## check
    assert set(np.concatenate((indx_train_CV[i], indx_valid_CV[i])))==set(indx_all)

filename_CV_datasplit = "./data/VGG_dataset_CV_datasplit_NFOLDS_" + str(FOLDS) + ".pickle"
CV_datasplit_dict = {"indx_train_CV":indx_train_CV, "indx_valid_CV":indx_valid_CV}
with open(filename_CV_datasplit, 'wb') as pf:
    pickle.dump(CV_datasplit_dict, pf)
with open(filename_CV_datasplit, 'rb') as pf:
    CV_datasplit_dict = pickle.load(pf)
