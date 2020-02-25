
wd = '/home/xin/OneDrive/Working_directory/DDRE_Sampling_GANs/MNIST'

import os
os.chdir(wd)
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import h5py
import random
from PIL import Image

N_TRAIN = 1000
N_VALID = 10000

IMG_SIZE = 28
NC = 1
N_CLASS = 10


# random seed
SEED=2019
random.seed(SEED)
torch.manual_seed(SEED)

means = [0.5]; stds = [0.5]
transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(means, stds),
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)

images_train_all = trainset.data.numpy()
images_train_all = images_train_all[:,np.newaxis,:,:]
labels_train_all = trainset.targets.numpy()
indx_all = np.arange(len(images_train_all))
np.random.shuffle(indx_all)
indx_train = indx_all[0:N_TRAIN]
indx_valid = indx_all[N_TRAIN:(N_TRAIN+N_VALID)]
images_train = images_train_all[indx_train]
labels_train = labels_train_all[indx_train]
images_valid = images_train_all[indx_valid]
labels_valid = labels_train_all[indx_valid]


h5py_file = wd+'/data/MNIST_reduced_trainset_'+str(N_TRAIN)+'.h5'
with h5py.File(h5py_file, "w") as f:
    f.create_dataset('images_train', data = images_train)
    f.create_dataset('labels_train', data = labels_train)
    f.create_dataset('images_valid', data = images_valid)
    f.create_dataset('labels_valid', data = labels_valid)

#test h5 file
hf = h5py.File(h5py_file, 'r')
images_train = hf['images_train'][:]
labels_train = hf['labels_train'][:]
images_valid = hf['images_valid'][:]
labels_valid = hf['labels_valid'][:]


