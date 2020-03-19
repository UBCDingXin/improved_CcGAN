"""

Pre-train a CNN on the whole dataset for evaluation purpose

"""
import os
wd = '/home/xin/OneDrive/Working_directory/Continuous_cGAN/CellCounting'

import argparse
import shutil
import os
os.chdir(wd)
import timeit
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch import autograd
from torchvision.utils import save_image
import csv
from tqdm import tqdm
import gc
import h5py

from models import *
from utils import IMGs_dataset


#############################
# Settings
#############################

parser = argparse.ArgumentParser(description='Pre-train CNNs')
parser.add_argument('--start_count', type=int, default=0, metavar='N')
parser.add_argument('--end_count', type=int, default=300, metavar='N')
parser.add_argument('--CNN', type=str, default='ResNet34_class',
                    help='CNN for training; ResNetXX')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train CNNs (default: 200)')
parser.add_argument('--batch_size_train', type=int, default=256, metavar='N',
                    help='input batch size for training')
parser.add_argument('--batch_size_valid', type=int, default=10, metavar='N',
                    help='input batch size for testing')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='learning rate, default=0.1')
parser.add_argument('--weight_dacay', type=float, default=1e-4,
                    help='Weigth decay, default=1e-4')
parser.add_argument('--seed', type=int, default=2020, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--transform', action='store_true', default=False,
                    help='crop images for CNN training')
parser.add_argument('--CVMode', action='store_true', default=False,
                    help='CV mode?')
parser.add_argument('--img_size', type=int, default=64, metavar='N',
                    choices=[64,128])
args = parser.parse_args()


# cuda
device = torch.device("cuda")
ngpu = torch.cuda.device_count()  # number of gpus

# random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# directories for checkpoint, images and log files
save_models_folder = wd + '/Output/saved_models/'
os.makedirs(save_models_folder, exist_ok=True)

save_logs_folder = wd + '/Output/saved_logs/'
os.makedirs(save_logs_folder, exist_ok=True)


###########################################################################################################
# Necessary functions
###########################################################################################################

#initialize CNNs
def net_initialization(Pretrained_CNN_Name, ngpu = 1, num_classes=args.end_count):
    if Pretrained_CNN_Name == "ResNet18_class":
        net = ResNet18_class(num_classes=num_classes, ngpu = ngpu)
    elif Pretrained_CNN_Name == "ResNet34_class":
        net = ResNet34_class(num_classes=num_classes, ngpu = ngpu)
    elif Pretrained_CNN_Name == "ResNet50_class":
        net = ResNet50_class(num_classes=num_classes, ngpu = ngpu)
    elif Pretrained_CNN_Name == "ResNet101_class":
        net = ResNet101_class(num_classes=num_classes, ngpu = ngpu)

    net_name = 'PreCNNForEvalGANs_' + Pretrained_CNN_Name #get the net's name
    net = net.to(device)

    return net, net_name

#adjust CNN learning rate
def adjust_learning_rate(optimizer, epoch, BASE_LR_CNN):
    lr = BASE_LR_CNN
    # if epoch >= 35:
    #     lr /= 10
    # if epoch >= 70:
    #     lr /= 10
    if epoch >= 50:
        lr /= 10
    if epoch >= 120:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_CNN():

    for epoch in range(args.epochs):
        net.train()
        train_loss = 0
        adjust_learning_rate(optimizer, epoch, args.base_lr)
        for batch_idx, (batch_train_images, batch_train_labels) in enumerate(trainloader):

            # batch_train_images = nn.functional.interpolate(batch_train_images, size = (299,299), scale_factor=None, mode='bilinear', align_corners=False)

            batch_train_images = batch_train_images.type(torch.float).cuda()
            batch_train_labels = batch_train_labels.type(torch.long).cuda()

            #Forward pass
            outputs,_ = net(batch_train_images)
            loss = criterion(outputs, batch_train_labels)

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
        #end for batch_idx
        train_loss = train_loss / len(trainloader)

        if args.CVMode:
            valid_acc = valid_CNN(verbose=False)
            print('CNN: [epoch %d/%d] train_loss:%f valid_acc:%f' % (epoch+1, args.epochs, train_loss, valid_acc))
        else:
            print('CNN: [epoch %d/%d] train_loss:%f' % (epoch+1, args.epochs, train_loss))

    #end for epoch

    return net, optimizer

if args.CVMode:
    def valid_CNN(verbose=True):
        net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_idx, (images, labels) in enumerate(validloader):
                images = images.type(torch.float).cuda()
                labels = labels.type(torch.long).cuda()
                outputs,_ = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            if verbose:
                print('Valid Accuracy of the model on the validation set: {} %'.format(100.0 * correct / total))
        return 100.0 * correct / total




###########################################################################################################
# Training and Testing
###########################################################################################################
# data loader
h5py_file = wd+'/data/Cell300_' + str(args.img_size) + 'x' + str(args.img_size) + '.h5'
hf = h5py.File(h5py_file, 'r')
counts = hf['CellCounts'][:]
counts = counts.astype(np.float)
images = hf['IMGs_grey'][:]
hf.close()

selected_cellcounts = np.arange(args.start_count, args.end_count+1)
n_unique_cellcount = len(selected_cellcounts)
images_subset = np.zeros((n_unique_cellcount*1000, 1, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
counts_subset = np.zeros(n_unique_cellcount*1000)
for i in range(n_unique_cellcount):
    curr_cellcount = selected_cellcounts[i]
    index_curr_cellcount = np.where(counts==curr_cellcount)[0]

    if i == 0:
        images_subset = images[index_curr_cellcount]
        counts_subset = counts[index_curr_cellcount]
    else:
        images_subset = np.concatenate((images_subset, images[index_curr_cellcount]), axis=0)
        counts_subset = np.concatenate((counts_subset, counts[index_curr_cellcount]))
# for i
images = images_subset
counts = counts_subset
del images_subset, counts_subset; gc.collect()
# counts = counts/args.end_count #in classification mode, we don't need to normalize labels


## convert cell count to class labels
count2class = dict() #convert count to class label
class2count = dict() #convert class label to count
for i in range(n_unique_cellcount):
    count2class[selected_cellcounts[i]]=i
    class2count[i] = selected_cellcounts[i]
counts_new = -1*np.ones(len(counts))
for i in range(len(counts)):
    counts_new[i] = count2class[counts[i]]
assert np.sum(counts_new<0)==0
counts = counts_new
del counts_new; gc.collect()

# compute number of samples
N = len(images)
N_train = int(0.8*N)
N_valid = N - N_train
assert len(images) == len(counts)

print("Number of images: {} ({}/{})".format(N, N_train, N_valid))


if args.CVMode: # divide dataset into a training set 80% and a validation set 20%
    indx_all = np.arange(N)
    np.random.shuffle(indx_all)
    indx_train = indx_all[0:N_train]
    indx_valid = indx_all[N_train:]
    assert len(indx_train) == N_train
    assert len(indx_valid) == N_valid
    images_train = images[indx_train]
    counts_train = counts[indx_train]
    images_valid = images[indx_valid]
    counts_valid = counts[indx_valid]
else:
    images_train = images
    counts_train = counts
del images, counts; gc.collect()


if args.transform:
    trainset = IMGs_dataset(images_train, counts_train, normalize=True, rotate=True, degrees = [90,180,270], hflip = True, vflip = True)
else:
    trainset = IMGs_dataset(images_train, counts_train, normalize=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=8)

if args.CVMode:
    validset = IMGs_dataset(images_valid, counts_valid, normalize=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size_train, shuffle=False, num_workers=8)


# model initialization
net, net_name = net_initialization(args.CNN, ngpu = ngpu, num_classes = n_unique_cellcount)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = args.base_lr, momentum= 0.9, weight_decay=args.weight_dacay)

if args.CVMode:
    filename_ckpt = save_models_folder + '/ckpt_' + net_name + '_epoch_' + str(args.epochs) +  '_SEED_' + str(args.seed) + '_Transformation_' + str(args.transform) + '_CVMode_' + str(args.CVMode) + '_Cell_' + str(args.end_count)
else:
    filename_ckpt = save_models_folder + '/ckpt_' + net_name + '_epoch_' + str(args.epochs) +  '_SEED_' + str(args.seed) + '_Transformation_' + str(args.transform) + '_Cell_' + str(args.end_count)

# training
if not os.path.isfile(filename_ckpt):
    # TRAIN CNN
    print("\n Begin training CNN: ")
    start = timeit.default_timer()
    net, optimizer = train_CNN()
    stop = timeit.default_timer()
    print("Time elapses: {}s".format(stop - start))
    # save model
    torch.save({
    'net_state_dict': net.state_dict(),
    }, filename_ckpt)
else:
    print("\n Ckpt already exists")
    print("\n Loading...")
torch.cuda.empty_cache()#release GPU mem which is  not references


#validation
if args.CVMode:
    _ = valid_CNN(True)
    torch.cuda.empty_cache()
