"""

Pre-train a CNN on the whole dataset for evaluation purpose

"""
import os
import argparse
import shutil
import os
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




#############################
# Settings
#############################

parser = argparse.ArgumentParser(description='Pre-train CNNs')
parser.add_argument('--root_path', type=str, default='/home/xin/OneDrive/Working_directory/CcGAN/Cell200')
parser.add_argument('--data_path', type=str, default='/home/xin/OneDrive/Working_directory/CcGAN/dataset/Cell200')
parser.add_argument('--start_count', type=int, default=1, metavar='N')
parser.add_argument('--end_count', type=int, default=200, metavar='N')
parser.add_argument('--CNN', type=str, default='ResNet34_regre',
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
#parser.add_argument('--CVMode', action='store_true', default=False,
#                    help='CV mode?')
parser.add_argument('--img_size', type=int, default=64, metavar='N',
                    choices=[64,128])
args = parser.parse_args()

# change working directory
wd = args.root_path
os.chdir(wd)

from models import *
from utils import IMGs_dataset

# cuda
device = torch.device("cuda")
ngpu = torch.cuda.device_count()  # number of gpus

# random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True

# directories for checkpoint, images and log files
save_models_folder = wd + '/output/saved_models/'
os.makedirs(save_models_folder, exist_ok=True)

save_logs_folder = wd + '/output/saved_logs/'
os.makedirs(save_logs_folder, exist_ok=True)


###########################################################################################################
# Necessary functions
###########################################################################################################

#initialize CNNs
def net_initialization(Pretrained_CNN_Name, ngpu = 1):
    if Pretrained_CNN_Name == "ResNet18_regre":
        net = ResNet18_regre(ngpu = ngpu)
    elif Pretrained_CNN_Name == "ResNet34_regre":
        net = ResNet34_regre(ngpu = ngpu)
    elif Pretrained_CNN_Name == "ResNet50_regre":
        net = ResNet50_regre(ngpu = ngpu)
    elif Pretrained_CNN_Name == "ResNet101_regre":
        net = ResNet101_regre(ngpu = ngpu)

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
            batch_train_labels = batch_train_labels.type(torch.float).view(-1,1).cuda()

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

        valid_loss = valid_CNN(verbose=False)
        print('CNN: [epoch %d/%d] train_loss:%f valid_loss (avg_abs):%f' % (epoch+1, args.epochs, train_loss, valid_loss))
    #end for epoch

    return net, optimizer


def valid_CNN(verbose=True):
    net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    with torch.no_grad():
        abs_diff_avg = 0
        total = 0
        for batch_idx, (images, labels) in enumerate(validloader):
            images = images.type(torch.float).cuda()
            labels = labels.type(torch.float).view(-1).cpu().numpy()
            outputs,_ = net(images)
            outputs = outputs.view(-1).cpu().numpy()
            abs_diff_avg += np.sum(np.abs(labels-outputs))
            total += len(labels)
        if verbose:
            print('Validation Average Absolute Difference: {}'.format(abs_diff_avg / total))
    return abs_diff_avg / total

###########################################################################################################
# Training and Testing
###########################################################################################################
# data loader
h5py_file = args.data_path + '/Cell200_' + str(args.img_size) + 'x' + str(args.img_size) + '.h5'
hf = h5py.File(h5py_file, 'r')
counts_train = hf['CellCounts'][:]
counts_train = counts_train.astype(np.float)
images_train = hf['IMGs_grey'][:]
counts_valid = hf['CellCounts_test'][:]
counts_valid = counts_valid.astype(np.float)
images_valid = hf['IMGs_grey_test'][:]
hf.close()

selected_cellcounts = np.arange(args.start_count, args.end_count+1)
n_unique_cellcount = len(selected_cellcounts)
images_subset = np.zeros((n_unique_cellcount*1000, 1, args.img_size, args.img_size), dtype=np.uint8)
counts_subset = np.zeros(n_unique_cellcount*1000)
for i in range(n_unique_cellcount):
    curr_cellcount = selected_cellcounts[i]
    index_curr_cellcount = np.where(counts_train==curr_cellcount)[0]

    if i == 0:
        images_subset = images_train[index_curr_cellcount]
        counts_subset = counts_train[index_curr_cellcount]
    else:
        images_subset = np.concatenate((images_subset, images_train[index_curr_cellcount]), axis=0)
        counts_subset = np.concatenate((counts_subset, counts_train[index_curr_cellcount]))
# for i
images_train = images_subset
counts_train = counts_subset
del images_subset, counts_subset; gc.collect()

N_train = len(images_train)
N_valid = len(images_valid)
assert len(images_train) == len(counts_train)

print("Number of images: {}/{}".format(N_train, N_valid))

# noralization is very important here!!!!!!!!!
# counts = counts/np.max(counts)
counts_train = counts_train/args.end_count
counts_valid = counts_valid/args.end_count

if args.transform:
    trainset = IMGs_dataset(images_train, counts_train, normalize=True, rotate=True, degrees = [90,180,270], hflip = True, vflip = True)
else:
    trainset = IMGs_dataset(images_train, counts_train, normalize=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=8)

validset = IMGs_dataset(images_valid, counts_valid, normalize=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size_valid, shuffle=False, num_workers=8)


# model initialization
net, net_name = net_initialization(args.CNN, ngpu = ngpu)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr = args.base_lr, momentum= 0.9, weight_decay=args.weight_dacay)

filename_ckpt = save_models_folder + '/ckpt_' + net_name + '_epoch_' + str(args.epochs) +  '_seed_' + str(args.seed) + '_Transformation_' + str(args.transform) + '_Cell_' + str(args.end_count) + '.pth'

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
    checkpoint = torch.load(filename_ckpt)
    net.load_state_dict(checkpoint['net_state_dict'])
torch.cuda.empty_cache()#release GPU mem which is  not references


#validation
_ = valid_CNN(True)
torch.cuda.empty_cache()
