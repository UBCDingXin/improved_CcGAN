"""

Pre-train a CNN on the whole dataset for evaluation purpose

"""
import os
import argparse
import shutil
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
parser.add_argument('--root_path', type=str, default='')
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--CNN', type=str, default='ResNet34_regre',
                    help='CNN for training; ResNetXX')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train CNNs (default: 200)')
parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                    help='input batch size for training')
parser.add_argument('--batch_size_valid', type=int, default=10, metavar='N',
                    help='input batch size for testing')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='learning rate, default=0.1')
parser.add_argument('--weight_dacay', type=float, default=1e-4,
                    help='Weigth decay, default=1e-4')
parser.add_argument('--seed', type=int, default=2020, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--CVMode', action='store_true', default=False,
                   help='CV mode?')
parser.add_argument('--img_size', type=int, default=256, metavar='N')
parser.add_argument('--min_label', type=float, default=0.0)
parser.add_argument('--max_label', type=float, default=90.0)
args = parser.parse_args()


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
np.random.seed(args.seed)

# directories for checkpoint, images and log files
save_models_folder = wd + '/output/eval_models/'
os.makedirs(save_models_folder, exist_ok=True)


###########################################################################################################
# Data
###########################################################################################################
# data loader
data_filename = args.data_path + '/RC-49_' + str(args.img_size) + 'x' + str(args.img_size) + '.h5'
hf = h5py.File(data_filename, 'r')
labels = hf['labels'][:]
labels = labels.astype(float)
images = hf['images'][:]
hf.close()
N_all = len(images)
assert len(images) == len(labels)

q1 = args.min_label
q2 = args.max_label
indx = np.where((labels>q1)*(labels<q2)==True)[0]
labels = labels[indx]
images = images[indx]
assert len(labels)==len(images)


# normalize to [0, 1]
#min_label = np.min(labels)
#labels += np.abs(min_label)
#max_label = np.max(labels)
#labels /= max_label
labels /= args.max_label


# define training and validation sets
if args.CVMode:
    #80% Training; 20% valdation
    indx_all = np.arange(len(labels))
    np.random.shuffle(indx_all)
    indx_valid = indx_all[0:int(0.2*len(labels))]
    indx_train = indx_all[int(0.2*len(labels)):]

    trainset = IMGs_dataset(images[indx_train], labels[indx_train], normalize=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)
    validset = IMGs_dataset(images[indx_valid], labels[indx_valid], normalize=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size_valid, shuffle=False, num_workers=args.num_workers)

else:
    trainset = IMGs_dataset(images, labels, normalize=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True, num_workers=args.num_workers)




###########################################################################################################
# Necessary functions
###########################################################################################################

#initialize CNNs
def net_initialization(Pretrained_CNN_Name, ngpu = 1):
    if Pretrained_CNN_Name == "ResNet18_regre":
        net = ResNet18_regre_eval(ngpu = ngpu)
    elif Pretrained_CNN_Name == "ResNet34_regre":
        net = ResNet34_regre_eval(ngpu = ngpu)
    elif Pretrained_CNN_Name == "ResNet50_regre":
        net = ResNet50_regre_eval(ngpu = ngpu)
    elif Pretrained_CNN_Name == "ResNet101_regre":
        net = ResNet101_regre_eval(ngpu = ngpu)

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

    start_tmp = timeit.default_timer()
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

        if args.CVMode:
            valid_loss = valid_CNN(verbose=False)
            print('CNN: [epoch %d/%d] train_loss:%f valid_loss (avg_abs):%f Time: %.4f' % (epoch+1, args.epochs, train_loss, valid_loss, timeit.default_timer()-start_tmp))
        else:
            print('CNN: [epoch %d/%d] train_loss:%f Time:%.4f' % (epoch+1, args.epochs, train_loss, timeit.default_timer()-start_tmp))
    #end for epoch

    return net, optimizer

if args.CVMode:
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
                labels = labels * args.max_label
                outputs = outputs * args.max_label
                abs_diff_avg += np.sum(np.abs(labels-outputs))
                total += len(labels)

            if verbose:
                print('Validation Average Absolute Difference: {}'.format(abs_diff_avg/total))
        return abs_diff_avg/total



###########################################################################################################
# Training and validation
###########################################################################################################


# model initialization
net, net_name = net_initialization(args.CNN, ngpu = ngpu)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr = args.base_lr, momentum= 0.9, weight_decay=args.weight_dacay)

filename_ckpt = save_models_folder + '/ckpt_{}_epoch_{}_seed_{}_CVMode_{}.pth'.format(net_name, args.epochs, args.seed, args.CVMode)


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


if args.CVMode:
    #validation
    _ = valid_CNN(True)
    torch.cuda.empty_cache()
