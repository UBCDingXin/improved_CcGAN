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
parser.add_argument('--root_path', type=str, default='')
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--CNN', type=str, default='ResNet34_class',
                    help='CNN for training; ResNetXX')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train CNNs (default: 200)')
parser.add_argument('--batch_size_train', type=int, default=64, metavar='N',
                    help='input batch size for training')
parser.add_argument('--batch_size_valid', type=int, default=10, metavar='N',
                    help='input batch size for testing')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='learning rate, default=0.01')
parser.add_argument('--weight_dacay', type=float, default=1e-4,
                    help='Weigth decay, default=1e-4')
parser.add_argument('--seed', type=int, default=2021, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--CVMode', action='store_true', default=False,
                   help='CV mode?')
parser.add_argument('--valid_proport', type=float, default=0.1,
                    help='Proportion of validation samples')
parser.add_argument('--img_size', type=int, default=192, metavar='N')
parser.add_argument('--min_label', type=int, default=1, metavar='N')
parser.add_argument('--max_label', type=int, default=60, metavar='N')
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


# data loader
data_filename = args.data_path + '/UTKFace_' + str(args.img_size) + 'x' + str(args.img_size) + '.h5'
hf = h5py.File(data_filename, 'r')
ages = hf['labels'][:]
ages = ages.astype(float)
labels = hf['races'][:]
images = hf['images'][:]
hf.close()
num_classes = len(set(labels))

# subset of UTKFace
selected_ages = np.arange(args.min_label, args.max_label+1)
for i in range(len(selected_ages)):
    curr_age = selected_ages[i]
    index_curr_age = np.where(ages==curr_age)[0]
    if i == 0:
        images_subset = images[index_curr_age]
        labels_subset = labels[index_curr_age]
        ages_subset = ages[index_curr_age]
    else:
        images_subset = np.concatenate((images_subset, images[index_curr_age]), axis=0)
        labels_subset = np.concatenate((labels_subset, labels[index_curr_age]))
        ages_subset = np.concatenate((ages_subset, ages[index_curr_age]))
# for i
images = images_subset
labels = labels_subset
ages = ages_subset
del images_subset, labels_subset, ages_subset; gc.collect()


# define training (and validaiton) set
if args.CVMode:
    for i in range(num_classes):
        indx_i = np.where(labels==i)[0] #i-th class
        np.random.shuffle(indx_i)
        num_imgs_all_i = len(indx_i)
        num_imgs_valid_i = int(num_imgs_all_i*args.valid_proport)
        if i == 0:
            indx_valid = indx_i[0:num_imgs_valid_i]
            indx_train = indx_i[num_imgs_valid_i:]
        else:
            indx_valid = np.concatenate((indx_valid, indx_i[0:num_imgs_valid_i]))
            indx_train = np.concatenate((indx_train, indx_i[num_imgs_valid_i:]))
    #end for i
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
def net_initialization(Pretrained_CNN_Name, ngpu = 1, num_classes=num_classes):
    if Pretrained_CNN_Name == "ResNet18_class":
        net = ResNet18_class_eval(num_classes=num_classes, ngpu = ngpu)
    elif Pretrained_CNN_Name == "ResNet34_class":
        net = ResNet34_class_eval(num_classes=num_classes, ngpu = ngpu)
    elif Pretrained_CNN_Name == "ResNet50_class":
        net = ResNet50_class_eval(num_classes=num_classes, ngpu = ngpu)
    elif Pretrained_CNN_Name == "ResNet101_class":
        net = ResNet101_class_eval(num_classes=num_classes, ngpu = ngpu)

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
# model initialization
net, net_name = net_initialization(args.CNN, ngpu = ngpu, num_classes = num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr = args.base_lr, momentum= 0.9, weight_decay=args.weight_dacay)

filename_ckpt = save_models_folder + '/ckpt_{}_epoch_{}_seed_{}_classify_{}_races_CVMode_{}.pth'.format(net_name, args.epochs, args.seed, num_classes, args.CVMode)

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
