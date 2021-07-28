
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

parser = argparse.ArgumentParser(description='Pre-train AE for computing FID')
parser.add_argument('--root_path', type=str, default='')
parser.add_argument('--data_path', type=str, default='')
parser.add_argument('--dim_bottleneck', type=int, default=512)
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train CNNs (default: 200)')
parser.add_argument('--resume_epoch', type=int, default=0)
parser.add_argument('--save_ckpt_freq', type=int, default=25)
parser.add_argument('--batch_size_train', type=int, default=256, metavar='N',
                    help='input batch size for training')
parser.add_argument('--batch_size_valid', type=int, default=10, metavar='N',
                    help='input batch size for testing')
parser.add_argument('--base_lr', type=float, default=1e-3,
                    help='learning rate, default=1e-3')
parser.add_argument('--lr_decay_epochs', type=int, default=50) #decay lr rate every dre_lr_decay_epochs epochs
parser.add_argument('--lr_decay_factor', type=float, default=0.1)
parser.add_argument('--lambda_sparsity', type=float, default=0, help='penalty for sparsity')
parser.add_argument('--weight_dacay', type=float, default=1e-5,
                    help='Weigth decay, default=1e-5')
parser.add_argument('--seed', type=int, default=2020, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--CVMode', action='store_true', default=False,
                   help='CV mode?')
parser.add_argument('--transform', action='store_true', default=False,
                    help='flip or rotate images for AE training')
parser.add_argument('--img_size', type=int, default=64, metavar='N',
                    choices=[64])
parser.add_argument('--min_label', type=float, default=1)
parser.add_argument('--max_label', type=float, default=200)
args = parser.parse_args()

wd = args.root_path
os.chdir(wd)
from models import *
from utils import IMGs_dataset, SimpleProgressBar

# some parameters in the opts
dim_bottleneck = args.dim_bottleneck
epochs = args.epochs
base_lr = args.base_lr
lr_decay_epochs = args.lr_decay_epochs
lr_decay_factor = args.lr_decay_factor
resume_epoch = args.resume_epoch
lambda_sparsity = args.lambda_sparsity


# random seed
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

# directories for checkpoint, images and log files
save_models_folder = wd + '/output/saved_models'
os.makedirs(save_models_folder, exist_ok=True)
save_AE_images_in_train_folder = wd + '/output/saved_images/AE_lambda_{}_images_in_train'.format(lambda_sparsity)
os.makedirs(save_AE_images_in_train_folder, exist_ok=True)
save_AE_images_in_valid_folder = wd + '/output/saved_images/AE_lambda_{}_images_in_valid'.format(lambda_sparsity)
os.makedirs(save_AE_images_in_valid_folder, exist_ok=True)


###########################################################################################################
# Data
###########################################################################################################
# data loader
data_filename = args.data_path + '/Cell200_' + str(args.img_size) + 'x' + str(args.img_size) + '.h5'
hf = h5py.File(data_filename, 'r')
labels = hf['CellCounts'][:]
labels = labels.astype(np.float)
images = hf['IMGs_grey'][:]
hf.close()
N_all = len(images)
assert len(images) == len(labels)


q1 = args.min_label
q2 = args.max_label
indx = np.where((labels>=q1)*(labels<=q2)==True)[0]
labels = labels[indx]
images = images[indx]
assert len(labels)==len(images)

# define training and validation sets
if args.CVMode:
    #90% Training; 10% valdation
    valid_prop = 0.1 #proportion of the validation samples
    indx_all = np.arange(len(images))
    np.random.shuffle(indx_all)
    indx_valid = indx_all[0:int(valid_prop*len(images))]
    indx_train = indx_all[int(valid_prop*len(images)):]

    if args.transform:
        trainset = IMGs_dataset(images[indx_train], labels=None, normalize=True, rotate=True, degrees = [90,180,270], hflip = True, vflip = True)
    else:
        trainset = IMGs_dataset(images[indx_train], labels=None, normalize=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True)
    validset = IMGs_dataset(images[indx_valid], labels=None, normalize=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=args.batch_size_valid, shuffle=False)

else:
    if args.transform:
        trainset = IMGs_dataset(images, labels=None, normalize=True, rotate=True, degrees = [90,180,270], hflip = True, vflip = True)
    else:
        trainset = IMGs_dataset(images, labels=None, normalize=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_train, shuffle=True)


###########################################################################################################
# Necessary functions
###########################################################################################################

def adjust_learning_rate(epoch, epochs, optimizer, base_lr, lr_decay_epochs, lr_decay_factor):
    lr = base_lr

    for i in range(epochs//lr_decay_epochs):
        if epoch >= (i+1)*lr_decay_epochs:
            lr *= lr_decay_factor

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def adjust_learning_rate(optimizer, epoch, base_lr):
#     lr = base_lr
#     if epoch >= 50:
#         lr /= 10
#     if epoch >= 120:
#         lr /= 10
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr



def train_AE():

    # define optimizer
    params = list(net_encoder.parameters()) + list(net_decoder.parameters())
    optimizer = torch.optim.Adam(params, lr = base_lr, betas=(0.5, 0.999), weight_decay=args.weight_dacay)
    # optimizer = torch.optim.SGD(params, lr = base_lr, momentum=0.9)

    # criterion
    criterion = nn.MSELoss()

    if resume_epoch>0:
        print("Loading ckpt to resume training AE >>>")
        ckpt_fullpath = save_models_folder + "/AE_checkpoint_intrain/AE_checkpoint_epoch_{}_lambda_{}.pth".format(resume_epoch, lambda_sparsity)
        checkpoint = torch.load(ckpt_fullpath)
        net_encoder.load_state_dict(checkpoint['net_encoder_state_dict'])
        net_decoder.load_state_dict(checkpoint['net_decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
        gen_iterations = checkpoint['gen_iterations']
    else:
        gen_iterations = 0

    start_time = timeit.default_timer()
    for epoch in range(resume_epoch, epochs):

        adjust_learning_rate(epoch, epochs, optimizer, base_lr, lr_decay_epochs, lr_decay_factor)

        train_loss = 0

        for batch_idx, batch_real_images in enumerate(trainloader):

            net_encoder.train()
            net_decoder.train()

            batch_size_curr = batch_real_images.shape[0]

            batch_real_images = batch_real_images.type(torch.float).cuda()


            batch_features = net_encoder(batch_real_images)
            batch_recons_images = net_decoder(batch_features)

            '''
            based on https://debuggercafe.com/sparse-autoencoders-using-l1-regularization-with-pytorch/
            '''
            loss = criterion(batch_recons_images, batch_real_images) + lambda_sparsity * batch_features.mean()

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()

            gen_iterations += 1

            if gen_iterations % 100 == 0:
                n_row=min(10, int(np.sqrt(batch_size_curr)))
                net_encoder.eval()
                net_decoder.eval()
                with torch.no_grad():
                    batch_recons_images = net_decoder(net_encoder(batch_real_images[0:n_row**2]))
                    batch_recons_images = batch_recons_images.detach().cpu()
                save_image(batch_recons_images.data, save_AE_images_in_train_folder + '/{}.png'.format(gen_iterations), nrow=n_row, normalize=True)

            if gen_iterations % 20 == 0:
                print("AE+lambda{}: [step {}] [epoch {}/{}] [train loss {}] [Time {}]".format(lambda_sparsity, gen_iterations, epoch+1, epochs, train_loss/(batch_idx+1), timeit.default_timer()-start_time) )
        # end for batch_idx

        if (epoch+1) % args.save_ckpt_freq == 0:
            save_file = save_models_folder + "/AE_checkpoint_intrain/AE_checkpoint_epoch_{}_lambda_{}.pth".format(epoch+1, lambda_sparsity)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'gen_iterations': gen_iterations,
                    'net_encoder_state_dict': net_encoder.state_dict(),
                    'net_decoder_state_dict': net_decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)
    #end for epoch

    return net_encoder, net_decoder


if args.CVMode:
    def valid_AE():
        net_encoder.eval()
        net_decoder.eval()
        with torch.no_grad():
            for batch_idx, images in enumerate(validloader):
                images = images.type(torch.float).cuda()
                features = net_encoder(images)
                recons_images = net_decoder(features)
                save_image(recons_images.data, save_AE_images_in_valid_folder + '/{}_recons.png'.format(batch_idx), nrow=10, normalize=True)
                save_image(images.data, save_AE_images_in_valid_folder + '/{}_real.png'.format(batch_idx), nrow=10, normalize=True)
        return None



###########################################################################################################
# Training and validation
###########################################################################################################

# model initialization
net_encoder = encoder(dim_bottleneck=args.dim_bottleneck).cuda()
net_decoder = decoder(dim_bottleneck=args.dim_bottleneck).cuda()
net_encoder = nn.DataParallel(net_encoder)
net_decoder = nn.DataParallel(net_decoder)

filename_ckpt = save_models_folder + '/ckpt_AE_epoch_{}_seed_{}_CVMode_{}.pth'.format(args.epochs, args.seed, args.CVMode)

# training
if not os.path.isfile(filename_ckpt):
    print("\n Begin training AE: ")
    start = timeit.default_timer()
    net_encoder, net_decoder = train_AE()
    stop = timeit.default_timer()
    print("Time elapses: {}s".format(stop - start))
    # save model
    torch.save({
    'net_encoder_state_dict': net_encoder.state_dict(),
    'net_decoder_state_dict': net_decoder.state_dict(),
    }, filename_ckpt)
else:
    print("\n Ckpt already exists")
    print("\n Loading...")
    checkpoint = torch.load(filename_ckpt)
    net_encoder.load_state_dict(checkpoint['net_encoder_state_dict'])
    net_decoder.load_state_dict(checkpoint['net_decoder_state_dict'])

if args.CVMode:
    #validation
    _ = valid_AE()
