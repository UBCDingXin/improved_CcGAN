import os
wd = '/home/xin/OneDrive/Working_directory/Continuous_cGAN/CellCounting'

os.chdir(wd)
import timeit
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.nn import functional as F
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch import autograd
from torchvision.utils import save_image
from tqdm import tqdm
import gc
from itertools import groupby
import argparse
import multiprocessing
from multiprocessing import Pool
from scipy.stats import ks_2samp
import h5py
import pickle
from torch.utils.tensorboard import SummaryWriter

from utils import *
from models import *
from Train_DCGAN import *
from Train_cDCGAN import *
from Train_WGAN import *
from Train_cWGAN import *
from Train_Continuous_cDCGAN import *

#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
parser = argparse.ArgumentParser(description='Density-ratio based sampling for GANs')

parser.add_argument('--GAN', type=str, default='DCGAN',
                    choices=['DCGAN', 'cDCGAN', 'WGANGP', 'cWGANGP', 'Continuous_cDCGAN'])
parser.add_argument('--seed', type=int, default=2019, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--transform', action='store_true', default=False,
                    help='rotate or flip images for GAN training')
parser.add_argument('--normalize_count', action='store_true', default=False,
                    help='normalize cell counts')

parser.add_argument('--kernel_sigma', type=float, default=0.1)
parser.add_argument('--b_int_digits', type=int, default=16,
                    help='How many digits used to represent the integer part of a label')
parser.add_argument('--b_dec_digits', type=int, default=16,
                    help='How many digits used to represent the decimal part of a label')


parser.add_argument('--epoch_gan', type=int, default=1000)
parser.add_argument('--lr_g_gan', type=float, default=1e-4,
                    help='learning rate for generator')
parser.add_argument('--lr_d_gan', type=float, default=1e-4,
                    help='learning rate for discriminator')
parser.add_argument('--dim_gan', type=int, default=128,
                    help='Latent dimension of GAN')
parser.add_argument('--batch_size_gan', type=int, default=64, metavar='N',
                    help='input batch size for training GAN')
parser.add_argument('--resumeTrain_gan', type=int, default=0)


parser.add_argument('--samp_batch_size', type=int, default=10)
parser.add_argument('--nfake', type=int, default=50000)

args = parser.parse_args()

#-----------------------------
# images
NC = 1 #number of channels
IMG_SIZE = 64
n_all = 200
# nvalid = int(n_all/args.nfolds)
# ntrain = n_all-nvalid

#--------------------------------
# system
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NCPU = multiprocessing.cpu_count()
cudnn.benchmark = True # For fast training

#-------------------------------
# GAN
ResumeEpoch_gan = args.resumeTrain_gan
ADAM_beta1 = 0.5 #parameters for ADAM optimizer
ADAM_beta2 = 0.999

#-------------------------------
# sampling parameters
samp_batch_size = args.samp_batch_size #batch size for sampling from GAN or enhanced sampler

#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
np.random.seed(args.seed)

#-------------------------------
# output folders
save_models_folder = wd + '/Output/saved_models'
os.makedirs(save_models_folder, exist_ok=True)
save_images_folder = wd + '/Output/saved_images'
os.makedirs(save_images_folder, exist_ok=True)
save_GANimages_InTrain_folder = wd + '/Output/saved_images/' + args.GAN + '_InTrain/'
os.makedirs(save_GANimages_InTrain_folder, exist_ok=True)
save_traincurves_folder = wd + '/Output/Training_loss_fig'
os.makedirs(save_traincurves_folder, exist_ok=True)



#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
# data loader
h5py_file = wd+'/data/VGG_dataset_64x64.h5'
hf = h5py.File(h5py_file, 'r')
counts = hf['CellCounts'][:]
counts = counts.astype(np.float)
# images = hf['IMGs_rgb'][:]
images = hf['IMGs_grey'][:]
hf.close()

#normalize count
if args.normalize_count:
    min_count = np.min(counts)
    max_count = np.max(counts)
    if min_count<0:
        shift_count = abs(min_count)
        max_count = max_count + shift_count
    else:
        shift_count = 0

    counts += shift_count
    counts /= max_count # [0,1]

    counts = (counts-0.5)/0.5
else:
    shift_count = 0
    max_count = 1


# kernel_sigma = np.std(counts)
kernel_sigma = args.kernel_sigma
print("Sigma of kernel is %f" % kernel_sigma)

if args.transform:
    trainset = IMGs_dataset(images, counts, normalize=True, rotate=True, degrees = [90,180,270], hflip = True, vflip = True)
else:
    trainset = IMGs_dataset(images, counts, normalize=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_gan, shuffle=True, num_workers=8)






#######################################################################################
'''                                    GAN training                                 '''
#######################################################################################
Filename_GAN = save_models_folder + '/ckpt_' + args.GAN + '_epoch_' + str(args.epoch_gan) + '_SEED_' + str(args.seed)

start = timeit.default_timer()
print("\n Begin Training %s:" % args.GAN)
#----------------------------------------------
# DCGAN
if args.GAN == "DCGAN" and not os.path.isfile(Filename_GAN):
    netG = cnn_generator(NGPU, args.dim_gan)
    netD = cnn_discriminator(True, NGPU)
    if args.resumeTrain_gan==0:
        netG.apply(weights_init)
        netD.apply(weights_init)
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g_gan, betas=(ADAM_beta1, ADAM_beta2))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr_d_gan, betas=(ADAM_beta1, ADAM_beta2))

    # Start training
    netG, netD, optimizerG, optimizerD = train_DCGAN(args.epoch_gan, args.dim_gan, trainloader, netG, netD, optimizerG, optimizerD, criterion, save_GANimages_InTrain_folder, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan)

    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
    }, Filename_GAN)

    # function for sampling from a trained GAN
    def fn_sampleGAN_no_label(nfake, batch_size):
        images = SampDCGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
        return images

#----------------------------------------------
# WGANGP
elif args.GAN == "WGANGP"  and not os.path.isfile(Filename_GAN):
    netG = cnn_generator(NGPU, args.dim_gan)
    netD = cnn_discriminator(False, NGPU)
    if args.resumeTrain_gan==0:
        netG.apply(weights_init)
        netD.apply(weights_init)
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g_gan, betas=(ADAM_beta1, ADAM_beta2))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr_d_gan, betas=(ADAM_beta1, ADAM_beta2))

    # Start training
    netG, netD, optimizerG, optimizerD = train_WGANGP(args.epoch_gan, args.dim_gan, trainloader, netG, netD, optimizerG, optimizerD, save_GANimages_InTrain_folder, LAMBDA = 10, CRITIC_ITERS=5, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan)
    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
    }, Filename_GAN)

    # function for sampling from a trained GAN
    def fn_sampleGAN_no_label(nfake, batch_size):
        images = SampWGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
        return images

#----------------------------------------------
# cDCGAN
elif args.GAN == "cDCGAN"  and not os.path.isfile(Filename_GAN):
    netG = cond_cnn_generator(NGPU, args.dim_gan)
    netD = cond_cnn_discriminator(True, NGPU)
    if args.resumeTrain_gan==0:
        netG.apply(weights_init)
        netD.apply(weights_init)
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g_gan, betas=(ADAM_beta1, ADAM_beta2))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr_d_gan, betas=(ADAM_beta1, ADAM_beta2))

    # Start training
    netG, netD, optimizerG, optimizerD = train_cDCGAN(args.epoch_gan, args.dim_gan, trainloader, netG, netD, optimizerG, optimizerD, criterion, save_GANimages_InTrain_folder, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan, normalize_count = args.normalize_count, shift_label = shift_count, max_label = max_count)

    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
    }, Filename_GAN)

    # function for sampling from a trained GAN
    def fn_sampleGAN_with_label(nfake, batch_size):
        images, cellcounts = SampcDCGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, normalize_count = args.normalize_count, shift_label = shift_count, max_label = max_count)
        return images, cellcounts

#----------------------------------------------
# cWGANGP
elif args.GAN == "cWGANGP"  and not os.path.isfile(Filename_GAN):
    netG = cond_cnn_generator(NGPU, args.dim_gan)
    netD = cond_cnn_discriminator(False, NGPU)
    if args.resumeTrain_gan==0:
        netG.apply(weights_init)
        netD.apply(weights_init)
    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g_gan, betas=(ADAM_beta1, ADAM_beta2))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr_d_gan, betas=(ADAM_beta1, ADAM_beta2))

    # Start training
    netG, netD, optimizerG, optimizerD = train_cWGANGP(args.epoch_gan, args.dim_gan, trainloader, netG, netD, optimizerG, optimizerD, save_GANimages_InTrain_folder, LAMBDA = 10, CRITIC_ITERS=5, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan, normalize_count = args.normalize_count, shift_label = shift_count, max_label = max_count)

    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
    }, Filename_GAN)

    # function for sampling from a trained GAN
    def fn_sampleGAN_with_label(nfake, batch_size):
        images, cellcounts = SampcWANGP(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, normalize_count = args.normalize_count, shift_label = shift_count, max_label = max_count)
        return images, cellcounts


#----------------------------------------------
# Concitnuous cDCGAN
elif args.GAN == "Continuous_cDCGAN"  and not os.path.isfile(Filename_GAN):
    netG = cond_cnn_generator(NGPU, args.dim_gan)
    netD = cond_cnn_discriminator(True, NGPU)
    if args.resumeTrain_gan==0:
        netG.apply(weights_init)
        netD.apply(weights_init)
    optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g_gan, betas=(ADAM_beta1, ADAM_beta2))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr_d_gan, betas=(ADAM_beta1, ADAM_beta2))

    tfboard_writer = SummaryWriter(wd+'/Output/saved_logs')

    # Start training
    netG, netD, optimizerG, optimizerD = train_Continuous_cDCGAN(counts, kernel_sigma, args.epoch_gan, args.dim_gan, trainloader, netG, netD, optimizerG, optimizerD, save_GANimages_InTrain_folder, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan, normalize_count = args.normalize_count, shift_label = shift_count, max_label = max_count, tfboard_writer=tfboard_writer)

    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
    }, Filename_GAN)

    # function for sampling from a trained GAN
    def fn_sampleGAN_with_label(nfake, batch_size):
        images, cellcounts = SampcDCGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, normalize_count = args.normalize_count, shift_label = shift_count, max_label = max_count)
        return images, cellcounts



stop = timeit.default_timer()
print("GAN training finished; Time elapses: {}s".format(stop - start))
