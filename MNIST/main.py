import os
wd = '/home/xin/OneDrive/Working_directory/Continuous_cGAN/MNIST'

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
from sklearn.linear_model import LogisticRegression
import multiprocessing
from multiprocessing import Pool
from scipy.stats import ks_2samp
import h5py

from utils import *
from models import *
from Train_DCGAN import *
from Train_cDCGAN import *
from eval_metrics import FID, FID_RAW, IS_RAW


#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
parser = argparse.ArgumentParser(description='Density-ratio based sampling for GANs')

parser.add_argument('--GAN', type=str, default='DCGAN',
                    choices=['DCGAN', 'cDCGAN'])
parser.add_argument('--seed', type=int, default=2019, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--transform', action='store_true', default=False,
                    help='transform images for GAN training')

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
parser.add_argument('--comp_ISFID', action='store_true', default=False)

args = parser.parse_args()


#--------------------------------
# system
NGPU = torch.cuda.device_count()
device = torch.device("cuda")
NCPU = multiprocessing.cpu_count()
cudnn.benchmark = True # For fast training


#-------------------------------
# GAN
N_CLASS = 10
NC = 1 #number of channels
IMG_SIZE = 28
ResumeEpoch_gan = args.resumeTrain_gan
resize = (299, 299)
ADAM_beta1 = 0.5 #parameters for ADAM optimizer
ADAM_beta2 = 0.999

#-------------------------------
# sampling parameters
samp_batch_size = args.samp_batch_size #batch size for sampling from GAN or enhanced sampler
NFAKE = args.nfake

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
save_GANimages_InTrain_folder = wd + '/Output/saved_images/'+args.GAN+'_InTrain/'
os.makedirs(save_GANimages_InTrain_folder, exist_ok=True)
save_traincurves_folder = wd + '/Output/Training_loss_fig'
os.makedirs(save_traincurves_folder, exist_ok=True)


#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################

if args.transform:
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
else:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_gan, shuffle=True, num_workers=NCPU)

#######################################################################################
'''                             Train GAN or Load Pre-trained GAN                '''
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
    netG, netD, optimizerG, optimizerD = train_cDCGAN(args.epoch_gan, args.dim_gan, trainloader, netG, netD, optimizerG, optimizerD, criterion, save_GANimages_InTrain_folder, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan, num_classes=N_CLASS)

    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
    }, Filename_GAN)

stop = timeit.default_timer()
print("GAN training finished; Time elapses: {}s".format(stop - start))


###############################################################################
'''                Function for different sampling methods                  '''
###############################################################################
# Load Pre-trained GAN
checkpoint = torch.load(Filename_GAN)
if args.GAN == "cDCGAN":
    netG = cond_cnn_generator(NGPU, args.dim_gan, num_classes=N_CLASS).to(device)
    def fn_sampleGAN(nfake, batch_size):
        images,labels = SampcDCGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
        return images,labels
    def fn_sampleGAN_given_label(nfake, given_label, batch_size):
        images,labels = SampcDCGAN_given_label(netG, given_label, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
        return images,labels
netG.load_state_dict(checkpoint['netG_state_dict'])



###############################################################################
'''                             Compute FID and IS                          '''
###############################################################################
#load pre-trained InceptionV3 (pretrained on CIFAR-10)
PreNetFIDIS = Inception3(num_classes=N_CLASS, aux_logits=True, transform_input=False)
Filename_PreCNNForEvalGANs = save_models_folder + '/ckpt_PreCNNForEvalGANs_InceptionV3_epoch_200_SEED_2019_Transformation_True'
checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
PreNetFIDIS = nn.DataParallel(PreNetFIDIS).cuda()
PreNetFIDIS.load_state_dict(checkpoint_PreNet['net_state_dict'])


if args.comp_ISFID:

    fake_imgs = fn_sampleGAN(NFAKE, samp_batch_size)

    print("\n Computing IS for %s >>> " % (args.GAN))
    (IS_fake_avg, IS_fake_std) = IS_RAW(PreNetFIDIS, fake_imgs, batch_size = 1000, splits=10, NGPU=NGPU, resize=resize)
    print("Inception Scores: %.3f, %.3f" % (IS_fake_avg, IS_fake_std))
