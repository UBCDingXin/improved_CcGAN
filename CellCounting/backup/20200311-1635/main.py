import os
wd = '/home/xin/OneDrive/Working_directory/Continuous_cGAN/CellCounting'

os.chdir(wd)
import sys
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
from Train_Continuous_cDCGAN import *
from eval_metrics import cal_FID, cal_labelscore

#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
parser = argparse.ArgumentParser(description='Density-ratio based sampling for GANs')

parser.add_argument('--num_imgs_per_count', type=int, default=10, metavar='N',
                    help='number of images for each cell count')
parser.add_argument('--img_size', type=int, default=64, metavar='N',
                    choices=[64,128])
parser.add_argument('--start_count', type=int, default=0, metavar='N')
parser.add_argument('--end_count', type=int, default=300, metavar='N')
parser.add_argument('--stepsize_count', type=int, default=2, metavar='N')
parser.add_argument('--show_real_imgs', action='store_true', default=False)

parser.add_argument('--GAN', type=str, default='Continuous_cDCGAN',
                    choices=['DCGAN', 'cDCGAN', 'Continuous_cDCGAN'])
parser.add_argument('--seed', type=int, default=2019, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--transform', action='store_true', default=False,
                    help='rotate or flip images for GAN training')
# parser.add_argument('--normalize_count', action='store_true', default=False,
#                     help='normalize cell counts')


parser.add_argument('--kernel_sigma', type=float, default=0.1)
parser.add_argument('--threshold_type', type=str, default='soft',
                    choices=['soft', 'hard'])
parser.add_argument('--kappa', type=float, default=1)
parser.add_argument('--b_int_digits', type=int, default=16,
                    help='How many digits used to represent the integer part of a label')
parser.add_argument('--b_dec_digits', type=int, default=16,
                    help='How many digits used to represent the decimal part of a label')


parser.add_argument('--epoch_gan', type=int, default=500)
parser.add_argument('--lr_g_gan', type=float, default=1e-4,
                    help='learning rate for generator')
parser.add_argument('--lr_d_gan', type=float, default=2e-4,
                    help='learning rate for discriminator')
parser.add_argument('--dim_gan', type=int, default=128,
                    help='Latent dimension of GAN')
parser.add_argument('--batch_size_gan', type=int, default=64, metavar='N',
                    help='input batch size for training GAN')
parser.add_argument('--resumeTrain_gan', type=int, default=0)


parser.add_argument('--samp_batch_size', type=int, default=100)
parser.add_argument('--nfake', type=int, default=50000)
parser.add_argument('--comp_FID', action='store_true', default=False)
parser.add_argument('--comp_LS', action='store_true', default=False)
# parser.add_argument('--num_eval_labels', type=int, default=200)

args = parser.parse_args()

print("{}, Sigma is {}, Kappa is {}".format(args.threshold_type, args.kernel_sigma, args.kappa))


#-----------------------------
# images
NC = 1 #number of channels
IMG_SIZE = 64

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
if args.GAN in ['Continuous_cDCGAN']:
    save_GANimages_InTrain_folder = wd + '/Output/saved_images/' + args.GAN + '_' + args.threshold_type + '_' + str(args.kernel_sigma) + '_' + str(args.kappa) + '_InTrain/'
else:
    save_GANimages_InTrain_folder = wd + '/Output/saved_images/' + args.GAN + '_InTrain/'
os.makedirs(save_GANimages_InTrain_folder, exist_ok=True)
save_traincurves_folder = wd + '/Output/Training_loss_fig'
os.makedirs(save_traincurves_folder, exist_ok=True)



#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
# data loader
h5py_file = wd+'/data/Cell300_' + str(args.img_size) + 'x' + str(args.img_size) + '.h5'
hf = h5py.File(h5py_file, 'r')
counts = hf['CellCounts'][:]
counts = counts.astype(np.float)
images = hf['IMGs_grey'][:]
hf.close()

### show some real  images
if args.show_real_imgs:
    unique_counts_show = sorted(list(set(counts)))
    nrow = len(unique_counts_show); ncol = 10
    images_show = np.zeros((nrow*ncol, images.shape[1], images.shape[2], images.shape[3]))
    for i in range(nrow):
        curr_label = unique_counts_show[i]
        indx_curr_label = np.where(counts==curr_label)[0:ncol]
        images_show[i,:,:,:] = images[indx_curr_label]
    print(images_show.shape)
    images_show = (images_show/255.0-0.5)/0.5
    images_show = torch.from_numpy(images_show)
    save_image(images_show.data, save_images_folder +'/real_images.png', nrow=n_row, normalize=True)
    sys.exist()



#############
# images for evaluation
images_eval = images

#############3
# images for training GAN
# for each cell count select n_imgs_per_cellcount images
n_imgs_per_cellcount = args.num_imgs_per_count
# unique_cellcounts = list(set(counts))
# n_unique_cellcount = len(unique_cellcounts)
selected_cellcounts = np.arange(args.start_count, args.end_count+1, args.stepsize_count)
n_unique_cellcount = len(selected_cellcounts)

images_subset = np.zeros((n_imgs_per_cellcount*n_unique_cellcount, 1, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
counts_subset = np.zeros(n_imgs_per_cellcount*n_unique_cellcount)
for i in range(n_unique_cellcount):
    curr_cellcount = selected_cellcounts[i]
    index_curr_cellcount = np.where(counts==curr_cellcount)[0]

    if i == 0:
        images_subset = images[index_curr_cellcount[0:n_imgs_per_cellcount]]
        counts_subset = counts[index_curr_cellcount[0:n_imgs_per_cellcount]]
    else:
        images_subset = np.concatenate((images_subset, images[index_curr_cellcount[0:n_imgs_per_cellcount]]), axis=0)
        counts_subset = np.concatenate((counts_subset, counts[index_curr_cellcount[0:n_imgs_per_cellcount]]))
# for i
images = images_subset
counts = counts_subset
del images_subset, counts_subset; gc.collect()

print("Number of images: %d" % len(images))

if args.GAN == "cDCGAN": #treated as classification; convert cell counts to class labels
    unique_counts = np.sort(np.array(list(set(counts)))).astype(np.int)
    count2class = dict() #convert count to class label
    class2count = dict() #convert class label to count
    for i in range(len(unique_counts)):
        count2class[unique_counts[i]]=i
        class2count[i] = unique_counts[i]
    counts_new = -1*np.ones(len(counts))
    for i in range(len(counts)):
        counts_new[i] = count2class[counts[i]]
    assert np.sum(counts_new<0)==0
    counts = counts_new
    del counts_new; gc.collect()
    unique_labels = np.sort(np.array(list(set(counts)))).astype(np.int)
else:
    counts /= args.end_count # normalize to [0,1]





if args.transform:
    trainset = IMGs_dataset(images, counts, normalize=True, rotate=True, degrees = [90,180,270], hflip = True, vflip = True)
else:
    trainset = IMGs_dataset(images, counts, normalize=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_gan, shuffle=True, num_workers=8)






#######################################################################################
'''                                    GAN training                                 '''
#######################################################################################
# Filename_GAN = save_models_folder + '/ckpt_' + args.GAN + '_epoch_' + str(args.epoch_gan) + '_SEED_' + str(args.seed) + '_' + args.threshold_type + '_' + str(args.kernel_sigma) + '_' + str(args.kappa)

start = timeit.default_timer()
print("\n Begin Training %s:" % args.GAN)
#----------------------------------------------
# DCGAN
if args.GAN == "DCGAN":
    Filename_GAN = save_models_folder + '/ckpt_' + args.GAN + '_epoch_' + str(args.epoch_gan) + '_SEED_' + str(args.seed)
    if not os.path.isfile(Filename_GAN):
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
    else:
        print("Loading pre-trained generator >>>")
        checkpoint = torch.load(Filename_GAN)
        netG = cnn_generator(NGPU, args.dim_gan).to(device)
        netG.load_state_dict(checkpoint['netG_state_dict'])

    # function for sampling from a trained GAN
    def fn_sampleGAN(nfake, batch_size):
        fake_images = SampDCGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
        return fake_images

#----------------------------------------------
# cDCGAN: treated as a classification dataset
elif args.GAN == "cDCGAN":
    Filename_GAN = save_models_folder + '/ckpt_' + args.GAN + '_epoch_' + str(args.epoch_gan) + '_SEED_' + str(args.seed)
    if not os.path.isfile(Filename_GAN):
        print("There are {} unique cell counts".format(len(unique_labels)))

        netG = cond_cnn_generator(args.dim_gan, num_classes=len(unique_labels))
        netD = cond_cnn_discriminator(True, num_classes=len(unique_labels))
        if args.resumeTrain_gan==0:
            netG.apply(weights_init)
            netD.apply(weights_init)
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)
        criterion = nn.BCELoss()
        optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g_gan, betas=(ADAM_beta1, ADAM_beta2))
        optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr_d_gan, betas=(ADAM_beta1, ADAM_beta2))

        # Start training
        netG, netD, optimizerG, optimizerD = train_cDCGAN(unique_labels, args.epoch_gan, args.dim_gan, trainloader, netG, netD, optimizerG, optimizerD, criterion, save_GANimages_InTrain_folder, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan)

        # store model
        torch.save({
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
        }, Filename_GAN)
    else:
        print("Loading pre-trained generator >>>")
        checkpoint = torch.load(Filename_GAN)
        netG = cond_cnn_generator(args.dim_gan, num_classes=len(unique_labels)).to(device)
        netG = nn.DataParallel(netG)
        netG.load_state_dict(checkpoint['netG_state_dict'])

    # function for sampling from a trained GAN
    def fn_sampleGAN(nfake, batch_size):
        fake_images, fake_counts = SampcDCGAN(netG, class2label=class2count, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, num_classes = len(unique_labels), device=device)
        fake_counts = fake_counts/args.end_count
        return fake_images, fake_counts

    def fn_sampleGAN_given_label(nfake, label, batch_size):
        label = int(label*args.end_count) #back to original scale of cell count
        fake_images, fake_counts = SampcDCGAN_given_label(netG, label, unique_labels=unique_counts, label2class=count2class, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
        fake_counts = fake_counts/args.end_count
        return fake_images, fake_counts

#----------------------------------------------
# Concitnuous cDCGAN
elif args.GAN == "Continuous_cDCGAN":
    Filename_GAN = save_models_folder + '/ckpt_' + args.GAN + '_epoch_' + str(args.epoch_gan) + '_SEED_' + str(args.seed) + '_' + args.threshold_type + '_' + str(args.kernel_sigma) + '_' + str(args.kappa)
    if not os.path.isfile(Filename_GAN):
        netG = cont_cond_cnn_generator(args.dim_gan)
        netD = cont_cond_cnn_discriminator(True)
        if args.resumeTrain_gan==0:
            netG.apply(weights_init)
            netD.apply(weights_init)
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)
        optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g_gan, betas=(ADAM_beta1, ADAM_beta2))
        optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr_d_gan, betas=(ADAM_beta1, ADAM_beta2))

        tfboard_writer = SummaryWriter(wd+'/Output/saved_logs')

        # Start training
        netG, netD, optimizerG, optimizerD = train_Continuous_cDCGAN(counts, args.kernel_sigma, args.threshold_type, args.kappa, args.epoch_gan, args.dim_gan, trainloader, netG, netD, optimizerG, optimizerD, save_GANimages_InTrain_folder, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan, tfboard_writer=tfboard_writer)

        # store model
        torch.save({
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
        }, Filename_GAN)

    else:
        print("Loading pre-trained generator >>>")
        checkpoint = torch.load(Filename_GAN)
        netG = cont_cond_cnn_generator(args.dim_gan).to(device)
        netG = nn.DataParallel(netG)
        netG.load_state_dict(checkpoint['netG_state_dict'])

    def fn_sampleGAN_given_label(nfake, label, batch_size):
        fake_images, fake_counts = SampCcDCGAN_given_label(netG, label, path=None, dim_GAN = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
        return fake_images, fake_counts

stop = timeit.default_timer()
print("GAN training finished; Time elapses: {}s".format(stop - start))


#######################################################################################
'''                                  Evaluation                                     '''
#######################################################################################
if args.comp_FID or args.comp_LS:
    if args.comp_FID:
        PreNetFID = ResNet50(ngpu = NGPU, is_label_positive=True).to(device)
        Filename_PreCNNForEvalGANs = save_models_folder + '/ckpt_PreCNNForEvalGANs_ResNet50_epoch_100_SEED_2020_Transformation_True_Cell_200'
        checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
        PreNetFID.load_state_dict(checkpoint_PreNet['net_state_dict'])
    if args.comp_LS:
        PreNetLS = ResNet50(ngpu = NGPU, is_label_positive=True).to(device)
        Filename_PreCNNForEvalGANs = save_models_folder + '/ckpt_PreCNNForEvalGANs_ResNet50_epoch_100_SEED_2020_Transformation_True_Cell_200'
        checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
        PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])

    if args.GAN == 'DCGAN':
        fake_images = fn_sampleGAN(args.nfake, args.samp_batch_size)
        if args.comp_FID:
            # FID
            real_images_norm = (images_eval/255.0-0.5)/0.5
            del images_eval; gc.collect()
            indx_shuffle_real = np.arange(len(real_images_norm)); np.random.shuffle(indx_shuffle_real)
            indx_shuffle_fake = np.arange(args.nfake); np.random.shuffle(indx_shuffle_fake)
            FID = cal_FID(PreNetFID, real_images_norm[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size = 100, resize = None)
            print("\n DCGAN FID: {}".format(FID))
    else:
        #####################
        # generate nfake images
        print("Start sampling {} fake images from GAN >>>".format(args.nfake))
        eval_labels = np.arange(args.start_count, args.end_count + 1)/args.end_count
        num_eval_labels = len(eval_labels)

        # print(eval_labels.reshape(-1))

        for i in tqdm(range(num_eval_labels)):
            curr_label = eval_labels[i]
            if i == 0:
                curr_nfake = args.nfake - (args.nfake//num_eval_labels) * (num_eval_labels-1)
                curr_fake_images, curr_fake_counts = fn_sampleGAN_given_label(curr_nfake, curr_label, args.samp_batch_size)
                fake_images = curr_fake_images
                fake_labels_assigned = curr_fake_counts.reshape(-1)
            else:
                curr_nfake = args.nfake//num_eval_labels
                curr_fake_images, curr_fake_counts = fn_sampleGAN_given_label(curr_nfake, curr_label, args.samp_batch_size)
                fake_images = np.concatenate((fake_images, curr_fake_images), axis=0)
                fake_labels_assigned = np.concatenate((fake_labels_assigned, curr_fake_counts.reshape(-1)))
        assert len(fake_images) == args.nfake
        assert len(fake_labels_assigned) == args.nfake
        print("End sampling!")

        #####################
        # FID
        if args.comp_FID:
            real_images_norm = (images_eval/255.0-0.5)/0.5
            del images_eval; gc.collect()
            indx_shuffle_real = np.arange(len(real_images_norm)); np.random.shuffle(indx_shuffle_real)
            indx_shuffle_fake = np.arange(args.nfake); np.random.shuffle(indx_shuffle_fake)
            FID = cal_FID(PreNetFID, real_images_norm[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size = 100, resize = None)
        else:
            FID = 'NaN'

        #####################
        # Label score (LS)
        if args.comp_LS:
            labelscore = cal_labelscore(PreNetLS, fake_images, fake_labels_assigned, batch_size = 100, resize = None)
            labelscore = labelscore*args.end_count #back to original scale
        else:
            labelscore = 'NaN'

        print("\n {} FID: {}; LS: {}.".format(args.GAN, FID, labelscore))
