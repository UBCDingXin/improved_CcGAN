print("\n===================================================================================================")


import os
wd = '/home/xin/OneDrive/Working_directory/Continuous_cGAN/UTKFace'

os.chdir(wd)
import sys
import timeit
import torch
from torch import nn
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
from Train_CcGAN import *
from Train_cGAN import *
from eval_metrics import cal_FID, cal_labelscore

#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
parser = argparse.ArgumentParser(description='Density-ratio based sampling for GANs')

parser.add_argument('--GAN', type=str, default='ContcDCGAN',
                    choices=['cDCGAN', 'ContcDCGAN'])
parser.add_argument('--seed', type=int, default=2019, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--img_size', type=int, default=64, metavar='N',
                    choices=[64,128])
parser.add_argument('--max_num_img_per_label', type=int, default=10000, metavar='N')
parser.add_argument('--show_real_imgs', action='store_true', default=False)
parser.add_argument('--visualize_fake_images', action='store_true', default=False)


parser.add_argument('--kernel_sigma', type=float, default=-1.0,
                    help='If kernel_sigma<0, then use rule-of-thumb formula to compute the sigma.')
parser.add_argument('--threshold_type', type=str, default='soft',
                    choices=['soft', 'hard'])
parser.add_argument('--kappa', type=float, default=-1.0)


parser.add_argument('--epoch_gan', type=int, default=2000)
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
parser.add_argument('--epoch_FID_CNN', type=int, default=200)
parser.add_argument('--comp_LS', action='store_true', default=False)
# parser.add_argument('--num_eval_labels', type=int, default=80)
parser.add_argument('--FID_num_classes', type=int, default=20)

args = parser.parse_args()


#-----------------------------
# images
NC = 3 #number of channels
IMG_SIZE = args.img_size

#--------------------------------
# system
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NCPU = multiprocessing.cpu_count()

#-------------------------------
# GAN
ResumeEpoch_gan = args.resumeTrain_gan
ADAM_beta1 = 0.5 #parameters for ADAM optimizer
ADAM_beta2 = 0.999

#-------------------------------
# sampling parameters
samp_batch_size = args.samp_batch_size #batch size for sampling from GAN or enhanced sampler

#-------------------------------
#seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False # For fast training
np.random.seed(args.seed)

#-------------------------------
# output folders
save_models_folder = wd + '/Output/saved_models'
os.makedirs(save_models_folder, exist_ok=True)
save_images_folder = wd + '/Output/saved_images'
os.makedirs(save_images_folder, exist_ok=True)
save_traincurves_folder = wd + '/Output/Training_loss_fig'
os.makedirs(save_traincurves_folder, exist_ok=True)



#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
# data loader
data_filename = wd+'/data/UTKFace_' + str(args.img_size) + 'x' + str(args.img_size) + '.h5'
hf = h5py.File(data_filename, 'r')
labels = hf['labels'][:]
labels = labels.astype(np.float)
images = hf['images'][:]
hf.close()

if args.visualize_fake_images or args.comp_FID or args.comp_LS:
    raw_images = images #backup images;
    raw_labels = labels #backup labels

### show some real  images
if args.show_real_imgs:
    unique_labels_show = sorted(list(set(labels)))
    nrow = len(unique_labels_show); ncol = 10
    images_show = np.zeros((nrow*ncol, images.shape[1], images.shape[2], images.shape[3]))
    for i in range(nrow):
        curr_label = unique_labels_show[i]
        indx_curr_label = np.where(labels==curr_label)[0][0:ncol]
        for j in range(ncol):
            images_show[i*ncol+j,:,:,:] = images[indx_curr_label[j]]
    print(images_show.shape)
    images_show = (images_show/255.0-0.5)/0.5
    images_show = torch.from_numpy(images_show)
    save_image(images_show.data, save_images_folder +'/real_images_grid_{}x{}.png'.format(nrow, ncol), nrow=ncol, normalize=True)

# # subset of UTKFace
# range_selected_labels = np.arange(1, 31)
# n_unique_labels = len(range_selected_labels)
# for i in range(n_unique_labels):
#     curr_label = range_selected_labels[i]
#     index_curr_label = np.where(labels==curr_label)[0]
#     if i == 0:
#         images_subset = images[index_curr_label]
#         labels_subset = labels[index_curr_label]
#     else:
#         images_subset = np.concatenate((images_subset, images[index_curr_label]), axis=0)
#         labels_subset = np.concatenate((labels_subset, labels[index_curr_label]))
# # for i
# images = images_subset
# labels = labels_subset
# del images_subset, labels_subset; gc.collect()


# for each age, take no more than args.max_num_img_per_label images
image_num_threshold = args.max_num_img_per_label
print("\n Original set has {} images; For each age, take no more than {} images>>>".format(len(images), image_num_threshold))
unique_labels_tmp = np.sort(np.array(list(set(labels))))
for i in tqdm(range(len(unique_labels_tmp))):
    indx_i = np.where(labels == unique_labels_tmp[i])[0]
    if len(indx_i)>image_num_threshold:
        np.random.shuffle(indx_i)
        indx_i = indx_i[0:image_num_threshold]
    if i == 0:
        sel_indx = indx_i
    else:
        sel_indx = np.concatenate((sel_indx, indx_i))
images = images[sel_indx]
labels = labels[sel_indx]
print("{} images left.".format(len(images)))


hist_filename = wd + "/histogram_unnormalized_age_" + str(args.img_size) + 'x' + str(args.img_size)
num_bins = len(list(set(labels)))
plt.figure()
plt.hist(labels, num_bins, facecolor='blue', density=False)
plt.savefig(hist_filename)





# normalize labels
print("\n Range of unnormalized labels: ({},{})".format(np.min(labels), np.max(labels)))
max_label = np.max(labels)
if args.GAN == "cDCGAN": #treated as classification; convert ages to class labels
    unique_labels = np.sort(np.array(list(set(labels)))).astype(np.int)
    label2class = dict() #convert age to class label
    class2label = dict() #convert class label to age
    for i in range(len(unique_labels)):
        label2class[unique_labels[i]]=i
        class2label[i] = unique_labels[i]
    labels_new = -1*np.ones(len(labels))
    for i in range(len(labels)):
        labels_new[i] = label2class[labels[i]]
    assert np.sum(labels_new<0)==0
    labels = labels_new
    del labels_new; gc.collect()
    unique_labels = np.sort(np.array(list(set(labels)))).astype(np.int)
else:
    labels /= max_label #normalize to [0,1]

    hist_filename = wd + "/histogram_normalized_age_" + str(args.img_size) + 'x' + str(args.img_size)
    num_bins = len(list(set(labels)))
    plt.figure()
    plt.hist(labels, num_bins, facecolor='blue', density=False)
    plt.savefig(hist_filename)

    print("\n Range of normalized labels: ({},{})".format(np.min(labels), np.max(labels)))

    unique_labels_norm = np.sort(np.array(list(set(labels))))

    if args.kernel_sigma<0:
        std_label = np.std(labels)
        args.kernel_sigma =1.06*std_label*(len(labels))**(-1/5)
        print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")
        print("\n The std of {} labels is {} so the kernel sigma is {}".format(len(labels), std_label, args.kernel_sigma))

    if args.kappa<0:
        n_unique = len(unique_labels_norm)

        diff_list = []
        for i in range(1,n_unique):
            diff_list.append(unique_labels_norm[i] - unique_labels_norm[i-1])
        kappa_base = 2*np.max(np.array(diff_list))
        # kappa_base = 1*np.max(np.array(diff_list))

        if args.threshold_type=="hard":
            args.kappa = kappa_base
        else:
            # args.kappa = min(1/kappa_base**2, 5000)
            args.kappa = 1/kappa_base**2



trainset = IMGs_dataset(images, labels, normalize=True)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_gan, shuffle=True, num_workers=8)



#######################################################################################
'''                                    GAN training                                 '''
#######################################################################################

print("{}, Sigma is {}, Kappa is {}".format(args.threshold_type, args.kernel_sigma, args.kappa))

if args.GAN in ['ContcDCGAN']:
    save_GANimages_InTrain_folder = wd + '/Output/saved_images/' + args.GAN + '_' + args.threshold_type + '_' + str(args.kernel_sigma) + '_' + str(args.kappa) + '_size_{}x{}'.format(args.img_size, args.img_size) + '_InTrain/'
else:
    save_GANimages_InTrain_folder = wd + '/Output/saved_images/' + args.GAN + '_size_{}x{}'.format(args.img_size, args.img_size) + '_InTrain/'
os.makedirs(save_GANimages_InTrain_folder, exist_ok=True)

start = timeit.default_timer()
print("\n Begin Training %s:" % args.GAN)

#----------------------------------------------
# cDCGAN: treated as a classification dataset
if args.GAN in ["cDCGAN"]:
    Filename_GAN = save_models_folder + '/ckpt_' + args.GAN + '_epoch_' + str(args.epoch_gan) + '_SEED_' + str(args.seed) + '_size_{}x{}'.format(args.img_size, args.img_size)
    if not os.path.isfile(Filename_GAN):
        print("There are {} unique labels".format(len(unique_labels)))

        netG = cond_cnn_generator(args.dim_gan, num_classes=len(unique_labels))
        netD = cond_cnn_discriminator(True, num_classes=len(unique_labels))
        # if args.resumeTrain_gan==0:
        #     netG.apply(weights_init)
        #     netD.apply(weights_init)
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)
        criterion = nn.BCELoss()
        optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g_gan, betas=(ADAM_beta1, ADAM_beta2))
        optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr_d_gan, betas=(ADAM_beta1, ADAM_beta2))

        # Start training
        netG, netD, optimizerG, optimizerD = train_cGAN(args.GAN, unique_labels, args.epoch_gan, args.dim_gan, trainloader, netG, netD, optimizerG, optimizerD, criterion, save_GANimages_InTrain_folder, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan)

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
        fake_images, fake_labels = SampcGAN(netG, class2label=class2label, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, img_size=args.img_size, num_classes = len(unique_labels), device=device)
        fake_labels = (fake_labels + np.abs(min_label_before_shift)) / max_label_after_shift
        return fake_images, fake_labels

    def fn_sampleGAN_given_label(nfake, label, batch_size):
        label = int((label * max_label_after_shift) - np.abs(min_label_before_shift)) #back to original scale
        fake_images, fake_labels = SampcDCGAN_given_label(netG, label, unique_labels=unique_labels, label2class=label2class, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, img_size=args.img_size, device=device)
        fake_labels = (fake_labels + np.abs(min_label_before_shift)) / max_label_after_shift
        return fake_images, fake_labels

#----------------------------------------------
# Concitnuous cDCGAN
if args.GAN in ["ContcDCGAN"]:
    Filename_GAN = save_models_folder + '/ckpt_' + args.GAN + '_epoch_' + str(args.epoch_gan) + '_SEED_' + str(args.seed) + '_' + args.threshold_type + '_' + str(args.kernel_sigma) + '_size_{}x{}'.format(args.img_size, args.img_size) + '_' + str(args.kappa)
    if not os.path.isfile(Filename_GAN):
        netG = cont_cond_cnn_generator(nz=args.dim_gan).to(device)
        netD = cont_cond_cnn_discriminator(use_sigmoid = True).to(device)
        # if args.resumeTrain_gan==0:
        #     netG.apply(weights_init)
        #     netD.apply(weights_init)
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)
        optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g_gan, betas=(ADAM_beta1, ADAM_beta2))
        optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr_d_gan, betas=(ADAM_beta1, ADAM_beta2))

        tfboard_writer = SummaryWriter(wd+'/Output/saved_logs')

        # Start training
        netG, netD, optimizerG, optimizerD = train_CcGAN(args.GAN, labels, args.kernel_sigma, args.threshold_type, args.kappa, args.epoch_gan, args.dim_gan, trainloader, netG, netD, optimizerG, optimizerD, save_GANimages_InTrain_folder, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan, tfboard_writer=tfboard_writer)

        # store model
        torch.save({
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
        }, Filename_GAN)

    else:
        print("Loading pre-trained generator >>>")
        checkpoint = torch.load(Filename_GAN)
        netG = cont_cond_cnn_generator(nz=args.dim_gan).to(device)
        netG = nn.DataParallel(netG)
        netG.load_state_dict(checkpoint['netG_state_dict'])

    def fn_sampleGAN_given_label(nfake, label, batch_size):
        fake_images, fake_labels = SampCcGAN_given_label(netG, label, path=None, dim_GAN = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
        return fake_images, fake_labels

stop = timeit.default_timer()
print("GAN training finished; Time elapses: {}s".format(stop - start))


#######################################################################################
'''                                  Evaluation                                     '''
#######################################################################################
if args.comp_FID or args.comp_LS:
    if args.comp_FID:
        PreNetFID = ResNet34_class(num_classes=args.FID_num_classes, ngpu = NGPU).to(device)
        Filename_PreCNNForEvalGANs = save_models_folder + '/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_{}_SEED_2020_img_size_64_num_classes_{}_CVMode_False'.format(args.epoch_FID_CNN, args.FID_num_classes)
        checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
        PreNetFID.load_state_dict(checkpoint_PreNet['net_state_dict'])
    if args.comp_LS:
        PreNetLS = ResNet34_regre(ngpu = NGPU).to(device)
        Filename_PreCNNForEvalGANs = save_models_folder + '/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_{}_SEED_2020_img_size_64_CVMode_False'.format(args.epoch_FID_CNN)
        checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
        PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])

    #####################
    # generate nfake images
    print("Start sampling {} fake images from GAN >>>".format(args.nfake))

    eval_labels_norm = np.arange(1, max_label+1) / max_label
    num_eval_labels = len(eval_labels_norm)

    for i in tqdm(range(num_eval_labels)):
        curr_label = eval_labels_norm[i]
        if i == 0:
            curr_nfake = args.nfake - (args.nfake//num_eval_labels) * (num_eval_labels-1)
            curr_fake_images, curr_fake_labels = fn_sampleGAN_given_label(curr_nfake, curr_label, args.samp_batch_size)
            fake_images = curr_fake_images
            fake_labels_assigned = curr_fake_labels.reshape(-1)
        else:
            curr_nfake = args.nfake//num_eval_labels
            curr_fake_images, curr_fake_labels = fn_sampleGAN_given_label(curr_nfake, curr_label, args.samp_batch_size)
            fake_images = np.concatenate((fake_images, curr_fake_images), axis=0)
            fake_labels_assigned = np.concatenate((fake_labels_assigned, curr_fake_labels.reshape(-1)))
    assert len(fake_images) == args.nfake
    assert len(fake_labels_assigned) == args.nfake
    print("End sampling!")

    # FID
    if args.comp_FID:
        real_images_norm = (raw_images/255.0-0.5)/0.5
        indx_shuffle_real = np.arange(len(real_images_norm)); np.random.shuffle(indx_shuffle_real)
        indx_shuffle_fake = np.arange(args.nfake); np.random.shuffle(indx_shuffle_fake)
        FID = cal_FID(PreNetFID, real_images_norm[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size = 100, resize = None)
    else:
        FID = 'NaN'

    # Label score (LS)
    if args.comp_LS:
        ls_mean, ls_std = cal_labelscore(PreNetLS, fake_images, fake_labels_assigned, batch_size = 100, resize = None)
        ls_mean = ls_mean * max_label #back to original scale
    else:
        ls_mean = 'NaN'
        ls_std = 'NaN'

    # print("\n {} FID: {}({}); LS: {}({}).".format(args.GAN, FID_mean, FID_std, ls_mean, ls_std))
    print("\n {} FID: {}; LS: {}({}).".format(args.GAN, FID, ls_mean, ls_std))



#######################################################################################
'''               Visualize fake images of the trained GAN                          '''
#######################################################################################
if args.visualize_fake_images:
    # First, visualize conditional generation
    ## 10 rows; 3 columns (3 samples for each age)
    n_row = 10
    n_col = 3
    # displayed_normalized_labels = np.linspace(0.05, 1, n_row)
    displayed_labels = [4, 12, 20, 28, 36, 44, 52, 60, 68, 76];
    assert len(displayed_labels) == n_row
    displayed_normalized_labels = displayed_labels/max_label
    ### output fake images from a trained GAN
    if args.GAN == 'ContcDCGAN':
        filename_fake_images = save_images_folder + '/{}_{}_sigma_{}_kappa_{}_fake_images_grid_{}x{}.png'.format(args.GAN, args.threshold_type, args.kernel_sigma, args.kappa, n_row, n_col)
    else:
        filename_fake_images = save_images_folder + '/{}_fake_images_grid_{}x{}.png'.format(args.GAN, n_row, n_col)
    images_show = np.zeros((n_row*n_col, images.shape[1], images.shape[2], images.shape[3]))
    for i_row in range(n_row):
        curr_label = displayed_normalized_labels[i_row]
        for j_col in range(n_col):
            curr_image, _ = fn_sampleGAN_given_label(1, curr_label, 1)
            images_show[i_row*n_col+j_col,:,:,:] = curr_image
    images_show = torch.from_numpy(images_show)
    print(n_row)
    save_image(images_show.data, filename_fake_images, nrow=n_col, normalize=True)







    # ## 3 rows (3 samples); 10 columns
    # n_row = 3
    # n_col = 10
    # displayed_normalized_labels = np.linspace(0.05, 1, n_col)
    # ### output fake images from a trained GAN
    # if args.GAN == 'ContcDCGAN':
    #     filename_fake_images = save_images_folder + '/{}_{}_sigma_{}_kappa_{}_fake_images_grid_{}x{}.png'.format(args.GAN, args.threshold_type, args.kernel_sigma, args.kappa, n_row, n_col)
    # else:
    #     filename_fake_images = save_images_folder + '/{}_fake_images_grid_{}x{}.png'.format(args.GAN, n_row, n_col)
    # images_show = np.zeros((n_row*n_col, images.shape[1], images.shape[2], images.shape[3]))
    # for i_row in range(n_row):
    #     for j_col in range(n_col):
    #         curr_label = displayed_normalized_labels[j_col]
    #         curr_image, _ = fn_sampleGAN_given_label(1, curr_label, 1)
    #         images_show[i_row*n_col+j_col,:,:,:] = curr_image
    # images_show = torch.from_numpy(images_show)
    # save_image(images_show.data, filename_fake_images, nrow=n_col, normalize=True)

    # Second, fix z but increase y; check whether there is a continuous change, only for CcGAN
    if args.GAN == 'ContcDCGAN':
        n_continuous_labels = 10
        normalized_continuous_labels = np.linspace(0.05, 1, n_continuous_labels)
        z = torch.randn(1, args.dim_gan, dtype=torch.float).to(device)
        continuous_images_show = torch.zeros(n_continuous_labels, NC, IMG_SIZE, IMG_SIZE, dtype=torch.float)

        netG.eval()
        with torch.no_grad():
            for i in range(n_continuous_labels):
                y = np.ones(1) * normalized_continuous_labels[i]
                y = torch.from_numpy(y).type(torch.float).view(-1,1).to(device)
                fake_image_i = netG(z, y)
                continuous_images_show[i,:,:,:] = fake_image_i.cpu()

        filename_continous_fake_images = save_images_folder + '/{}_{}_sigma_{}_kappa_{}_continuous_fake_images_grid.png'.format(args.GAN, args.threshold_type, args.kernel_sigma, args.kappa)
        save_image(continuous_images_show.data, filename_continous_fake_images, nrow=n_continuous_labels, normalize=True)

        print("Continuous ys: ", (normalized_continuous_labels*max_label).astype(np.int))
