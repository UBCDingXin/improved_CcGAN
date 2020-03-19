'''

2D-Gaussian Simulation

'''

wd = '/home/xin/OneDrive/Working_directory/Continuous_cGAN/Simulation'

import os
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
import pickle
import csv


from utils import *
from models import *
from Train_GAN import *
from Train_cGAN import *
from Train_CcGAN import *



#######################################################################################
'''                                  Settings                                     '''
#######################################################################################
parser = argparse.ArgumentParser(description='Simulation')
'''Overall Settings'''
parser.add_argument('--GAN', type=str, default='CcGAN',
                    choices=['GAN', 'cGAN', 'CcGAN'])
parser.add_argument('--nsim', type=int, default=1,
                    help = "How many times does this experiment need to be repeated?")
parser.add_argument('--seed', type=int, default=2020, metavar='S',
                    help='random seed')
parser.add_argument('--show_visualization', action='store_true', default=False,
                    help='Plot fake samples in 2D coordinate')

''' Data Generation '''
parser.add_argument('--n_gaussians', type=int, default=500,
                    help = "Number of Gaussians (number of angles's).")
parser.add_argument('--n_samp_per_gaussian_train', type=int, default=10) # n_gaussians*n_rsamp_per_gaussian = ntrain
parser.add_argument('--radius', type=float, default=2.0)

''' GAN settings '''
parser.add_argument('--epoch_gan', type=int, default=500) #default 500
parser.add_argument('--lr_gan', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--dim_gan', type=int, default=2,
                    help='Latent dimension of GAN')
parser.add_argument('--batch_size_gan', type=int, default=512, metavar='N',
                    help='input batch size for training GAN')
parser.add_argument('--resumeTrain_gan', type=int, default=0)

parser.add_argument('--threshold_type', type=str, default='hard',
                    choices=['soft', 'hard'])
parser.add_argument('--kernel_sigma', type=float, default=0.01)
parser.add_argument('--kappa', type=float, default=1)

''' Sampling and Evaluation '''
parser.add_argument('--eval', action='store_true', default=False) #evaluation fake samples
parser.add_argument('--n_gaussians_eval', type=int, default=1000) #number of labels for evaluation
parser.add_argument('--n_samp_per_gaussian_eval', type=int, default=10) # number of fake images for each Gaussian
parser.add_argument('--samp_batch_size_eval', type=int, default=100)

args = parser.parse_args()


#--------------------------------
# system
assert torch.cuda.is_available()
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if NGPU>0 else "cpu")
cores= multiprocessing.cpu_count()

#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
np.random.seed(args.seed)

#--------------------------------
# Extra Data Generation Settings
n_gaussians = args.n_gaussians
n_gaussians_eval = args.n_gaussians_eval
n_features = 2 # 2-D
radius = args.radius
angle_grid_train = np.linspace(0, 2*np.pi, n_gaussians) # 12 clock is the start point
angle_grid_eval = np.linspace(0, 2*np.pi, n_gaussians_eval)
sigma_gaussian = 0.05
quality_threshold = sigma_gaussian*4 #good samples are within 4 standard deviation


#--------------------------------
# GAN Settings
epoch_GAN = args.epoch_gan
lr_GAN = args.lr_gan
batch_size_GAN = args.batch_size_gan
dim_GAN = args.dim_gan
plot_in_train = True
gan_Adam_beta1 = 0.5
gan_Adam_beta2 = 0.999


#-------------------------------
# output folders
save_models_folder = wd + '/Output/saved_models/'
os.makedirs(save_models_folder,exist_ok=True)
save_images_folder = wd + '/Output/saved_images/'
os.makedirs(save_images_folder,exist_ok=True)
# save_traincurves_folder = wd + '/Output/Training_loss_fig/'
# os.makedirs(save_traincurves_folder,exist_ok=True)
save_GANimages_InTrain_folder = wd + '/Output/saved_images/' + args.GAN + '_InTrain'
os.makedirs(save_GANimages_InTrain_folder,exist_ok=True)
save_objects_folder = wd + '/Output/saved_objects'
os.makedirs(save_objects_folder,exist_ok=True)


#######################################################################################
'''                               Start Experiment                                 '''
#######################################################################################
#---------------------------------
# sampler for target distribution
def generate_data(n_samp_per_gaussian, angle_grid):
    return sampler_CircleGaussian(n_samp_per_gaussian, angle_grid, radius = radius, sigma = sigma_gaussian, dim = n_features)

prop_recovered_modes = np.zeros(args.nsim) # num of recovered modes diveded by num of modes
prop_good_samples = np.zeros(args.nsim) # num of good fake samples diveded by num of all fake samples


print("\n Begin The Experiment. Sample from a GAN! >>>")
start = timeit.default_timer()
for nSim in range(args.nsim):
    print("Round %s" % (nSim))
    np.random.seed(nSim) #set seed for current simulation

    ###############################################################################
    # Data generation and dataloaders
    ###############################################################################
    samples_train, angles_train, means_train = generate_data(args.n_samp_per_gaussian_train, angle_grid_train)
    samples_plot_in_train, _, _ = generate_data(args.n_samp_per_gaussian_eval, angle_grid_eval)

    if args.GAN == "cGAN": #treated as classification; convert angles to class labels
        angles_train = angles_train.astype(np.single)
        unique_angles_train = np.sort(np.array(list(set(angles_train))))
        angle2class = dict() #convert angle to class label
        class2angle = dict() #convert class label to angle
        for i in range(len(unique_angles_train)):
            angle2class[unique_angles_train[i]]=i
            class2angle[i] = unique_angles_train[i]
        angles_temp = -1*np.ones(len(angles_train))

        for i in range(len(angles_train)):
            angles_temp[i] = angle2class[angles_train[i]]
        assert np.sum(angles_temp<0)==0
        angles_train = angles_temp
        del angles_temp; gc.collect()
        unique_angles_train = np.sort(np.array(list(set(angles_train)))).astype(np.int)
        assert len(unique_angles_train) == n_gaussians
    else:
        # angles_train = angles_train
        angles_train = angles_train/(2*np.pi)


    train_dataset = custom_dataset(samples_train, angles_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_gan, shuffle=True, num_workers=0)


    ###############################################################################
    # Train a GAN model
    ###############################################################################
    #----------------------------------------------
    # GAN
    if args.GAN == "GAN":
        Filename_GAN = save_models_folder + '/ckpt_' + args.GAN + '_epoch_' + str(args.epoch_gan) + '_SEED_' + str(args.seed) + '_nSim_' + str(nSim)
        if not os.path.isfile(Filename_GAN):
            netG = generator(ngpu=NGPU, nz=dim_GAN, out_dim=n_features)
            netD = discriminator(ngpu=NGPU, input_dim = n_features)
            criterion = nn.BCELoss()
            optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_GAN, betas=(gan_Adam_beta1, gan_Adam_beta2))
            optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_GAN, betas=(gan_Adam_beta1, gan_Adam_beta2))

            # Start training
            netG, netD, optimizerG, optimizerD = train_GAN(epoch_GAN, dim_GAN, train_dataloader, netG, netD, optimizerG, optimizerD, criterion, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan, device=device, plot_in_train=plot_in_train, save_images_folder = save_GANimages_InTrain_folder, samples_tar = samples_plot_in_train)

            # store model
            torch.save({
                'netG_state_dict': netG.state_dict(),
            }, Filename_GAN)
        else:
            print("Loading pre-trained generator >>>")
            checkpoint = torch.load(Filename_GAN)
            netG = generator(ngpu=NGPU, nz=dim_GAN, out_dim=n_features).to(device)
            netG.load_state_dict(checkpoint['netG_state_dict'])

        # function for sampling from a trained GAN
        def fn_sampleGAN(nfake, batch_size):
            return SampGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)

    #----------------------------------------------
    # cGAN: treated as a classification dataset
    elif args.GAN == "cGAN":
        Filename_GAN = save_models_folder + '/ckpt_' + args.GAN + '_epoch_' + str(args.epoch_gan) + '_SEED_' + str(args.seed) + '_nSim_' + str(nSim)
        if not os.path.isfile(Filename_GAN):
            netG = cond_generator(ngpu=NGPU, nz=dim_GAN, out_dim=n_features, num_classes=n_gaussians)
            netD = cond_discriminator(ngpu=NGPU, input_dim = n_features, num_classes=n_gaussians)

            criterion = nn.BCELoss()
            optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_GAN, betas=(gan_Adam_beta1, gan_Adam_beta2))
            optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_GAN, betas=(gan_Adam_beta1, gan_Adam_beta2))
            # Start training
            netG, netD, optimizerG, optimizerD = train_cGAN(epoch_GAN, dim_GAN, train_dataloader, netG, netD, optimizerG, optimizerD, criterion, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan, device=device, plot_in_train=plot_in_train, save_images_folder = save_GANimages_InTrain_folder, samples_tar = samples_plot_in_train, num_classes=n_gaussians)

            # store model
            torch.save({
                'netG_state_dict': netG.state_dict(),
            }, Filename_GAN)
        else:
            print("Loading pre-trained generator >>>")
            checkpoint = torch.load(Filename_GAN)
            netG = cond_generator(ngpu=NGPU, nz=dim_GAN, out_dim=n_features, num_classes=n_gaussians).to(device)
            netG.load_state_dict(checkpoint['netG_state_dict'])

        # function for sampling from a trained GAN
        def fn_sampleGAN(nfake, batch_size):
            fake_samples, _ = SampcGAN(netG, class2label=class2angle, GAN_Latent_Length = dim_GAN, NFAKE = nfake, batch_size = batch_size, num_classes = n_gaussians, device=device)
            return fake_samples

        def fn_sampleGAN_given_label(nfake, angle, batch_size):
            angle = (angle*2*np.pi).astype(np.single) #back to original scale of angle [0, 2*pi]
            fake_images, fake_angles = SampcGAN_given_label(netG, angle, unique_labels=unique_angles_train, label2class=angle2class, GAN_Latent_Length = dim_GAN, NFAKE = nfake, batch_size = batch_size, device=device)
            fake_angles = fake_angles/(2*np.pi) #convert to [0,1]
            return fake_images, fake_angles

    #----------------------------------------------
    # Concitnuous cGAN
    elif args.GAN == "CcGAN":
        Filename_GAN = save_models_folder + '/ckpt_' + args.GAN + '_epoch_' + str(args.epoch_gan) + '_SEED_' + str(args.seed) + '_' + args.threshold_type + '_' + str(args.kernel_sigma) + '_' + str(args.kappa) + '_nSim_' + str(nSim)
        if not os.path.isfile(Filename_GAN):
            netG = cont_cond_generator(ngpu=NGPU, nz=dim_GAN, out_dim=n_features)
            netD = cont_cond_discriminator(ngpu=NGPU, input_dim = n_features)

            optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_GAN, betas=(gan_Adam_beta1, gan_Adam_beta2))
            optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_GAN, betas=(gan_Adam_beta1, gan_Adam_beta2))

            # Start training
            netG, netD, optimizerG, optimizerD = train_CcGAN(args.kernel_sigma, args.threshold_type, args.kappa, epoch_GAN, dim_GAN, train_dataloader, netG, netD, optimizerG, optimizerD, save_images_folder=save_GANimages_InTrain_folder, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan, device=device, plot_in_train = plot_in_train, samples_tar = samples_plot_in_train, angle_grid = angle_grid_eval)

            # store model
            torch.save({
                'netG_state_dict': netG.state_dict(),
            }, Filename_GAN)
        else:
            print("Loading pre-trained generator >>>")
            checkpoint = torch.load(Filename_GAN)
            netG = cont_cond_generator(ngpu=NGPU, nz=dim_GAN, out_dim=n_features).to(device)
            netG.load_state_dict(checkpoint['netG_state_dict'])

        def fn_sampleGAN_given_label(nfake, label, batch_size):
            fake_samples, fake_angles = SampCcGAN_given_label(netG, label, path=None, dim_GAN = dim_GAN, NFAKE = nfake, batch_size = batch_size, device=device)
            return fake_samples, fake_angles

stop = timeit.default_timer()
print("GAN training finished; Time elapses: {}s".format(stop - start))
