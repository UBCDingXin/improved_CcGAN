import os
wd = '/home/xin/OneDrive/Working_directory/GAN_DA_Subsampling/CellCounting'

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
import pickle

from utils import weights_init, IMGs_dataset, PlotLoss, SampPreRegGAN
from models import *
from Train_DRE import *
from Train_RegDCGAN import *
from SimpleProgressBar import SimpleProgressBar


#######################################################################################
'''                                   Settings                                      '''
#######################################################################################
parser = argparse.ArgumentParser(description='Density-ratio based sampling for GANs')
'''Overall Settings'''
parser.add_argument('--DA_method', type=str, default='RegDCGAN',
                    choices=['RegDCGAN', 'RegDCGAN+DRE-F-SP+RS', 'RegDCGAN+DRE-F-SP+RS+double'])
parser.add_argument('--seed', type=int, default=2019, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--transform_GAN_and_DRE', action='store_true', default=False,
                    help='rotate or flip images for GAN and DRE training')

''' Data Split '''
parser.add_argument('--nfolds', type=int, default=4)

''' GAN settings '''
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


'''DRE settings'''
## cDRE_F_SP
parser.add_argument('--DR_Net', type=str, default='MLP5',
                    choices=['MLP3', 'MLP5', 'MLP7', 'MLP9'],
                    help='DR Model') # DRE in Feature Space
parser.add_argument('--PreCNN_DR', type=str, default='ResNet34',
                    choices=['ResNet34'],
                    help='Pre-trained CNN for DRE in Feature Space; Candidate: ResNetXX')
parser.add_argument('--epoch_pretrainCNN', type=int, default=200)
parser.add_argument('--transform_PreCNN_DR', action='store_true', default=True,
                    help='rotate or flip images for CNN training')
parser.add_argument('--epoch_DRE', type=int, default=2000)
parser.add_argument('--base_lr_DRE', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--not_decay_lr_DRE', action='store_true', default=False,
                    help='not decay learning rate')
parser.add_argument('--batch_size_DRE', type=int, default=64, metavar='N',
                    help='input batch size for training DRE')
parser.add_argument('--lambda_DRE', type=float, default=0,
                    help='penalty in DRE')
parser.add_argument('--lambda_DRE2', type=float, default=0,
                    help='penalty in 2nd DRE')
parser.add_argument('--weightdecay_DRE', type=float, default=1e-4,
                    help='weight decay in DRE')
parser.add_argument('--resumeTrain_DRE', type=int, default=0)
parser.add_argument('--DR_final_ActFn', type=str, default='ReLU',
                    help='Final layer of the Density-ratio model; Candidiate: Softplus or ReLU')


'''Sampling Settings'''
parser.add_argument('--samp_batch_size', type=int, default=10)

'''Data augmentation setting'''
parser.add_argument('--da_output', action='store_true', default=False,
                    help='Output fake images for data augmentation?')
parser.add_argument('--da_nfake', type=int, default=10000)
args = parser.parse_args()

#-----------------------------
# images
NC = 1 #number of channels
IMG_SIZE = 64
n_all = 200
nvalid = int(n_all/args.nfolds)
ntrain = n_all-nvalid

#--------------------------------
# system
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NCPU = multiprocessing.cpu_count()
# NCPU = 0
cudnn.benchmark = True # For fast training

#-------------------------------
# GAN
ResumeEpoch_gan = args.resumeTrain_gan
ADAM_beta1 = 0.5 #parameters for ADAM optimizer
ADAM_beta2 = 0.999

#-------------------------------
# sampling parameters
samp_batch_size = args.samp_batch_size #batch size for sampling from GAN or enhanced sampler
DR_comp_batch_size = 10
assert samp_batch_size>=DR_comp_batch_size

#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
np.random.seed(args.seed)

#-------------------------------
# output folders
save_models_folder = wd + '/Output/saved_models'
if not os.path.exists(save_models_folder):
    os.makedirs(save_models_folder)
save_images_folder = wd + '/Output/saved_images'
if not os.path.exists(save_images_folder):
    os.makedirs(save_images_folder)
save_GANimages_InTrain_folder = wd + '/Output/saved_images/RegDCGAN_InTrain/'
if not os.path.exists(save_GANimages_InTrain_folder):
    os.makedirs(save_GANimages_InTrain_folder)
save_traincurves_folder = wd + '/Output/Training_loss_fig'
if not os.path.exists(save_traincurves_folder):
    os.makedirs(save_traincurves_folder)


#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
# data loader
h5py_file = wd+'/data/VGG_dataset_64x64.h5'
hf = h5py.File(h5py_file, 'r')
counts_all = hf['CellCounts'][:]
# images_all = hf['IMGs_rgb'][:]
images_all = hf['IMGs_grey'][:]
hf.close()
# load the data split in the CV
filename_CV_datasplit = "./data/VGG_dataset_CV_datasplit_NFOLDS_" + str(args.nfolds) + ".pickle"
with open(filename_CV_datasplit, 'rb') as pf:
    CV_datasplit_dict = pickle.load(pf)
indx_train_CV = CV_datasplit_dict['indx_train_CV']
indx_valid_CV = CV_datasplit_dict['indx_valid_CV']



#######################################################################################
'''                                    Start CV                                    '''
#######################################################################################
for nround in range(args.nfolds):
    print("CV Round %d" % nround)

    save_GANimages_InTrain_folder = wd + '/Output/saved_images/RegDCGAN_InTrain/' + str(nround) + '/'
    if not os.path.exists(save_GANimages_InTrain_folder):
        os.makedirs(save_GANimages_InTrain_folder)

    # split data into a training set and a validation set
    indx_train = indx_train_CV[nround]
    indx_valid = indx_valid_CV[nround]
    images_train = images_all[indx_train]
    counts_train = counts_all[indx_train]
    images_valid = images_all[indx_valid]
    counts_valid = counts_all[indx_valid]

    #normalize count
    mean_count_train = np.mean(counts_train)
    std_count_train = np.std(counts_train)
    counts_train = (counts_train-mean_count_train)/std_count_train
    counts_valid = (counts_valid-mean_count_train)/std_count_train

    # dataloader
    ## training set
    if args.transform_GAN_and_DRE:
        trainset = IMGs_dataset(images_train, counts_train, normalize=True, rotate=True, degrees = [90,180,270], hflip = True, vflip = True)
    else:
        trainset = IMGs_dataset(images_train, counts_train, normalize=True)
    trainloader_GAN = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_gan, shuffle=True, num_workers=8)
    trainloader_DRE = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_DRE, shuffle=True, num_workers=8)
    ## validation set
    validset = IMGs_dataset(images_valid, counts_valid, normalize=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=100, shuffle=False, num_workers=8)


    #---------------------------------------------------------------------------------------
    '''                                    Train GAN                                      '''
    #---------------------------------------------------------------------------------------
    Filename_GAN = save_models_folder + '/ckpt_RegDCGAN_epoch_' + str(args.epoch_gan) + '_SEED_' + str(args.seed) + '_NFOLDS_' + str(args.nfolds) + '_nround_' + str(nround)

    if not os.path.isfile(Filename_GAN):
        start = timeit.default_timer()
        print("\n Begin Training GAN:")
        #model initialization
        netG = cnn_generator(NGPU, args.dim_gan)
        netG.apply(weights_init)
        netD = cnn_discriminator(True, NGPU)
        netD.apply(weights_init)
        criterion = nn.BCELoss()
        optimizerG = torch.optim.Adam(netG.parameters(), lr=args.lr_g_gan, betas=(ADAM_beta1, ADAM_beta2))
        optimizerD = torch.optim.Adam(netD.parameters(), lr=args.lr_d_gan, betas=(ADAM_beta1, ADAM_beta2))

        # Start training
        netG, netD, optimizerG, optimizerD = train_RegDCGAN(args.epoch_gan, args.dim_gan, trainloader_GAN, netG, netD, optimizerG, optimizerD, criterion, save_GANimages_InTrain_folder, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan)

        # store model
        torch.save({
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
        }, Filename_GAN)

        stop = timeit.default_timer()
        print("GAN training finished! Time elapses: {}s".format(stop - start))
    else:
        print("\n Load pre-trained GAN:")
        checkpoint = torch.load(Filename_GAN)
        netG = cnn_generator(NGPU, args.dim_gan).to(device)
        netG.load_state_dict(checkpoint['netG_state_dict'])

    # function for sampling from a trained RegDCGAN
    def fn_sampleGAN(nfake, batch_size):
        images, counts = SampRegDCGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)
        return images, counts

    #---------------------------------------------------------------------------------------
    '''                        Different DA methods                            '''
    #---------------------------------------------------------------------------------------
    # RegDCGAN
    if args.DA_method=='RegDCGAN' and args.da_output: #generate fake images from 10 DCGANs
        fake_imgs, fake_counts = fn_sampleGAN(args.da_nfake, samp_batch_size)
    #---------------------------------------------------------------------------------------
    # RegDCGAN
    elif args.DA_method in ['RegDCGAN+DRE-F-SP+RS', 'RegDCGAN+DRE-F-SP+RS+double']:
        # a necessary function for DRE-F-SP
        def CNN_net_init(Pretrained_CNN_Name, NGPU):
            if Pretrained_CNN_Name == "ResNet18":
                net = ResNet18_extract(ngpu = NGPU)
            elif Pretrained_CNN_Name == "ResNet34":
                net = ResNet34_extract(ngpu = NGPU)
            elif Pretrained_CNN_Name == "ResNet50":
                net = ResNet50_extract(ngpu = NGPU)
            elif Pretrained_CNN_Name == "ResNet101":
                net = ResNet101_extract(ngpu = NGPU)

            net_name = 'PreCNNForDRE_' + Pretrained_CNN_Name #get the net's name
            return net, net_name

        # Construct a function to compute density-ratio
        def DR_net_init(DR_net_name):
            if DR_net_name in ["MLP3", "MLP5", "MLP7", "MLP9"]:
                net = DR_MLP(DR_net_name, ngpu=NGPU, final_ActFn=args.DR_final_ActFn)
            return net

        # PreTrain CNN for DRE in feature space
        _, net_name = CNN_net_init(args.PreCNN_DR, NGPU)
        Filename_PreCNNForDRE = save_models_folder + '/ckpt_' + net_name + '_epoch_' + str(args.epoch_pretrainCNN) + '_SEED_' + str(args.seed) + '_Transformation_' + str(args.transform_PreCNN_DR) + '_NFOLDS_' + str(args.nfolds) + '_nround_' + str(nround)
        ### load pretrained CNN
        PreNetDRE, _ = CNN_net_init(args.PreCNN_DR, NGPU)
        PreNetDRE = PreNetDRE.to(device)
        checkpoint = torch.load(Filename_PreCNNForDRE)
        PreNetDRE.load_state_dict(checkpoint['net_state_dict'])

        #---------------------------------------------------------------
        # Train the first DR model
        start = timeit.default_timer()
        # initialize DRE model and optimizer
        netDR = DR_net_init(args.DR_Net)
        optimizer = torch.optim.Adam(netDR.parameters(), lr = args.base_lr_DRE, betas=(ADAM_beta1, ADAM_beta2), weight_decay=args.weightdecay_DRE)
        #filename for DR Model
        Filename_DRE = save_models_folder + '/ckpt_DRE-F-SP_' + args.DR_Net + '_' + args.DR_final_ActFn + '_epoch_' + str(args.epoch_DRE) + '_SEED_' + str(args.seed) + '_Lambda_' + str(args.lambda_DRE) + '_RegDCGAN_epoch_' + str(args.epoch_gan) + '_NFOLDS_' + str(args.nfolds) + '_nround_' + str(nround)
        #if DR model exists, then load the pretrained model; otherwise, start training the model.
        if not os.path.isfile(Filename_DRE):
            print("\n Begin Training DR in Feature Space: >>>\n")
            netDR, optimizer, avg_train_loss = train_cDREF(NGPU, args.epoch_DRE, args.base_lr_DRE, trainloader_DRE, netDR, optimizer, PreNetDRE, netG, args.dim_gan, LAMBDA = args.lambda_DRE, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_DRE, loss_type = "SP", device=device, not_decay_lr=args.not_decay_lr_DRE, decay_epochs=int(args.epoch_DRE/4))
            # save model
            torch.save({
            'net_state_dict': netDR.state_dict(),
            }, Filename_DRE)
        else:
            # if already trained, load pre-trained DR model
            checkpoint_netDR = torch.load(Filename_DRE)
            netDR = DR_net_init(args.DR_Net)
            netDR.load_state_dict(checkpoint_netDR['net_state_dict'])
            netDR = netDR.to(device)
        torch.cuda.empty_cache()
        stop = timeit.default_timer()
        print("DRE fitting finished; Time elapses: {}s".format(stop - start))

        # function for computing a bunch of images in a numpy array
        def comp_density_ratio(imgs, counts):
            #imgs: an numpy array
            n_imgs = imgs.shape[0]
            if DR_comp_batch_size<n_imgs:
                batch_size_tmp = DR_comp_batch_size
            else:
                batch_size_tmp = n_imgs
            dataset_tmp = IMGs_dataset(imgs, counts)
            dataloader_tmp = torch.utils.data.DataLoader(dataset_tmp, batch_size=batch_size_tmp, shuffle=False, num_workers=0)
            data_iter = iter(dataloader_tmp)
            density_ratios = np.zeros((n_imgs+batch_size_tmp, 1))

            netDR.eval()
            PreNetDRE.eval()
            # print("\n Begin computing density ratio for images >>")
            with torch.no_grad():
                tmp = 0
                while tmp < n_imgs:
                    batch_imgs, batch_counts = data_iter.next()
                    batch_imgs = batch_imgs.type(torch.float).to(device)
                    batch_counts = batch_counts.type(torch.float).to(device)
                    _, batch_features = PreNetDRE(batch_imgs)
                    batch_features = torch.cat((batch_features, batch_counts), dim=1)
                    batch_weights = netDR(batch_features)
                    density_ratios[tmp:(tmp+batch_size_tmp)] = batch_weights.cpu().detach().numpy()
                    tmp += batch_size_tmp
                #end while
                return density_ratios[0:n_imgs]

        # enhanced_sampler for the first DR model
        # Rejection Sampling:"Discriminator Rejection Sampling"; based on https://github.com/shinseung428/DRS_Tensorflow/blob/master/config.py
        def fn_enhanceSampler(nfake, batch_size=samp_batch_size):
            ## Burn-in Stage
            n_burnin = 10000
            burnin_imgs, burnin_counts = fn_sampleGAN(n_burnin, batch_size)
            burnin_densityratios = comp_density_ratio(burnin_imgs, burnin_counts)
            # print((burnin_densityratios.min(),np.median(burnin_densityratios),burnin_densityratios.max()))
            M_bar = np.max(burnin_densityratios)
            del burnin_imgs, burnin_densityratios; gc.collect()
            torch.cuda.empty_cache()

            ## Rejection sampling
            enhanced_imgs = np.zeros((1, NC, IMG_SIZE, IMG_SIZE)) #initilize
            enhanced_counts = np.zeros(1)
            pb = SimpleProgressBar()
            num_imgs = 0
            while num_imgs < nfake:
                batch_imgs, batch_counts = fn_sampleGAN(batch_size, batch_size)
                batch_ratios = comp_density_ratio(batch_imgs, batch_counts)
                M_bar = np.max([M_bar, np.max(batch_ratios)])
                #threshold
                batch_p = batch_ratios/M_bar
                batch_psi = np.random.uniform(size=batch_size).reshape(-1,1)
                indx_accept = np.where(batch_psi<=batch_p)[0]
                if len(indx_accept)>0:
                    enhanced_imgs = np.concatenate((enhanced_imgs, batch_imgs[indx_accept]))
                    enhanced_counts = np.concatenate((enhanced_counts, batch_counts[indx_accept]))
                num_imgs=len(enhanced_imgs)-1
                del batch_imgs, batch_ratios; gc.collect()
                torch.cuda.empty_cache()
                pb.update(np.min([float(num_imgs)*100/nfake,100]))
            return enhanced_imgs[1:(nfake+1)], enhanced_counts[1:(nfake+1)] #remove the first all zero array

        # generate enhanced images based on the first DR model
        if args.da_output and args.DA_method not in ['RegDCGAN+DRE-F-SP+RS+double']:
            fake_imgs, fake_counts = fn_enhanceSampler(args.da_nfake, samp_batch_size)

        #---------------------------------------------------------------
        # train the second DR model
        if args.DA_method=='RegDCGAN+DRE-F-SP+RS+double':
            start = timeit.default_timer()
            # initialize DRE model and optimizer
            netDR2 = DR_net_init(args.DR_Net)
            optimizer = torch.optim.Adam(netDR2.parameters(), lr = args.base_lr_DRE, betas=(ADAM_beta1, ADAM_beta2), weight_decay=args.weightdecay_DRE)
            #filename for DR Model
            Filename_DRE2 = save_models_folder + '/ckpt_DRE-F-SP_' + args.DR_Net + '_' + args.DR_final_ActFn + '_epoch_' + str(args.epoch_DRE) + '_SEED_' + str(args.seed) + '_Lambda_' + str(args.lambda_DRE) + '_RegDCGAN_epoch_' + str(args.epoch_gan) + '_NFOLDS_' + str(args.nfolds) + '_nround_' + str(nround) + '_2'
            #if DR model exists, then load the pretrained model; otherwise, start training the model.
            if not os.path.isfile(Filename_DRE2):
                print("\n Begin Training 2nd DR in Feature Space: >>>\n")
                netDR2, optimizer, avg_train_loss = train_2nd_DREF(NGPU, args.epoch_DRE, args.base_lr_DRE, trainloader, netDR2, optimizer, PreNetDRE, netDR, netG, args.dim_gan, LAMBDA=args.lambda_DRE2, save_models_folder = save_models_folder, loss_type = "SP", device=device, not_decay_lr=args.not_decay_lr_DRE, decay_epochs=int(args.epoch_DRE/4))
                # save model
                torch.save({
                'net_state_dict': netDR2.state_dict(),
                }, Filename_DRE2)
            else:
                # if already trained, load pre-trained DR model
                checkpoint_netDR2 = torch.load(Filename_DRE2)
                netDR2 = DR_net_init(args.DR_Net)
                netDR2.load_state_dict(checkpoint_netDR2['net_state_dict'])
                netDR2 = netDR2.to(device)
            torch.cuda.empty_cache()
            stop = timeit.default_timer()
            print("DRE fitting finished; Time elapses: {}s".format(stop - start))

            # function for computing a bunch of images in a numpy array
            def comp_density_ratio_2(imgs, counts):
                #imgs: an numpy array
                n_imgs = imgs.shape[0]
                if DR_comp_batch_size<n_imgs:
                    batch_size_tmp = DR_comp_batch_size
                else:
                    batch_size_tmp = n_imgs
                dataset_tmp = IMGs_dataset(imgs, counts)
                dataloader_tmp = torch.utils.data.DataLoader(dataset_tmp, batch_size=batch_size_tmp, shuffle=False, num_workers=0)
                data_iter = iter(dataloader_tmp)
                density_ratios = np.zeros((n_imgs+batch_size_tmp, 1))

                netDR2.eval()
                PreNetDRE.eval()
                # print("\n Begin computing density ratio for images >>")
                with torch.no_grad():
                    tmp = 0
                    while tmp < n_imgs:
                        batch_imgs, batch_counts = data_iter.next()
                        batch_imgs = batch_imgs.type(torch.float).to(device)
                        batch_counts = batch_counts.type(torch.long).to(device)
                        _, batch_features = PreNetDRE(batch_imgs)
                        batch_weights = netDR2(torch.cat((batch_features, batch_counts),dim=1))
                        density_ratios[tmp:(tmp+batch_size_tmp)] = batch_weights.cpu().detach().numpy()
                        tmp += batch_size_tmp
                    #end while
                    return density_ratios[0:n_imgs]

            # enhanced_sampler for the second DR model
            # Rejection Sampling:"Discriminator Rejection Sampling"; based on https://github.com/shinseung428/DRS_Tensorflow/blob/master/config.py
            def fn_enhanceSampler_2(nfake, batch_size=samp_batch_size):
                ## Burn-in Stage
                n_burnin = 10000
                burnin_imgs, burnin_counts = fn_enhanceSampler(nfake, batch_size)
                burnin_densityratios = comp_density_ratio_2(burnin_imgs, burnin_counts)
                M_bar = np.max(burnin_densityratios)
                del burnin_imgs, burnin_densityratios; gc.collect()
                torch.cuda.empty_cache()

                ## Rejection sampling
                enhanced_imgs = np.zeros((1, NC, IMG_SIZE, IMG_SIZE)) #initilize
                enhanced_counts = np.zeros(1)
                pb = SimpleProgressBar()
                num_imgs = 0
                while num_imgs < nfake:
                    batch_imgs, batch_counts = fn_enhanceSampler(batch_size, batch_size)
                    batch_ratios = comp_density_ratio_2(batch_imgs, batch_counts)
                    M_bar = np.max([M_bar, np.max(batch_ratios)])
                    #threshold
                    batch_p = batch_ratios/M_bar
                    batch_psi = np.random.uniform(size=batch_size).reshape(-1,1)
                    indx_accept = np.where(batch_psi<=batch_p)[0]
                    if len(indx_accept)>0:
                        enhanced_imgs = np.concatenate((enhanced_imgs, batch_imgs[indx_accept]))
                        enhanced_counts = np.concatenate((enhanced_counts, batch_counts[indx_accept]))
                    num_imgs=len(enhanced_imgs)-1
                    del batch_imgs, batch_ratios; gc.collect()
                    torch.cuda.empty_cache()
                    pb.update(np.min([float(num_imgs)*100/nfake,100]))
                return enhanced_imgs[1:(nfake+1)], enhanced_counts[1:(nfake+1)] #remove the first all zero array

            # generate enhanced images based on the second DR model
            if args.da_output:
                fake_imgs, fake_counts = fn_enhanceSampler_2(args.da_nfake, samp_batch_size)

    #dump to h5 file
    if args.da_output:
        #denormlize: [-1,1]--->[0,255]
        fake_imgs = (fake_imgs*0.5+0.5)*255.0
        fake_imgs = fake_imgs.astype(np.uint8)

        h5py_file = wd+'/data/VGG_dataset_CV_' + args.DA_method + '_NFAKE_'+ str(len(fake_imgs)) + '_NFOLDS_' + str(args.nfolds) + '_nround_' + str(nround) + '.h5'
        with h5py.File(h5py_file, "w") as f:
            f.create_dataset('fake_images', data = fake_imgs)
            f.create_dataset('fake_counts', data = fake_counts)


# end for nround
