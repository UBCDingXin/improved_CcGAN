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
# from sklearn.model_selection import GridSearchCV
# from sklearn import mixture


from utils import *
from models import *
from Train_GAN import *



#######################################################################################
'''                                  Settings                                     '''
#######################################################################################
parser = argparse.ArgumentParser(description='Simulation')
'''Overall Settings'''
parser.add_argument('--NSIM', type=int, default=1,
                    help = "How many times does this experiment need to be repeated?")
parser.add_argument('--DIM', type=int, default=2,
                    help = "Dimension of the Euclidean space of our interest")
parser.add_argument('--n_comp_tar', type=int, default=25,
                    help = "Number of mixture components in the target distribution")
parser.add_argument('--DRE', type=str, default='DRE_SP',
                    choices=['None', 'GT', 'DRE_uLSIF', 'DRE_DSKL', 'DRE_BARR', 'DRE_SP', 'disc', 'disc_MHcal', 'disc_KeepTrain'], #GT is ground truth
                    help='Density ratio estimation method; None means randomly sample from the proposal distribution or the trained GAN')
parser.add_argument('--Sampling', type=str, default='RS',
                    help='Sampling method; Candidiate: None, RS, MH, SIR') #RS: rejection sampling, MH: Metropolis-Hastings; SIR: Sampling-Importance Resampling
parser.add_argument('--seed', type=int, default=2019, metavar='S',
                    help='random seed (default: 2019)')
parser.add_argument('--show_reference', action='store_true', default=False,
                    help='Assign 1 as density ratios to all samples and compute errors')
parser.add_argument('--show_visualization', action='store_true', default=False,
                    help='Plot fake samples in 2D coordinate')

''' Data Generation '''
parser.add_argument('--NTRAIN', type=int, default=50000)
parser.add_argument('--NTEST', type=int, default=10000)

''' GAN settings '''
parser.add_argument('--epoch_gan', type=int, default=50) #default 50
parser.add_argument('--lr_gan', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--dim_gan', type=int, default=2,
                    help='Latent dimension of GAN')
parser.add_argument('--batch_size_gan', type=int, default=512, metavar='N',
                    help='input batch size for training GAN')
parser.add_argument('--resumeTrain_gan', type=int, default=0)
parser.add_argument('--compute_disc_err', action='store_true', default=False,
                    help='Compute the distance between the discriminator and its optimality')

'''DRE Settings'''
parser.add_argument('--DR_Net', type=str, default='MLP5',
                    choices=['MLP3', 'MLP5', 'MLP7', 'MLP9',
                             'CNN5'],
                    help='DR Model') # DR models
parser.add_argument('--epoch_DRE', type=int, default=200)
parser.add_argument('--base_lr_DRE', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--decay_lr_DRE', action='store_true', default=False,
                    help='decay learning rate')
parser.add_argument('--lr_decay_epochs_DRE', type=int, default=400)
parser.add_argument('--batch_size_DRE', type=int, default=512, metavar='N',
                    help='input batch size for training DRE')
parser.add_argument('--lambda_DRE', type=float, default=0.0, #BARR: lambda=10
                    help='penalty in DRE')
parser.add_argument('--weightdecay_DRE', type=float, default=0.0,
                    help='weight decay in DRE')
parser.add_argument('--resumeTrain_DRE', type=int, default=0)
parser.add_argument('--DR_final_ActFn', type=str, default='ReLU',
                    help='Final layer of the Density-ratio model; Candidiate: Softplus or ReLU')
parser.add_argument('--TrainPreNetDRE', action='store_true', default=False,
                    help='Pre-trained MLP for DRE in Feature Space')
parser.add_argument('--DRE_save_at_epoch', nargs='+', type=int)
parser.add_argument('--epoch_KeepTrain', type=int, default=20)
parser.add_argument('--compute_dre_err', action='store_true', default=False,
                    help='Compare the DRE method with the ground truth')





''' Mixture Gaussian (for density estimation) Settings '''
parser.add_argument('--gmm_nfake', type=int, default=100000)
# parser.add_argument('--gmm_ncomp', type=int, default=0) #gmm_ncomp is non-positive, then we do ncomp selection
parser.add_argument('--gmm_ncomp_nsim', nargs='+', type=int) #A list of ncomp for NSIM rounds. If gmm_ncomp is None, then we do ncomp selection
parser.add_argument('--gmm_ncomp_grid', nargs='+', type=int)
parser.add_argument('--gmm_ncomp_grid_lb', type=int, default=1)
parser.add_argument('--gmm_ncomp_grid_ub', type=int, default=100)
parser.add_argument('--gmm_ncomp_grid_step', type=int, default=1)


args = parser.parse_args()


#--------------------------------
# system
assert torch.cuda.is_available()
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if NGPU>0 else "cpu")
cores= multiprocessing.cpu_count()


#--------------------------------
# Extra Data Generation Settings
n_comp_tar = args.n_comp_tar
n_features = args.DIM
mean_grid_tar = [-2, -1, 0, 1, 2]
sigma_tar = 0.05
n_classes = n_comp_tar
quality_threshold = sigma_tar*4 #good samples are within 4 standard deviation


#--------------------------------
# GAN Settings
epoch_GAN = args.epoch_gan
lr_GAN = args.lr_gan
batch_size_GAN = args.batch_size_gan
dim_GAN = args.dim_gan
plot_in_train = True
gan_Adam_beta1 = 0.5
gan_Adam_beta2 = 0.999

#--------------------------------
# Extra DRE Settings
DRE_Adam_beta1 = 0.5
DRE_Adam_beta2 = 0.999
comp_ratio_bs = 1000 #batch size for computing density ratios
base_lr_PreNetDRE = 1e-1
epoch_PreNetDRE = 100
DRE_save_at_epoch = args.DRE_save_at_epoch # save checkpoints at these epochs
# DRE_save_at_epoch = [20, 50, 100, 150, 200, 300, 400, 500, 800]
epoch_KeepTrain = args.epoch_KeepTrain   #keep training for DRS
ckp_epoch_KeepTrain = [i for i in range(100) if i%5==0]

#--------------------------------
# Mixture Gaussian Setting
gmm_nfake = args.gmm_nfake
# gmm_ncomp = args.gmm_ncomp
gmm_ncomp_nsim = args.gmm_ncomp_nsim
# if gmm_ncomp_nsim is not None:
#     assert len(gmm_ncomp_nsim) == args.NSIM
if args.gmm_ncomp_grid is not None:
    gmm_ncomp_grid = args.gmm_ncomp_grid
else:
    gmm_ncomp_grid = np.arange(args.gmm_ncomp_grid_lb, args.gmm_ncomp_grid_ub+args.gmm_ncomp_grid_step, args.gmm_ncomp_grid_step)


#--------------------------------
# Extra Sampling Settings
NFAKE = args.NTEST
samp_batch_size = 10000
MH_K = 100
MH_mute = True #do not print sampling progress


#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
np.random.seed(args.seed)

#-------------------------------
# output folders
save_models_folder = wd + '/Output/saved_models/'
os.makedirs(save_models_folder,exist_ok=True)
save_images_folder = wd + '/Output/saved_images/'
os.makedirs(save_images_folder,exist_ok=True)
save_traincurves_folder = wd + '/Output/Training_loss_fig/'
os.makedirs(save_traincurves_folder,exist_ok=True)
save_GANimages_InTrain_folder = wd + '/Output/saved_images/GAN_InTrain'
os.makedirs(save_GANimages_InTrain_folder,exist_ok=True)
save_objects_folder = wd + '/Output/saved_objects'
os.makedirs(save_objects_folder,exist_ok=True)

#######################################################################################
'''                               Start Experiment                                 '''
#######################################################################################
#---------------------------------
# sampler for reference distribution
means_tar = np.zeros((1,n_features))
for i in mean_grid_tar:
    for j in mean_grid_tar:
        means_tar = np.concatenate((means_tar, np.array([i,j]).reshape(-1,n_features)), axis=0)
means_tar = means_tar[1:]
assert means_tar.shape[0] == n_comp_tar
assert means_tar.shape[1] == n_features
def generate_data_tar(nsamp):
    return sampler_MixGaussian(nsamp, means_tar, sigma = sigma_tar, dim = n_features)
def p_r(samples): #pdf of the underlying distribution; samples is a n by n_features sample matrix
    return pdf_MixGaussian(samples, means_tar, sigma_tar)

prop_recovered_modes = np.zeros(args.NSIM) # num of removed modes diveded by num of modes
prop_good_samples = np.zeros(args.NSIM) # num of good fake samples diveded by num of all fake samples
valid_densityratios_all = [] #store denstiy ratios for validation samples
train_densityratios_all = []
ks_test_results = np.zeros((args.NSIM,2))

dre_errors_all = np.zeros(args.NSIM) #compute density ratios on the test set (hold-out set) with each DRE method and the ground truth
dre_errors_hq = np.zeros(args.NSIM)
dre_errors_lq = np.zeros(args.NSIM)
esti_avg_densityratio = np.zeros((args.NSIM, 4)) #estimated density ratios of testing samples, NFAKE fake samples, HQ fake samples, LQ fake samples
true_avg_densityratio = np.zeros((args.NSIM, 4)) #true density ratios of testing samples, NFAKE fake samples, HQ fake samples, LQ fake samples

disc_errors_all = np.zeros(args.NSIM) #compute the distance between the discriminator and its optimality
nfake_in_train = np.zeros(args.NSIM)


print("\n Begin The Experiment. Sample from a GAN! >>>")
start = timeit.default_timer()
for nSim in range(args.NSIM):
    print("Round %s" % (nSim))
    np.random.seed(nSim) #set seed for current simulation
    ###############################################################################
    # Data generation and dataloaders
    ###############################################################################

    train_samples_tar, train_labels_tar = generate_data_tar(args.NTRAIN)
    valid_samples_tar, valid_labels_tar = generate_data_tar(args.NTRAIN)
    test_samples_tar, test_labels_tar = generate_data_tar(args.NTEST)

    train_dataset_tar = custom_dataset(train_samples_tar, train_labels_tar)
    test_dataset_tar = custom_dataset(test_samples_tar, test_labels_tar)
    train_dataloader_tar = torch.utils.data.DataLoader(train_dataset_tar, batch_size=args.batch_size_DRE, shuffle=True, num_workers=0)
    test_dataloader_tar = torch.utils.data.DataLoader(test_dataset_tar, batch_size=100, shuffle=False, num_workers=0)

    # #compute the criterion for determing good smaples through train_samples_tar
    # # for each mixture component, compute the average distance of real samples from this component to the mean
    # l2_dis_train_samples = np.zeros(args.NTRAIN) #l2 distance between a fake sample and a mode
    # for i in range(args.NTRAIN):
    #     indx_mean = int(train_labels_tar[i])
    #     l2_dis_train_samples[i] = np.sqrt(np.sum((train_samples_tar[i]-means_tar[indx_mean])**2))
    # print(l2_dis_train_samples.max())

    ###############################################################################
    # Train a GAN model
    ###############################################################################
    Filename_GAN = save_models_folder + '/ckpt_GAN_epoch_' + str(args.epoch_gan) + '_SEED_' + str(args.seed) + '_nSim_' + str(nSim)
    print("\n Begin Training GAN:")
    #model initialization
    netG = generator(ngpu=NGPU, nz=dim_GAN, out_dim=n_features)
    netD = discriminator(ngpu=NGPU, input_dim = n_features)
    if not os.path.isfile(Filename_GAN):
        criterion = nn.BCELoss()
        optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_GAN, betas=(gan_Adam_beta1, gan_Adam_beta2))
        optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_GAN, betas=(gan_Adam_beta1, gan_Adam_beta2))

        # Start training
        netG, netD, optimizerG, optimizerD = train_GAN(epoch_GAN, dim_GAN, train_dataloader_tar, netG, netD, optimizerG, optimizerD, criterion, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan, device=device, plot_in_train=plot_in_train, save_images_folder = save_GANimages_InTrain_folder, samples_tar = test_samples_tar)

        # store model
        torch.save({
           'netG_state_dict': netG.state_dict(),
           'netD_state_dict': netD.state_dict(),
        }, Filename_GAN)

        torch.cuda.empty_cache()
    else: #load pre-trained GAN
        print("\n GAN exists! Loading Pretrained Model>>>")
        checkpoint = torch.load(Filename_GAN)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        netG = netG.to(device)
        netD = netD.to(device)

    def fn_sampleGAN(nfake, batch_size=1000):
        return SampGAN(netG, GAN_Latent_Length = args.dim_gan, NFAKE = nfake, batch_size = batch_size, device=device)

    ###############################################################################
    # Construct a function to compute density-ratio
    ###############################################################################
    # Approximate DR by NN
    if args.DRE in ['DRE_uLSIF', 'DRE_DSKL', 'DRE_BARR', 'DRE_SP']:

        # TRAIN DRE
        DRE_loss_type = args.DRE[4:] #loss type

        if args.DR_Net in ['MLP3', 'MLP5', 'MLP7', 'MLP9']:
            netDR = DR_MLP(args.DR_Net, init_in_dim = n_features, ngpu=NGPU, final_ActFn=args.DR_final_ActFn)
        elif args.DR_Net in ['CNN5']:
            netDR = DR_CNN(args.DR_Net, init_in_dim = n_features, ngpu=NGPU, final_ActFn=args.DR_final_ActFn)
        optimizer = torch.optim.Adam(netDR.parameters(), lr = args.base_lr_DRE, betas=(DRE_Adam_beta1, DRE_Adam_beta2), weight_decay=args.weightdecay_DRE)
        #optimizer = torch.optim.RMSprop(netDR.parameters(), lr= args.base_lr_DRE, alpha=0.99, eps=1e-08, weight_decay=args.weightdecay_DRE, momentum=0.9, centered=False)

        Filename_DRE = save_models_folder + '/ckpt_' + args.DRE +'_LAMBDA_' + str(args.lambda_DRE) + '_FinalActFn_' + args.DR_final_ActFn  + '_epoch_' + str(args.epoch_DRE) \
            + "_PreNetDRE_" + str(args.TrainPreNetDRE) + '_SEED_' + str(args.seed) + '_nSim_' + str(nSim) + '_epochGAN_' + str(epoch_GAN)
        filename0 = save_traincurves_folder + '/TrainCurve_' + args.DRE +'_LAMBDA_' + str(args.lambda_DRE) + '_FinalActFn_' + args.DR_final_ActFn  + '_epoch_' \
            + str(args.epoch_DRE) + "_PreNetDRE_" + str(args.TrainPreNetDRE) + '_SEED_' + str(args.seed) + "_nSim_" + str(nSim) + '_epochGAN_' + str(epoch_GAN) + "_TrainLoss"
        plot_filename = filename0 + '.pdf'
        npy_filename = filename0 + '.npy'

        # Train a net to extract features for DR net
        if args.TrainPreNetDRE:
            print("\n Begin Training PreNetDRE Net:")
            Filename_PreNetDRE = save_models_folder + '/ckpt_PreNetDRE_epochPreNetDRE_' + str(epoch_PreNetDRE) + '_SEED_' + str(args.seed) + '_nSim_' + str(nSim) + '_epochGAN_' + str(epoch_GAN)
            PreNetDRE_MLP = PreNetDRE_MLP(init_in_dim = n_features, ngpu=NGPU)
            if not os.path.isfile(Filename_PreNetDRE):
                criterion_PreNetDRE = nn.CrossEntropyLoss()
                optimizer_PreNetDRE = torch.optim.SGD(PreNetDRE_MLP.parameters(), lr = base_lr_PreNetDRE, momentum= 0.9, weight_decay=1e-4)
                PreNetDRE_MLP, _ = train_PreNetDRE(epoch_PreNetDRE, train_dataloader_tar, test_dataloader_tar, PreNetDRE_MLP, base_lr_PreNetDRE, optimizer_PreNetDRE, criterion_PreNetDRE, device=device)
                # save model
                torch.save({
                'net_state_dict': PreNetDRE_MLP.state_dict(),
                }, Filename_PreNetDRE)
            else:
                print("\n PreNetDRE Net exits and start loading:")
                checkpoint_PreNetDRE_MLP = torch.load(Filename_PreNetDRE)
                PreNetDRE_MLP.load_state_dict(checkpoint_PreNetDRE_MLP['net_state_dict'])
                PreNetDRE_MLP = PreNetDRE_MLP.to(device)

            def extract_features(samples):
                #samples: an numpy array
                n_samples = samples.shape[0]
                batch_size_tmp = 1000
                dataset_tmp = custom_dataset(samples)
                dataloader_tmp = torch.utils.data.DataLoader(dataset_tmp, batch_size=batch_size_tmp, shuffle=False, num_workers=0)
                data_iter = iter(dataloader_tmp)
                extracted_features = np.zeros((n_samples+batch_size_tmp, n_features))

                PreNetDRE_MLP.eval()
                with torch.no_grad():
                    tmp = 0
                    while tmp < n_samples:
                        batch_samples,_ = data_iter.next()
                        batch_samples = batch_samples.type(torch.float).to(device)
                        _, batch_features = PreNetDRE_MLP(batch_samples)
                        extracted_features[tmp:(tmp+batch_size_tmp)] = batch_features.cpu().detach().numpy()
                        tmp += batch_size_tmp
                    #end while
                return extracted_features[0:n_samples]

            test_features_tar = extract_features(test_samples_tar)
            plt.switch_backend('agg')
            mpl.style.use('seaborn')
            plt.figure()
            plt.grid(b=True)
            flag0 = 0; flag1=0
            colors = ['b','g','r','c','m','y','k']
            marker_styles = ['.', 'o', 'v', 's']
            for nc in range(n_classes):
                indx = np.where(test_labels_tar == nc)[0]
                plt.scatter(test_features_tar[indx, 0], test_features_tar[indx, 1], c=colors[flag0], marker=marker_styles[flag1], s=8)
                flag0 += 1
                if flag0 % 7 ==0 :
                    flag0 = 0; flag1+=1
            filename0 = save_images_folder + '/test.pdf'
            plt.savefig(filename0)
            plt.close()

        if not os.path.isfile(Filename_DRE):
            # Train
            print("\n Begin Training DRE NET:")
            if args.TrainPreNetDRE:
                netDR, optimizer, avg_train_loss = train_DRE_GAN(net=netDR, optimizer=optimizer, BASE_LR_DRE=args.base_lr_DRE, EPOCHS_DRE=args.epoch_DRE, LAMBDA=args.lambda_DRE, tar_dataloader=train_dataloader_tar, netG=netG, dim_gan=dim_GAN, PreNetDRE = PreNetDRE_MLP, decay_lr=args.decay_lr_DRE, decay_epochs=args.lr_decay_epochs_DRE, loss_type=DRE_loss_type, save_models_folder = save_models_folder, ResumeEpoch=args.resumeTrain_DRE, NGPU=NGPU, device=device, save_at_epoch = DRE_save_at_epoch, current_nsim=nSim)
            else:
                netDR, optimizer, avg_train_loss = train_DRE_GAN(net=netDR, optimizer=optimizer, BASE_LR_DRE=args.base_lr_DRE, EPOCHS_DRE=args.epoch_DRE, LAMBDA=args.lambda_DRE, tar_dataloader=train_dataloader_tar, netG=netG, dim_gan=dim_GAN, decay_lr=args.decay_lr_DRE, decay_epochs=args.lr_decay_epochs_DRE, loss_type=DRE_loss_type, save_models_folder = save_models_folder, ResumeEpoch=args.resumeTrain_DRE, NGPU=NGPU, device=device, save_at_epoch = DRE_save_at_epoch, current_nsim=nSim)

            # Plot loss
            PlotLoss(avg_train_loss, plot_filename)
            np.save(npy_filename, np.array(avg_train_loss))
            # save model
            torch.save({
            'net_state_dict': netDR.state_dict(),
            }, Filename_DRE)
        else: #if the DR model is already trained, load the checkpoint
            print("\n DRE NET exists and start loading:")
            checkpoint_netDR = torch.load(Filename_DRE)
            netDR.load_state_dict(checkpoint_netDR['net_state_dict'])
            netDR = netDR.to(device)

        def comp_density_ratio(samples, verbose=False):
            #samples: an numpy array
            n_samples = samples.shape[0]
            batch_size_tmp = 1000
            dataset_tmp = custom_dataset(samples)
            dataloader_tmp = torch.utils.data.DataLoader(dataset_tmp, batch_size=batch_size_tmp, shuffle=False, num_workers=0)
            data_iter = iter(dataloader_tmp)
            density_ratios = np.zeros((n_samples+batch_size_tmp, 1))
            netDR.eval()
            if args.TrainPreNetDRE:
                PreNetDRE_MLP.eval()
            with torch.no_grad():
                tmp = 0
                while tmp < n_samples:
                    batch_samples,_ = data_iter.next()
                    batch_samples = batch_samples.type(torch.float).to(device)
                    if args.TrainPreNetDRE:
                        _, batch_features = PreNetDRE_MLP(batch_samples)
                        batch_weights = netDR(batch_features)
                    else:
                        batch_weights = netDR(batch_samples)
                    #density_ratios[tmp:(tmp+batch_size_tmp)] = batch_weights.cpu().detach().numpy()
                    density_ratios[tmp:(tmp+batch_size_tmp)] = batch_weights.cpu().numpy()
                    tmp += batch_size_tmp
                    if verbose:
                        print(batch_weights.cpu().numpy().mean())
                #end while
            return density_ratios[0:n_samples]+1e-14

    ###################
    # DRE based on GAN property
    elif args.DRE in ['disc', 'disc_MHcal', 'disc_KeepTrain']:
        if args.DRE == 'disc': #use GAN property to compute density ratio; ratio=D/(1-D);
            # function for computing a bunch of images
            # def comp_density_ratio(samples, netD):
            def comp_density_ratio(samples):
               #samples: an numpy array
               n_samples = samples.shape[0]
               batch_size_tmp = 1000
               dataset_tmp = custom_dataset(samples)
               dataloader_tmp = torch.utils.data.DataLoader(dataset_tmp, batch_size=batch_size_tmp, shuffle=False, num_workers=0)
               data_iter = iter(dataloader_tmp)
               density_ratios = np.zeros((n_samples+batch_size_tmp, 1))

               # print("\n Begin computing density ratio for images >>")
               netD.eval()
               with torch.no_grad():
                   tmp = 0
                   while tmp < n_samples:
                       batch_samples,_ = data_iter.next()
                       batch_samples = batch_samples.type(torch.float).to(device)
                       disc_probs = netD(batch_samples).cpu().detach().numpy()
                       disc_probs = np.clip(disc_probs.astype(np.float), 1e-14, 1 - 1e-14)
                       density_ratios[tmp:(tmp+batch_size_tmp)] = np.divide(disc_probs, 1-disc_probs)
                       tmp += batch_size_tmp
                   #end while
               # print("\n End computing density ratio.")
               return density_ratios[0:n_samples]

        #-----------------------------------
        elif args.DRE == 'disc_MHcal': #use the calibration method in MH-GAN to calibrate disc
            n_test = valid_samples_tar.shape[0]
            batch_size_tmp = 1000
            cal_labels_fake = np.zeros((n_test,1))
            cal_labels_real = np.ones((n_test,1))
            cal_samples_fake = fn_sampleGAN(nfake=n_test, batch_size=batch_size_tmp)
            dataset_fake = custom_dataset(cal_samples_fake)
            dataloader_fake = torch.utils.data.DataLoader(dataset_fake, batch_size=batch_size_tmp, shuffle=False, num_workers=0)
            dataset_real = custom_dataset(valid_samples_tar)
            dataloader_real = torch.utils.data.DataLoader(dataset_real, batch_size=batch_size_tmp, shuffle=False, num_workers=0)
            del cal_samples_fake; gc.collect()

            # get the output of disc before the final sigmoid layer; the \tilde{D} in Eq.(4) in "Discriminator Rejection Sampling"
            # def comp_disc_scores(samples_dataloader, netD):
            def comp_disc_scores(samples_dataloader):
                # samples_dataloader: the data loader for images
                n_samples = len(samples_dataloader.dataset)
                data_iter = iter(samples_dataloader)
                batch_size_tmp = samples_dataloader.batch_size
                disc_scores = np.zeros((n_samples+batch_size_tmp, 1))
                netD.eval()
                with torch.no_grad():
                    tmp = 0
                    while tmp < n_samples:
                        batch_samples,_ = data_iter.next()
                        batch_samples = batch_samples.type(torch.float).to(device)
                        disc_probs = netD(batch_samples).cpu().detach().numpy()
                        disc_probs = np.clip(disc_probs.astype(np.float), 1e-14, 1 - 1e-14)
                        disc_scores[tmp:(tmp+batch_size_tmp)] = np.log(np.divide(disc_probs, 1-disc_probs))
                        tmp += batch_size_tmp
                    #end while
                return disc_scores[0:n_samples]

            cal_disc_scores_fake = comp_disc_scores(dataloader_fake) #discriminator scores for fake images
            cal_disc_scores_real = comp_disc_scores(dataloader_real) #discriminator scores for real images

            # Train a logistic regression model
            X_train = np.concatenate((cal_disc_scores_fake, cal_disc_scores_real),axis=0).reshape(-1,1)
            y_train = np.concatenate((cal_labels_fake, cal_labels_real), axis=0).reshape(-1)
            #del cal_disc_scores_fake, cal_disc_scores_real; gc.collect()
            cal_logReg = LogisticRegression(solver="liblinear").fit(X_train, y_train)

            # function for computing a bunch of images
            # def comp_density_ratio(samples, netD):
            def comp_density_ratio(samples):
               #samples: an numpy array
               dataset_tmp = custom_dataset(samples)
               dataloader_tmp = torch.utils.data.DataLoader(dataset_tmp, batch_size=batch_size_tmp, shuffle=False, num_workers=0)
               disc_scores = comp_disc_scores(dataloader_tmp)
               disc_probs = (cal_logReg.predict_proba(disc_scores))[:,1] #second column corresponds to the real class
               disc_probs = np.clip(disc_probs.astype(np.float), 1e-14, 1 - 1e-14)
               density_ratios = np.divide(disc_probs, 1-disc_probs)
               return density_ratios.reshape(-1,1)

        #---------------------------------------------
        # disc_KeepTrain
        elif args.DRE == "disc_KeepTrain":
            batch_size_KeepTrain = 256
            Filename_KeepTrain_Disc = save_models_folder + '/ckpt_KeepTrainDisc_epoch_'+str(epoch_KeepTrain)+ '_SEED_' + str(args.seed) + '_nSim_' + str(nSim) + '_epochGAN_' + str(epoch_GAN)
            if not os.path.isfile(Filename_KeepTrain_Disc):
                print("Resume training Discriminator for %d epochs" % epoch_KeepTrain)
                # keep train the discriminator
                n_heldout = valid_samples_tar.data.shape[0]
                batch_size_tmp = 500
                cal_labels = np.concatenate((np.zeros((n_heldout,1)), np.ones((n_heldout,1))), axis=0)
                cal_imgs_fake = fn_sampleGAN(nfake=n_heldout, batch_size=batch_size_tmp)
                cal_imgs = np.concatenate((cal_imgs_fake, valid_samples_tar), axis=0)
                del cal_imgs_fake; gc.collect()
                cal_dataset = custom_dataset(cal_imgs, cal_labels)
                cal_dataloader = torch.utils.data.DataLoader(cal_dataset, batch_size=batch_size_KeepTrain, shuffle=True, num_workers=0)

                criterion_KeepTrain = nn.BCELoss()
                # optimizerD_KeepTrain = torch.optim.SGD(netD.parameters(), lr = 1e-3, momentum= 0.9, weight_decay=1e-4)
                optimizerD_KeepTrain = torch.optim.Adam(netD.parameters(), lr=lr_GAN, betas=(gan_Adam_beta1, gan_Adam_beta2))
                # optimizerD_KeepTrain = torch.optim.Adam(netD.parameters(), lr=lr_GAN/2, betas=(gan_Adam_beta1, gan_Adam_beta2))

                for epoch in range(epoch_KeepTrain):
                    netD.train()
                    train_loss = 0
                    for batch_idx, (batch_train_samples, batch_train_labels) in enumerate(cal_dataloader):

                        batch_train_samples = batch_train_samples.type(torch.float).cuda()
                        batch_train_labels = batch_train_labels.type(torch.float).cuda()

                        #Forward pass
                        outputs = netD(batch_train_samples)
                        loss = criterion_KeepTrain(outputs, batch_train_labels)

                        #backward pass
                        optimizerD_KeepTrain.zero_grad()
                        loss.backward()
                        optimizerD_KeepTrain.step()

                        train_loss += loss.cpu().item()
                    #end for batch_idx
                    print('KeepTrain netD: [epoch %d/%d] train_loss:%.3f' % (epoch+1, epoch_KeepTrain, train_loss/(batch_idx+1)))

                    if epoch+1 in ckp_epoch_KeepTrain:
                        filename_tmp = save_models_folder + '/ckpt_KeepTrainDisc_epoch_'+str(epoch+1)+ '_SEED_' + str(args.seed) + '_nSim_' + str(nSim) + '_epochGAN_' + str(epoch_GAN)
                        torch.save({
                        'net_state_dict': netD.state_dict(),
                        }, filename_tmp)

                #end for epoch
                # save model
                torch.save({
                'net_state_dict': netD.state_dict(),
                }, Filename_KeepTrain_Disc)
                del batch_train_samples, batch_train_labels, cal_dataset, cal_dataloader; gc.collect()
                torch.cuda.empty_cache()
            else:
                checkpoint = torch.load(Filename_KeepTrain_Disc)
                netD.load_state_dict(checkpoint['net_state_dict'])

            # def comp_density_ratio(imgs, netD):
            def comp_density_ratio(samples):
               #samples: an numpy array
               n_samples = samples.shape[0]
               batch_size_tmp = 1000
               dataset_tmp = custom_dataset(samples)
               dataloader_tmp = torch.utils.data.DataLoader(dataset_tmp, batch_size=batch_size_tmp, shuffle=False, num_workers=0)
               data_iter = iter(dataloader_tmp)
               density_ratios = np.zeros((n_samples+batch_size_tmp, 1))

               # print("\n Begin computing density ratio for images >>")
               netD.eval()
               with torch.no_grad():
                   tmp = 0
                   while tmp < n_samples:
                       batch_samples,_ = data_iter.next()
                       batch_samples = batch_samples.type(torch.float).to(device)
                       disc_probs = netD(batch_samples).cpu().detach().numpy().reshape((-1,1))
                       disc_probs = np.clip(disc_probs.astype(np.float), 1e-14, 1 - 1e-14)
                       density_ratios[tmp:(tmp+batch_size_tmp)] = np.divide(disc_probs, 1-disc_probs)
                       tmp += batch_size_tmp
                   #end while
               # print("\n End computing density ratio.")
               return density_ratios[0:n_samples]



    ###############################################################################
    # Fitting Mixture Gaussians to estimate p_g
    ###############################################################################
    if args.DRE=="GT" or args.compute_dre_err or args.compute_disc_err:

        #---------------------------------------------------------------------------
        # Select the optimal gmm_ncomp by minimizing BIC
        if gmm_ncomp_nsim is None:
            print("\n Select the optimal ncomp by minimizing BIC >>>")
            fake_samples = fn_sampleGAN(gmm_nfake, samp_batch_size) #draw many samples from GAN directly

            lowest_bic = np.infty
            gmm_ncomp = gmm_ncomp_grid[0]
            bic = []
            for n_components in gmm_ncomp_grid:
                # Fit a Gaussian mixture with EM
                gmm = mixture.GaussianMixture(n_components=n_components)
                gmm.fit(fake_samples)
                bic.append(gmm.bic(fake_samples))
                print("ncomp: %d; bic: %f; lowest bic: %f" % (n_components, bic[-1], lowest_bic))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    gmm_ncomp = n_components
            print("\n The optimal gmm_ncomp is %d with BIC %f" % (gmm_ncomp, lowest_bic))

            # fit gmm on all training data
            p_g_gmm = mixture.GaussianMixture(n_components=gmm_ncomp)
            p_g_gmm.fit(fake_samples)

            gmm_filename = save_objects_folder + "/fitted_gmm_nfake_" + str(gmm_nfake) + "_ncomp_" + str(gmm_ncomp) + "_epochGAN_" + str(epoch_GAN) + "_nSim_" + str(nSim) + ".pkl"

            with open(gmm_filename, 'wb') as f:
                pickle.dump(p_g_gmm, f)

            del fake_samples; gc.collect()

        #---------------------------------------------------------------------------
        # Fit Mixture Guassians with specified gmm_ncomp
        else:
            gmm_ncomp = gmm_ncomp_nsim[nSim]

            gmm_filename = save_objects_folder + "/fitted_gmm_nfake_" + str(gmm_nfake) + "_ncomp_" + str(gmm_ncomp) + "_epochGAN_" + str(epoch_GAN) + "_nSim_" + str(nSim) + ".pkl"

            if not os.path.isfile(gmm_filename):
                fake_samples = fn_sampleGAN(gmm_nfake, samp_batch_size) #draw many samples from GAN directly
                p_g_gmm = mixture.GaussianMixture(n_components=gmm_ncomp)
                p_g_gmm.fit(fake_samples)
                with open(gmm_filename, 'wb') as f:
                    pickle.dump(p_g_gmm, f)
            else:
                with open(gmm_filename, 'rb') as f:
                    p_g_gmm = pickle.load(f)

            # generate NFAKE samples from fitted gmm
            filename_visual_gmm = save_images_folder + "/sample_gmm_nfake_" + str(args.NTEST) + "_ncomp_" + str(gmm_ncomp) + "_epochGAN_" + str(epoch_GAN) + "_nSim_" + str(nSim) + ".pdf"
            gmm_fake_samples = p_g_gmm.sample(args.NTEST)[0]
            ScatterPoints(test_samples_tar, gmm_fake_samples, filename_visual_gmm, plot_real_samples = False)

            # ## DEBUG:
            # fake_samples = fn_sampleGAN(10000, samp_batch_size)
            # out = np.exp(p_g_gmm.score_samples(fake_samples[0:10]))
            # print(out)

        # function for evalute the fitted GMM on data X
        def gmm_eval(X):
            return p_g_gmm.score_samples(X)
        def gmm_eval_parallel(X):
            pool = multiprocessing.Pool(processes=cores)
            nsamples_eval = len(X)
            batch_size = nsamples_eval//cores
            batch_size_left = nsamples_eval-batch_size*cores
            ## split data into chunks
            chunks = []
            total_got = 0; flag=1
            while total_got<nsamples_eval:
                if flag<cores:
                    chunks.append(X[total_got:(total_got+batch_size)])
                    total_got += batch_size
                else:
                    chunks.append(X[total_got:(total_got+batch_size+batch_size_left)])
                    total_got += batch_size+batch_size_left
                flag+=1
            gmm_testscores = []
            for result in pool.imap(gmm_eval, chunks):
                gmm_testscores.extend(list(result))
            gmm_testscores = np.array(gmm_testscores)
            del chunks; gc.collect()
            pool.close()
            return gmm_testscores


    #---------------------------------------------------------------------------
    # Compare each DRE method with the ground truth (p_r/p_g, where p_g is estimated by KDE);
    if args.DRE not in ["None", "GT"] and args.compute_dre_err:
        dre_errors_comp_truedensityratios_filename = save_objects_folder + "/dre_error_comp_truedensityratios_epochGAN_" + str(epoch_GAN) + "_nfake_" + str(NFAKE) + "_gmm_ncomp_" + str(gmm_ncomp) + "_nSim_" + str(nSim) + ".pkl"

        dre_errors_comp_fakesamples_filename = save_objects_folder + "/dre_error_comp_fakesamples_epochGAN_" + str(epoch_GAN) + "_nSim_" + str(nSim) + ".pkl"

        #evaluate the optimal discriminator on NFAKE fake images
        if not os.path.isfile(dre_errors_comp_fakesamples_filename):
            fake_samples = fn_sampleGAN(NFAKE, samp_batch_size) #draw many samples from GAN directly
            with open(dre_errors_comp_fakesamples_filename, "wb") as file:
                pickle.dump(fake_samples, file)
        else:
            with open(dre_errors_comp_fakesamples_filename, "rb") as file:
                fake_samples = pickle.load(file)

        if not os.path.isfile(dre_errors_comp_truedensityratios_filename):
            gmm_fakescore = gmm_eval_parallel(fake_samples)
            gmm_testscore = gmm_eval_parallel(test_samples_tar)
            fake_densityratios_true = np.divide(p_r(fake_samples), np.exp(gmm_fakescore)+1e-20)
            test_densityratios_true = np.divide(p_r(test_samples_tar), np.exp(gmm_testscore)+1e-20)

            with open(dre_errors_comp_truedensityratios_filename, "wb") as file:
                temp_densityratios_true = {"fake_densityratios_true":fake_densityratios_true, "test_densityratios_true":test_densityratios_true}
                pickle.dump(temp_densityratios_true, file)
                del temp_densityratios_true; gc.collect()
        else:
            with open(dre_errors_comp_truedensityratios_filename, "rb") as file:
                temp_densityratios_true = pickle.load(file)
                fake_densityratios_true = temp_densityratios_true["fake_densityratios_true"]
                test_densityratios_true = temp_densityratios_true["test_densityratios_true"]
                del temp_densityratios_true; gc.collect()

        fake_densityratios_esti = comp_density_ratio(fake_samples)
        test_densityratios_esti = comp_density_ratio(test_samples_tar)

        # # DEBUG
        # temp_filename = save_objects_folder + "/sum_dr_versus_epochs.csv"
        # with open(temp_filename, 'a') as f:
        #     writer = csv.writer(f, delimiter=',')
        #     writer.writerow([args.epoch_DRE, np.sum(fake_densityratios_true), np.sum(fake_densityratios_esti)])





        # deno_true = np.sum(fake_densityratios_true) #sum
        # deno_esti = np.sum(fake_densityratios_esti) #sum

        deno_true = 1 #no standarization
        deno_esti = 1

        dre_errors_all[nSim] = np.sqrt(np.mean((fake_densityratios_true/deno_true - fake_densityratios_esti/deno_esti)**2))

        l2_dis_fake_samples = np.zeros((NFAKE, n_comp_tar)) #l2 distance between a fake sample and a mode
        for i in tqdm(range(NFAKE)):
            for j in range(n_comp_tar):
                l2_dis_fake_samples[i,j] = np.sqrt(np.sum((fake_samples[i]-means_tar[j])**2))
        min_l2_dis_fake_samples = np.min(l2_dis_fake_samples, axis=1)

        index_hq = np.where(min_l2_dis_fake_samples<quality_threshold)[0]
        index_lq = np.where(min_l2_dis_fake_samples>=quality_threshold)[0]

        dre_errors_hq[nSim] = np.sqrt(np.mean((fake_densityratios_true[index_hq]/deno_true - fake_densityratios_esti[index_hq]/deno_esti)**2))
        dre_errors_lq[nSim] = np.sqrt(np.mean((fake_densityratios_true[index_lq]/deno_true - fake_densityratios_esti[index_lq]/deno_esti)**2))

        # print("True avg. density ratios, HQ %f, LQ %f" % (np.mean(fake_densityratios_true[index_hq]), np.mean(fake_densityratios_true[index_lq])))
        # print("Esti avg. density ratios, HQ %f, LQ %f" % (np.mean(fake_densityratios_esti[index_hq]), np.mean(fake_densityratios_esti[index_lq])))

        ## estimated/true density ratios of testing samples, NFAKE fake samples, HQ fake samples, LQ fake samples
        ''' mean of density ratios on 10K real/fake samples '''
        esti_avg_densityratio[nSim] = np.array([np.mean(test_densityratios_esti), np.mean(fake_densityratios_esti), np.mean(fake_densityratios_esti[index_hq]), np.mean(fake_densityratios_esti[index_lq])])
        true_avg_densityratio[nSim] = np.array([np.mean(test_densityratios_true), np.mean(fake_densityratios_true), np.mean(fake_densityratios_true[index_hq]), np.mean(fake_densityratios_true[index_lq])])


        # deno_test_esti = np.sum(test_densityratios_esti); deno_test_true = np.sum(test_densityratios_true)
        # deno_fake_esti = np.sum(fake_densityratios_esti); deno_fake_true = np.sum(fake_densityratios_true)
        # esti_avg_densityratio[nSim] = np.array([np.mean(test_densityratios_esti/deno_test_esti), np.mean(fake_densityratios_esti/deno_fake_esti), np.mean(fake_densityratios_esti[index_hq]/deno_fake_esti), np.mean(fake_densityratios_esti[index_lq]/deno_fake_esti)])
        # true_avg_densityratio[nSim] = np.array([np.mean(test_densityratios_true/deno_test_true), np.mean(fake_densityratios_true/deno_fake_true), np.mean(fake_densityratios_true[index_hq]/deno_fake_true), np.mean(fake_densityratios_true[index_lq]/deno_fake_true)])




    #-----------------------------------------------------------------------------
    # compute the distance between the disc and its optimality
    if args.DRE not in ["None", "GT"] and args.compute_disc_err:
        disc_errors_comp_truediscscores_filename = save_objects_folder + "/disc_error_comp_truediscscores_epochGAN_" + str(epoch_GAN) + "_nfake_" + str(NFAKE) + "_gmm_ncomp_" + str(gmm_ncomp) + "_nSim_" + str(nSim) + ".pkl"
        disc_errors_comp_fakesamples_filename = save_objects_folder + "/disc_error_comp_fakesamples_epochGAN_" + str(epoch_GAN) + "_nSim_" + str(nSim) + ".pkl"

        #evaluate the optimal discriminator on NFAKE fake images
        if not os.path.isfile(disc_errors_comp_fakesamples_filename):
            fake_samples = fn_sampleGAN(NFAKE, samp_batch_size) #draw many samples from GAN directly
            with open(disc_errors_comp_fakesamples_filename, "wb") as file:
                pickle.dump(fake_samples, file)
        else:
            with open(disc_errors_comp_fakesamples_filename, "rb") as file:
                fake_samples = pickle.load(file)

        if not os.path.isfile(disc_errors_comp_truediscscores_filename):
            gmm_fakescore = gmm_eval_parallel(fake_samples)
            fake_disc_true = np.divide(p_r(fake_samples), p_r(fake_samples) + np.exp(gmm_fakescore))
            with open(disc_errors_comp_truediscscores_filename, "wb") as file:
                pickle.dump(fake_disc_true, file)
        else:
            with open(disc_errors_comp_truediscscores_filename, "rb") as file:
                fake_disc_true = pickle.load(file)

        n_samples = fake_samples.shape[0]
        batch_size_tmp = 1000
        dataset_tmp = custom_dataset(fake_samples)
        dataloader_tmp = torch.utils.data.DataLoader(dataset_tmp, batch_size=batch_size_tmp, shuffle=False, num_workers=0)
        data_iter = iter(dataloader_tmp)
        fake_disc_esti = np.zeros((n_samples+batch_size_tmp, 1))

        # print("\n Begin computing density ratio for images >>")
        netD.eval()
        with torch.no_grad():
            tmp = 0
            while tmp < n_samples:
                batch_samples,_ = data_iter.next()
                batch_samples = batch_samples.type(torch.float).to(device)
                disc_probs = netD(batch_samples).cpu().detach().numpy()
                # disc_probs = np.clip(disc_probs.astype(np.float), 1e-14, 1 - 1e-14)
                fake_disc_esti[tmp:(tmp+batch_size_tmp)] = disc_probs
                tmp += batch_size_tmp
            #end while
        fake_disc_esti = (fake_disc_esti[0:n_samples]).reshape(-1)

        disc_errors_all[nSim] = np.sqrt(np.mean((fake_disc_true - fake_disc_esti)**2))


    #-----------------------------------------------------------------------------
    # Ground truth DRE
    if args.DRE=="GT":
        def comp_density_ratio(samples):
           return np.divide(p_r(samples), np.exp(gmm_eval_parallel(samples))+1e-20)



    ###############################################################################
    # Different sampling method
    ###############################################################################
    ##########################################
    # Rejection Sampling: "Discriminator Rejection Sampling"; based on https://github.com/shinseung428/DRS_Tensorflow/blob/master/config.py
    if args.Sampling == "RS":
        def fn_enhanceSampler(nfake, batch_size=samp_batch_size):
            ## Burn-in Stage
            n_burnin = 50000
            burnin_samples = fn_sampleGAN(n_burnin, batch_size=samp_batch_size)
            burnin_densityratios = comp_density_ratio(burnin_samples)
            print((burnin_densityratios.min(),np.median(burnin_densityratios),burnin_densityratios.max()))
            M_bar = np.max(burnin_densityratios)
            del burnin_samples, burnin_densityratios; gc.collect()
            torch.cuda.empty_cache()
            ## Burn-in with real images
            burnin_densityratios2 = comp_density_ratio(train_samples_tar)
            print((burnin_densityratios2.min(),np.median(burnin_densityratios2),burnin_densityratios2.max()))
            # M_bar = np.max([np.max(burnin_densityratios),M_bar])
            ## Rejection sampling
            enhanced_samples = np.zeros((1, n_features)) #initilize
            pb = SimpleProgressBar()
            num_samples = 0
            while num_samples < nfake:
                batch_samples = fn_sampleGAN(batch_size, batch_size)
                batch_ratios = comp_density_ratio(batch_samples)
                M_bar = np.max([M_bar, np.max(batch_ratios)])
                #threshold
                if args.DRE in ["disc", "disc_MHcal","disc_KeepTrain"]:
                    epsilon_tmp = 1e-8;
                    D_tilde_M = np.log(M_bar)
                    batch_F = np.log(batch_ratios) - D_tilde_M - np.log(1-np.exp(np.log(batch_ratios)-D_tilde_M-epsilon_tmp))
                    gamma_tmp = np.percentile(batch_F, 95) #80 percentile of each batch; follow DRS's setting
                    batch_F_hat = batch_F - gamma_tmp
                    batch_p = 1/(1+np.exp(-batch_F_hat))
                else:
                    batch_p = batch_ratios/M_bar
                batch_psi = np.random.uniform(size=batch_size).reshape(-1,1)
                indx_accept = np.where((batch_psi<=batch_p)==True)[0]
                if len(indx_accept)>0:
                    enhanced_samples = np.concatenate((enhanced_samples, batch_samples[indx_accept]))
                num_samples=len(enhanced_samples)-1
                del batch_samples, batch_ratios; gc.collect()
                torch.cuda.empty_cache()
                pb.update(np.min([float(num_samples)*100/nfake, 100]))
            return enhanced_samples[1:(nfake+1)] #remove the first all zero array

    ##########################################
    # MCMC, Metropolis-Hastings algorithm: MH-GAN
    elif args.Sampling == "MH":
        # trainloader_MH = torch.utils.data.DataLoader(train_samples_tar, batch_size=samp_batch_size, shuffle=True)
        trainloader_MH = torch.utils.data.DataLoader(test_samples_tar, batch_size=samp_batch_size, shuffle=True)
        def fn_enhanceSampler(nfake, batch_size=samp_batch_size):
            # enhanced_samples = np.zeros((1, n_features)) #initilize
            enhanced_samples = np.zeros((nfake, n_features)) #initilize
            num_samples = 0
            while num_samples < nfake:
                data_iter = iter(trainloader_MH)
                batch_samples_new = data_iter.next()
                batch_samples_new = batch_samples_new.cpu().detach().numpy()
                batch_update_flags = np.zeros(batch_size) #if a sample in a batch is updated during MH, replace corresponding entry with 1
                for k in tqdm(range(MH_K)):
                    if not MH_mute:
                        print((k, num_samples))
                    batch_samples_old = fn_sampleGAN(batch_size, batch_size)
                    batch_U = np.random.uniform(size=batch_size).reshape(-1,1)
                    batch_ratios_old = comp_density_ratio(batch_samples_old)
                    batch_ratios_new = comp_density_ratio(batch_samples_new)
                    # print(np.concatenate((batch_ratios_old[0:10], batch_ratios_new[0:10]), axis=1))
                    batch_p = batch_ratios_old/(batch_ratios_new+1e-14)
                    batch_p[batch_p>1]=1
                    indx_accept = np.where((batch_U<=batch_p)==True)[0]
                    if len(indx_accept)>0:
                        batch_samples_new[indx_accept] = batch_samples_old[indx_accept]
                        batch_update_flags[indx_accept] = 1 #if a sample in a batch is updated during MH, replace corresponding entry with 1
                indx_updated = np.where(batch_update_flags==1)[0]
                # enhanced_samples = np.concatenate((enhanced_samples, batch_samples_new[indx_updated]))
                # num_samples=len(enhanced_samples)-1
                enhanced_samples[num_samples:(num_samples+len(indx_updated))] = batch_samples_new[indx_updated]
                num_samples += len(indx_updated)
                del batch_samples_new, batch_samples_old; gc.collect()
                torch.cuda.empty_cache()
            # return enhanced_samples[1:(nfake+1)] #remove the first all zero array
            return enhanced_samples #remove the first all zero array

    ##########################################
    # Sampling-Importance Resampling
    elif args.Sampling == "SIR":
       def fn_enhanceSampler(nfake, batch_size=samp_batch_size):
           enhanced_samples = fn_sampleGAN(nfake*10, batch_size)
           enhanced_ratios = comp_density_ratio(enhanced_samples)
           weights = enhanced_ratios / np.sum(enhanced_ratios) #normlaize to [0,1]
           resampl_indx = np.random.choice(a = np.arange(len(weights)), size = nfake, replace = True, p = weights.reshape(weights.shape[0]))
           enhanced_samples = enhanced_samples[resampl_indx]
           return enhanced_samples



    ###############################################################################
    # Hyper-parameter selection
    ###############################################################################
    #compute average density ratio on validaiton set; for selecting lambda
    if (args.DRE !="None" or args.Sampling != "None") and args.DRE != "GT":
        valid_densityratios = comp_density_ratio(valid_samples_tar)
        valid_densityratios_all.extend(list(valid_densityratios))
        train_densityratios = comp_density_ratio(train_samples_tar)
        train_densityratios_all.extend(list(train_densityratios))
        ks_test = ks_2samp(train_densityratios.reshape(-1), valid_densityratios.reshape(-1))
        ks_test_results[nSim,0] = ks_test.statistic
        ks_test_results[nSim,1] = ks_test.pvalue





    ###############################################################################
    # Evaluation each subsampling method
    ###############################################################################
    if args.DRE == "None" or args.Sampling == "None":
        print("Directly sampling from GAN >>>")
        fake_samples = fn_sampleGAN(NFAKE, samp_batch_size) #fake images before re-selecting
    else:
        print("Start enhanced sampling >>>")
        fake_samples = fn_enhanceSampler(NFAKE, batch_size=samp_batch_size)



    #-------------------------------------------
    # Visualization
    if args.show_visualization:
        if args.DRE in ['DRE_BARR', 'DRE_SP']:
            filename = save_images_folder + '/'+ args.DRE + "+" + args.Sampling + "_epochGAN_" + str(args.epoch_gan) + "_epochDRE_" + str(args.epoch_DRE) + "_lambda_" + str(args.lambda_DRE)+ "_PreNetDRE_" + str(args.TrainPreNetDRE) + "_" + args.DR_final_ActFn + '_nSim_' + str(nSim) +'.pdf'
        elif args.DRE in ['DRE_uLSIF', 'DRE_DSKL']:
            filename = save_images_folder + '/' + args.DRE + "+" + args.Sampling + "_epochGAN_" + str(args.epoch_gan) + "_epochDRE_" + str(args.epoch_DRE) + "_PreNetDRE_" + str(args.TrainPreNetDRE) + "_" + args.DR_final_ActFn + '_nSim_' + str(nSim) + '.pdf'
        elif args.DRE == "GT":
            filename = save_images_folder + '/' + args.DRE + "+" + args.Sampling + "_gmm_ncomp_" + str(gmm_ncomp) + "_epochGAN_" + str(args.epoch_gan) + '_nSim_' + str(nSim) + '.pdf'
        else:
            filename = save_images_folder + '/' + args.DRE + "+" + args.Sampling + "_epochGAN_" + str(args.epoch_gan) + '_nSim_' + str(nSim) + '.pdf'
        ScatterPoints(test_samples_tar, fake_samples, filename, plot_real_samples = False)

    #-------------------------------------------
    # Compute number of recovered modes and number of good fake samples
    l2_dis_fake_samples = np.zeros((NFAKE, n_comp_tar)) #l2 distance between a fake sample and a mode
    for i in tqdm(range(NFAKE)):
        for j in range(n_comp_tar):
            l2_dis_fake_samples[i,j] = np.sqrt(np.sum((fake_samples[i]-means_tar[j])**2))
    min_l2_dis_fake_samples = np.min(l2_dis_fake_samples, axis=1)
    indx_cloeset_modes = np.argmin(l2_dis_fake_samples, axis=1) #indx of the closet mode in 25 modes for each sample

    prop_good_samples[nSim] = sum(min_l2_dis_fake_samples<quality_threshold)/NFAKE*100 #proportion of good fake samples
    prop_recovered_modes[nSim] = len(list(set(indx_cloeset_modes)))/n_comp_tar*100


    #-------------------------------------------
    # Check how many fake samples actually appear in the training set
    train_samples_tar = np.round(train_samples_tar,decimals=6)
    fake_samples = np.round(fake_samples,decimals=6)

    for i in fake_samples:
        if i in train_samples_tar:
            nfake_in_train[nSim]+=1


# end for nSim
stop = timeit.default_timer()
print("\n Time elapses: {}s".format(stop - start))

if args.DRE != "None" or args.Sampling != "None":
    train_densityratios_all = np.array(train_densityratios_all)
    #print("Avg/Med/STD of density ratio on training set over %d Sims: %.3f, %.3f, %.3f" % (args.NSIM, np.median(train_densityratios_all), np.mean(train_densityratios_all), np.std(train_densityratios_all)))
    valid_densityratios_all = np.array(valid_densityratios_all)
    #print("Avg/Med/STD of density ratio on validation set over %d Sims: %.3f, %.3f, %.3f" % (args.NSIM, np.median(valid_densityratios_all), np.mean(valid_densityratios_all), np.std(valid_densityratios_all)))
    #print("Kolmogorov-Smirnov test in %d Sims: avg. stat. %.3E, avg. pval %.3E" % (args.NSIM, np.mean(ks_test_results[:,0]), np.mean(ks_test_results[:,1])))
    print("\n KS test resutls (test_stat, p_value) >>>\n")
    print(ks_test_results)


if args.DRE not in ["None", "GT"] and args.compute_dre_err:
    #print("\n DRE errors on the test set in %d rounds >>>" % args.NSIM)
    #print(dre_errors_all)
    print("\n Avg. DRE erros: %e" % np.mean(dre_errors_all))
    print("\n Avg. HQ DRE erros: %e" % np.mean(dre_errors_hq))
    print("\n Avg. LQ DRE erros: %e" % np.mean(dre_errors_lq))

    temp_filename = save_objects_folder + "/dre_errors_comp_" + args.DRE + ".csv"
    with open(temp_filename, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([args.epoch_DRE, np.mean(dre_errors_all), np.mean(dre_errors_hq), np.mean(dre_errors_lq)])

    temp_filename = save_objects_folder + "/esti_densityratios_" + args.DRE + ".csv"
    with open(temp_filename, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        # writer.writerow([args.epoch_DRE, np.mean(esti_avg_densityratio[:,0]), np.mean(esti_avg_densityratio[:,1]), np.mean(esti_avg_densityratio[:,2]), np.mean(esti_avg_densityratio[:,3])])
        writer.writerow([args.epoch_DRE, np.format_float_scientific(np.mean(esti_avg_densityratio[:,0])), np.format_float_scientific(np.mean(esti_avg_densityratio[:,1])), np.format_float_scientific(np.mean(esti_avg_densityratio[:,2])), np.format_float_scientific(np.mean(esti_avg_densityratio[:,3]))])

    temp_filename = save_objects_folder + "/true_densityratios_" + args.DRE + ".csv"
    with open(temp_filename, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        # writer.writerow([args.epoch_DRE, np.mean(true_avg_densityratio[:,0]), np.mean(true_avg_densityratio[:,1]), np.mean(true_avg_densityratio[:,2]), np.mean(true_avg_densityratio[:,3])])
        writer.writerow([args.epoch_DRE, np.format_float_scientific(np.mean(true_avg_densityratio[:,0])), np.format_float_scientific(np.mean(true_avg_densityratio[:,1])), np.format_float_scientific(np.mean(true_avg_densityratio[:,2])), np.format_float_scientific(np.mean(true_avg_densityratio[:,3]))])


if args.DRE not in ["None", "GT"] and args.compute_disc_err:
    print("\n Disc errors on the test set in %d rounds >>>" % args.NSIM)
    print(disc_errors_all)
    print("\n Avg. Disc erros: %.3f" % np.mean(disc_errors_all))

    temp_filename = save_objects_folder + "/disc_errors_comp_" + args.DRE + ".csv"
    with open(temp_filename, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow([args.epoch_KeepTrain, np.mean(disc_errors_all)])


print("\n Prop. of good quality samples>>>\n")
print(prop_good_samples)
print("\n Prop. good samples over %d Sims: %.1f (%.1f)" % (args.NSIM, np.mean(prop_good_samples), np.std(prop_good_samples)))
print("\n Prop. of recovered modes>>>\n")
print(prop_recovered_modes)
print("\n Prop. recovered modes over %d Sims: %.1f (%.1f)" % (args.NSIM, np.mean(prop_recovered_modes), np.std(prop_recovered_modes)))


print("\n On average, %d fake samples appear in the training set" % np.mean(nfake_in_train))
