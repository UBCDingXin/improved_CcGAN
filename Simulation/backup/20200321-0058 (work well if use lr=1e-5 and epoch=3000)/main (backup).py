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

''' Data Generation '''
parser.add_argument('--n_gaussians', type=int, default=120,
                    help = "Number of Gaussians (number of angles's).") #half of them will be used for training and the rest are for testing
parser.add_argument('--n_samp_per_gaussian_train', type=int, default=10) # n_gaussians*n_rsamp_per_gaussian = ntrain
parser.add_argument('--radius', type=float, default=2.0)
parser.add_argument('--sigma_gaussian', type=float, default=0.01)


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
parser.add_argument('--kernel_sigma', type=float, default=-1.0,
                    help='If kernel_sigma<0, then use rule-of-thumb formula to compute the sigma.')
parser.add_argument('--kappa', type=float, default=1)


''' Sampling and Evaluation '''
parser.add_argument('--eval', action='store_true', default=False) #evaluation fake samples
parser.add_argument('--n_gaussians_eval', type=int, default=360) #number of labels for evaluation
parser.add_argument('--n_samp_per_gaussian_eval', type=int, default=100) # number of fake samples for each Gaussian
parser.add_argument('--samp_batch_size_eval', type=int, default=100)


''' Visualization '''
# parser.add_argument('--show_visualization', action='store_true', default=False,
#                     help='Plot fake samples in 2D coordinate')
parser.add_argument('--n_gaussians_plot', type=int, default=12) #number of unseen labels for plotting
parser.add_argument('--n_samp_per_gaussian_plot', type=int, default=100) # number of fake samples for each Gaussian
parser.add_argument('--samp_batch_size_plot', type=int, default=100)
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
# angles for training
angle_grid_train = np.linspace(0, 2*np.pi, n_gaussians+1) # 12 clock is the start point; last angle is dropped to avoid overlapping.
angle_grid_train = angle_grid_train[0:n_gaussians]
# angles for evaluation
# angle_grid_eval = np.linspace(0, 2*np.pi, n_gaussians_eval+1)
# angle_grid_eval = angle_grid_eval[0:n_gaussians_eval]
unseen_angles_all = np.linspace(0, 2*np.pi, n_gaussians*100+1)
unseen_angles_all = np.setdiff1d(unseen_angles_all[0:n_gaussians*100], angle_grid_train)
angle_grid_eval = np.zeros(args.n_gaussians_eval)
for i in range(args.n_gaussians_eval):
    quantile_i = (i+1)/args.n_gaussians_eval
    angle_grid_eval[i] = np.quantile(unseen_angles_all, quantile_i, interpolation='nearest')
assert len(np.intersect1d(angle_grid_eval, angle_grid_train))==0
# angles for plotting
unseen_angles_all = np.linspace(0, 2*np.pi, n_gaussians*100+1)
unseen_angles_all = np.setdiff1d(unseen_angles_all[0:n_gaussians*100], angle_grid_train)
unseen_angle_grid_plot = np.zeros(args.n_gaussians_plot)
for i in range(args.n_gaussians_plot):
    quantile_i = (i+1)/args.n_gaussians_plot
    unseen_angle_grid_plot[i] = np.quantile(unseen_angles_all, quantile_i, interpolation='nearest')
assert len(np.intersect1d(unseen_angle_grid_plot, angle_grid_train))==0
# standard deviation of each Gaussian
sigma_gaussian = args.sigma_gaussian
### threshold to determine high quality samples
# quality_threshold = sigma_gaussian*7 #good samples are within 7 standard deviation
# distance_between_2modes = args.radius*np.sqrt(2-2*np.cos(2*np.pi/n_gaussians))
# quality_threshold = min(sigma_gaussian*5, distance_between_2modes) + distance_between_2modes
# print("Quality threshold is {}".format(quality_threshold))

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
# Plot Settings
fig_size=7
point_size = 25

#-------------------------------
# output folders
save_models_folder = wd + '/Output/saved_models/'
os.makedirs(save_models_folder,exist_ok=True)
save_images_folder = wd + '/Output/saved_images/'
os.makedirs(save_images_folder,exist_ok=True)
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

avg_two_w_dist = np.zeros(args.nsim)

print("\n Begin The Experiment; Start Training {} >>>".format(args.GAN))
start = timeit.default_timer()
for nSim in range(args.nsim):
    print("Round %s" % (nSim))
    np.random.seed(nSim) #set seed for current simulation

    ###############################################################################
    # Data generation and dataloaders
    ###############################################################################
    samples_train, angles_train, means_train = generate_data(args.n_samp_per_gaussian_train, angle_grid_train) #this angles_train is not normalized; normalize if args.GAN is not cGAN.
    samples_plot_in_train, _, _ = generate_data(10, unseen_angle_grid_plot)

    # plot training samples and their theoretical means
    filename_tmp = save_images_folder + 'samples_train_with_means_nSim_' + str(nSim) + '.png'
    if not os.path.isfile(filename_tmp):
        plt.switch_backend('agg')
        mpl.style.use('seaborn')
        plt.figure(figsize=(fig_size, fig_size), facecolor='w')
        plt.grid(b=True)
        plt.scatter(samples_train[:, 0], samples_train[:, 1], c='blue', edgecolor='none', alpha=0.5, s=point_size, label="Real samples")
        plt.scatter(means_train[:, 0], means_train[:, 1], c='red', edgecolor='none', alpha=1, s=point_size, label="Means")
        plt.legend(loc=1)
        plt.savefig(filename_tmp)

    # preprocessing on labels
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
    else:
        # angles_train = angles_train
        angles_train = angles_train/(2*np.pi)

        # rule-of-thumb for the bandwidth selection
        if args.kernel_sigma<0:
            std_angles_train = np.std(angles_train)
            q75_angles_train, q25_angles_train = np.percentile(angles_train, [75 ,25])
            iqr_angles_train = q75_angles_train - q25_angles_train
            # args.kernel_sigma = 1.06*min(std_angles_train, iqr_angles_train/1.34)*(len(angles_train))**(-1/5)
            args.kernel_sigma = 0.9*min(std_angles_train, iqr_angles_train/1.34)*(len(angles_train))**(-1/5) # Silverman's (1986) rule of thumb
            print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")
            print("\n The std and iqr/1.34 of {} normalized angles are {} and {} so the kernel sigma is {}".format(len(angles_train), std_angles_train, iqr_angles_train/1.34, args.kernel_sigma))

            # args.kernel_sigma = 1/args.n_gaussians

        if args.kappa < 0:
            if args.threshold_type=="hard":
                args.kappa = 1/args.n_gaussians
                # args.kappa = 2/args.n_gaussians
                # args.kappa = (1/args.n_gaussians + 3*args.kernel_sigma) * (len(angles_train)/args.n_gaussians)**(-1/2)
            else:
                # args.kappa = 1/((1/args.n_gaussians)**2)
                args.kappa = 1/((2/args.n_gaussians)**2)

    train_dataset = custom_dataset(samples_train, angles_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size_gan, shuffle=True, num_workers=0)


    ###############################################################################
    # Train a GAN model
    ###############################################################################
    print("{}/{}, {}, Sigma is {}, Kappa is {}".format(nSim+1, args.nsim, args.threshold_type, args.kernel_sigma, args.kappa))

    if args.GAN in ['CcGAN']:
        # save_GANimages_InTrain_folder = wd + '/Output/saved_images/' + args.GAN + '_' + args.threshold_type + '_' + str(args.kernel_sigma) + '_' + str(args.kappa) + '_InTrain/'
        save_GANimages_InTrain_folder = wd + '/Output/saved_images/{}_{}_{}_{}_nSim_{}_InTrain'.format(args.GAN, args.threshold_type, args.kernel_sigma, args.kappa, nSim)
    else:
        save_GANimages_InTrain_folder = wd + '/Output/saved_images/' + args.GAN + '_nSim_' + str(nSim) + '_InTrain/'
    os.makedirs(save_GANimages_InTrain_folder,exist_ok=True)

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
            netG, netD, optimizerG, optimizerD = train_GAN(epoch_GAN, dim_GAN, train_dataloader, netG, netD, optimizerG, optimizerD, criterion, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan, device=device, plot_in_train=plot_in_train, save_images_folder = save_GANimages_InTrain_folder, samples_tar_eval = samples_plot_in_train, fig_size=fig_size, point_size=point_size)

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
            netG, netD, optimizerG, optimizerD = train_cGAN(epoch_GAN, dim_GAN, train_dataloader, netG, netD, optimizerG, optimizerD, criterion, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan, device=device, plot_in_train=plot_in_train, save_images_folder = save_GANimages_InTrain_folder, samples_tar_eval = samples_plot_in_train, angle_grid_eval = unseen_angle_grid_plot, num_classes=n_gaussians, unique_labels=unique_angles_train, label2class=angle2class, fig_size=fig_size, point_size=point_size)

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
            angle = np.single(angle*2*np.pi) #back to original scale of angle [0, 2*pi]
            fake_samples, _ = SampcGAN_given_label(netG, angle, unique_labels=unique_angles_train, label2class=angle2class, GAN_Latent_Length = dim_GAN, NFAKE = nfake, batch_size = batch_size, device=device)
            return fake_samples

    #----------------------------------------------
    # Concitnuous cGAN
    elif args.GAN == "CcGAN":
        Filename_GAN = save_models_folder + '/ckpt_' + args.GAN + '_epoch_' + str(args.epoch_gan) + '_SEED_' + str(args.seed) + '_' + args.threshold_type + '_' + str(args.kernel_sigma) + '_' + str(args.kappa) + '_nSim_' + str(nSim)
        if not os.path.isfile(Filename_GAN):
            netG = cont_cond_generator(ngpu=NGPU, nz=dim_GAN, out_dim=n_features, radius=radius)
            netD = cont_cond_discriminator(ngpu=NGPU, input_dim = n_features, radius=radius)

            optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_GAN, betas=(gan_Adam_beta1, gan_Adam_beta2))
            optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_GAN, betas=(gan_Adam_beta1, gan_Adam_beta2))

            # Start training
            netG, netD, optimizerG, optimizerD = train_CcGAN(args.kernel_sigma, args.threshold_type, args.kappa, epoch_GAN, dim_GAN, train_dataloader, netG, netD, optimizerG, optimizerD, save_images_folder=save_GANimages_InTrain_folder, save_models_folder = save_models_folder, ResumeEpoch = args.resumeTrain_gan, device=device, plot_in_train = plot_in_train, samples_tar_eval = samples_plot_in_train, angle_grid_eval = unseen_angle_grid_plot, fig_size=fig_size, point_size=point_size)

            # store model
            torch.save({
                'netG_state_dict': netG.state_dict(),
            }, Filename_GAN)
        else:
            print("Loading pre-trained generator >>>")
            checkpoint = torch.load(Filename_GAN)
            netG = cont_cond_generator(ngpu=NGPU, nz=dim_GAN, out_dim=n_features, radius=radius).to(device)
            netG.load_state_dict(checkpoint['netG_state_dict'])

        def fn_sampleGAN_given_label(nfake, label, batch_size):
            fake_samples, _ = SampCcGAN_given_label(netG, label, path=None, dim_GAN = dim_GAN, NFAKE = nfake, batch_size = batch_size, device=device)
            return fake_samples

    ###############################################################################
    # Evaluation
    ###############################################################################
    if args.eval:
        print("\n Start evaluation >>>")

        # # percentage of high quality and recovered modes
        # for i_ang in range(len(angle_grid_eval)):
        #     angle_curr = angle_grid_eval[i_ang]
        #     mean_curr = np.array([radius*np.sin(angle_curr), radius*np.cos(angle_curr)])
        #     fake_samples_curr = fn_sampleGAN_given_label(args.n_samp_per_gaussian_eval, angle_curr/(2*np.pi), batch_size=args.n_samp_per_gaussian_eval)
        #     mean_curr_repeat = np.repeat(mean_curr.reshape(1,n_features), args.n_samp_per_gaussian_eval, axis=0)
        #     assert mean_curr_repeat.shape[0]==args.n_samp_per_gaussian_eval and mean_curr_repeat.shape[1]==n_features
        #     assert fake_samples_curr.shape[0]==args.n_samp_per_gaussian_eval and fake_samples_curr.shape[1]==n_features
        #     #l2 distance between a fake sample and its mean
        #     l2_dis_fake_samples_curr = np.sqrt(np.sum((fake_samples_curr-mean_curr_repeat)**2, axis=1))
        #     assert len(l2_dis_fake_samples_curr)==args.n_samp_per_gaussian_eval
        #     if i_ang == 0:
        #         l2_dis_fake_samples = l2_dis_fake_samples_curr
        #     else:
        #         l2_dis_fake_samples = np.concatenate((l2_dis_fake_samples, l2_dis_fake_samples_curr))
        #
        #     # whether this mode is recovered?
        #     if sum(l2_dis_fake_samples_curr<=quality_threshold)>0:
        #         prop_recovered_modes[nSim] += 1
        # #end for i_ang
        # prop_recovered_modes[nSim] = (prop_recovered_modes[nSim]/len(angle_grid_eval))*100
        # prop_good_samples[nSim] = sum(l2_dis_fake_samples<=quality_threshold)/len(l2_dis_fake_samples)*100 #proportion of good fake samples


        # 2-Wasserstein Distance
        real_cov = np.eye(n_features)*sigma_gaussian**2 #covraiance matrix for each Gaussian
        for i_ang in tqdm(range(len(angle_grid_eval))):
            angle_curr = angle_grid_eval[i_ang]
            # the mean for current Gaussian (angle)
            real_mean_curr = np.array([radius*np.sin(angle_curr), radius*np.cos(angle_curr)])
            # sample from trained GAN
            fake_samples_curr = fn_sampleGAN_given_label(args.n_samp_per_gaussian_eval, angle_curr/(2*np.pi), batch_size=args.n_samp_per_gaussian_eval)
            # the sample mean and sample cov of fake samples with current label
            fake_mean_curr = np.mean(fake_samples_curr, axis = 0)
            fake_cov_curr = np.cov(fake_samples_curr.transpose())

            # print(real_mean_curr[0:5])
            # print(real_cov)
            # print(fake_mean_curr[0:5])
            # print(fake_cov_curr)

            # 2-W distance for current label
            two_w_dist_curr = two_wasserstein(real_mean_curr, fake_mean_curr, real_cov, fake_cov_curr, eps=1e-20)

            if i_ang == 0:
                two_w_dist_all = [two_w_dist_curr]
            else:
                two_w_dist_all.append(two_w_dist_curr)
        # end for i_ang
        avg_two_w_dist[nSim] = sum(two_w_dist_all)/len(two_w_dist_all) #average over all evaluation angles


        # ### visualize fake samples
        # if args.GAN == "CcGAN":
        #     filename_tmp = save_images_folder + '{}_real_fake_samples_{}_sigma_{}_kappa_{}_nSim_{}.png'.format(args.GAN, args.threshold_type, args.kernel_sigma, args.kappa, nSim)
        # else:
        #     filename_tmp = save_images_folder + '{}_real_fake_samples_nSim_{}.png'.format(args.GAN, nSim)
        #
        # fake_samples = np.zeros((args.n_gaussians_eval*args.n_samp_per_gaussian_eval, n_features))
        # for i_tmp in range(args.n_gaussians_eval):
        #     angle_curr = angle_grid_eval[i_tmp]
        #     fake_samples_curr = fn_sampleGAN_given_label(args.n_samp_per_gaussian_eval, angle_curr/(2*np.pi), batch_size=args.n_samp_per_gaussian_eval)
        #     if i_tmp == 0:
        #         fake_samples = fake_samples_curr
        #     else:
        #         fake_samples = np.concatenate((fake_samples, fake_samples_curr), axis=0)
        #
        # real_samples_eval, _, _ = generate_data(args.n_samp_per_gaussian_eval, angle_grid_eval)
        #
        # plt.switch_backend('agg')
        # mpl.style.use('seaborn')
        # plt.figure(figsize=(10, 10), facecolor='w')
        # plt.grid(b=True)
        # plt.scatter(real_samples_eval[:, 0], real_samples_eval[:, 1], c='blue', edgecolor='none', alpha=0.5, s=8, label="Real samples")
        # plt.scatter(fake_samples[:, 0], fake_samples[:, 1], c='red', edgecolor='none', alpha=1, s=8, label="Fake samples")
        # plt.legend()
        # plt.savefig(filename_tmp)

        ### visualize fake samples
        if args.GAN == "CcGAN":
            filename_tmp = save_images_folder + '{}_real_fake_samples_{}_sigma_{}_kappa_{}_nSim_{}.png'.format(args.GAN, args.threshold_type, args.kernel_sigma, args.kappa, nSim)
        else:
            filename_tmp = save_images_folder + '{}_real_fake_samples_nSim_{}.png'.format(args.GAN, nSim)

        fake_samples = np.zeros((args.n_gaussians_plot*args.n_samp_per_gaussian_plot, n_features))
        for i_tmp in range(args.n_gaussians_plot):
            angle_curr = unseen_angle_grid_plot[i_tmp]
            fake_samples_curr = fn_sampleGAN_given_label(args.n_samp_per_gaussian_plot, angle_curr/(2*np.pi), batch_size=args.n_samp_per_gaussian_plot)
            if i_tmp == 0:
                fake_samples = fake_samples_curr
            else:
                fake_samples = np.concatenate((fake_samples, fake_samples_curr), axis=0)

        real_samples_plot, _, _ = generate_data(args.n_samp_per_gaussian_plot, unseen_angle_grid_plot)

        plt.switch_backend('agg')
        mpl.style.use('seaborn')
        plt.figure(figsize=(fig_size, fig_size), facecolor='w')
        plt.grid(b=True)
        plt.scatter(real_samples_plot[:, 0], real_samples_plot[:, 1], c='blue', edgecolor='none', alpha=0.5, s=point_size, label="Real samples")
        plt.scatter(fake_samples[:, 0], fake_samples[:, 1], c='green', edgecolor='none', alpha=1, s=point_size, label="Fake samples")
        plt.legend(loc=1)
        plt.savefig(filename_tmp)

# for nSim
stop = timeit.default_timer()
print("GAN training finished; Time elapses: {}s".format(stop - start))



# print("\n Prop. of good quality samples>>>\n")
# print(prop_good_samples)
# print("\n Prop. good samples over %d Sims: %.1f (%.1f)" % (args.nsim, np.mean(prop_good_samples), np.std(prop_good_samples)))
# print("\n Prop. of recovered modes>>>\n")
# print(prop_recovered_modes)
# print("\n Prop. recovered modes over %d Sims: %.1f (%.1f)" % (args.nsim, np.mean(prop_recovered_modes), np.std(prop_recovered_modes)))

print("\n {}, Sigma is {}, Kappa is {}".format(args.threshold_type, args.kernel_sigma, args.kappa))
print("\r 2-Wasserstein Distance: %.3f (%.3f)"% (np.mean(avg_two_w_dist), np.std(avg_two_w_dist)))
print(avg_two_w_dist)
