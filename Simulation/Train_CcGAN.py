"""
Train a regression DCGAN

"""

import torch
import torch.nn as nn
import numpy as np
import os
import timeit
from PIL import Image

from utils import *
from opts import parse_opts

''' Settings '''
args = parse_opts()
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# some parameters in opts
niters = args.niters_gan
resume_niters = args.resume_niters_gan
dim_gan = args.dim_gan
lr_g = args.lr_gan
lr_d = args.lr_gan
save_niters_freq = args.save_niters_freq
batch_size_disc = args.batch_size_disc
batch_size_gene = args.batch_size_gene

threshold_type = args.threshold_type
nonzero_soft_weight_threshold = args.nonzero_soft_weight_threshold




def train_CcGAN(kernel_sigma, kappa, train_samples, train_labels, netG, netD, save_images_folder, save_models_folder = None, plot_in_train=False, samples_tar_eval = None, angle_grid_eval = None, fig_size=5, point_size=None):

    netG = netG.to(device)
    netD = netD.to(device)

    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))

    if save_models_folder is not None and resume_niters>0:
        save_file = save_models_folder + "/CcGAN_checkpoint_intrain/CcGAN_checkpoint_niter_{}.pth".format(resume_niters)
        checkpoint = torch.load(save_file)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if

    #################
    unique_train_labels = np.sort(np.array(list(set(train_labels))))

    start_time = timeit.default_timer()
    for niter in range(resume_niters, niters):

        '''  Train Discriminator   '''
        ## randomly draw batch_size_disc y's from unique_train_labels
        batch_target_labels_raw = np.random.choice(unique_train_labels, size=batch_size_disc, replace=True)
        ## add Gaussian noise; we estimate image distribution conditional on these labels
        batch_epsilons = np.random.normal(0, kernel_sigma, batch_size_disc)
        batch_target_labels = batch_target_labels_raw + batch_epsilons

        ## only for similation
        batch_target_labels[batch_target_labels<0] = batch_target_labels[batch_target_labels<0] + 1
        batch_target_labels[batch_target_labels>1] = batch_target_labels[batch_target_labels>1] - 1

        ## find index of real images with labels in the vicinity of batch_target_labels
        ## generate labels for fake image generation; these labels are also in the vicinity of batch_target_labels
        batch_real_indx = np.zeros(batch_size_disc, dtype=int) #index of images in the datata; the labels of these images are in the vicinity
        batch_fake_labels = np.zeros(batch_size_disc)


        for j in range(batch_size_disc):
            ## index for real images
            if threshold_type == "hard":
                indx_real_in_vicinity = np.where(np.abs(train_labels-batch_target_labels[j])<= kappa)[0]
            else:
                # reverse the weight function for SVDL
                indx_real_in_vicinity = np.where((train_labels-batch_target_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold)/kappa)[0]

            ## if the max gap between two consecutive ordered unique labels is large, it is possible that len(indx_real_in_vicinity)<1
            while len(indx_real_in_vicinity)<1:
                batch_epsilons_j = np.random.normal(0, kernel_sigma, 1)
                batch_target_labels[j] = batch_target_labels_raw[j] + batch_epsilons_j
                ## only for similation
                if batch_target_labels[j]<0:
                    batch_target_labels[j] = batch_target_labels[j] + 1
                if batch_target_labels[j]>1:
                    batch_target_labels[j] = batch_target_labels[j] - 1
                ## index for real images
                if threshold_type == "hard":
                    indx_real_in_vicinity = np.where(np.abs(train_labels-batch_target_labels[j])<= kappa)[0]
                else:
                    # reverse the weight function for SVDL
                    indx_real_in_vicinity = np.where((train_labels-batch_target_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold)/kappa)[0]
            #end while len(indx_real_in_vicinity)<1

            assert len(indx_real_in_vicinity)>=1

            batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]

            ## labels for fake images generation
            if threshold_type == "hard":
                lb = batch_target_labels[j] - kappa
                ub = batch_target_labels[j] + kappa
            else:
                lb = batch_target_labels[j] - np.sqrt(-np.log(nonzero_soft_weight_threshold)/kappa)
                ub = batch_target_labels[j] + np.sqrt(-np.log(nonzero_soft_weight_threshold)/kappa)
            lb = max(0.0, lb); ub = min(ub, 1.0)
            assert lb<=ub
            assert lb>=0 and ub>=0
            assert lb<=1 and ub<=1
            batch_fake_labels[j] = np.random.uniform(lb, ub, size=1)[0]
        #end for j

        ## draw the real image batch from the training set
        batch_real_samples = train_samples[batch_real_indx]
        batch_real_labels = train_labels[batch_real_indx]
        batch_real_samples = torch.from_numpy(batch_real_samples).type(torch.float).to(device)
        batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float).to(device)

        ## generate the fake image batch
        batch_fake_labels = torch.from_numpy(batch_fake_labels).type(torch.float).to(device)
        z = torch.randn(batch_size_disc, dim_gan, dtype=torch.float).to(device)
        batch_fake_samples = netG(z, batch_fake_labels)

        ## target labels on gpu
        batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(device)

        ## weight vector
        if threshold_type == "soft":
            real_weights = torch.exp(-kappa*(batch_real_labels-batch_target_labels)**2).to(device)
            fake_weights = torch.exp(-kappa*(batch_fake_labels-batch_target_labels)**2).to(device)
        else:
            real_weights = torch.ones(batch_size_disc, dtype=torch.float).to(device)
            fake_weights = torch.ones(batch_size_disc, dtype=torch.float).to(device)
        #end if threshold type

        # forward pass

        real_dis_out = netD(batch_real_samples, batch_target_labels)
        fake_dis_out = netD(batch_fake_samples.detach(), batch_target_labels)

        d_loss = - torch.mean(real_weights.view(-1) * torch.log(real_dis_out.view(-1)+1e-20)) - torch.mean(fake_weights.view(-1) * torch.log(1 - fake_dis_out.view(-1)+1e-20))

        optimizerD.zero_grad()
        d_loss.backward()
        optimizerD.step()


        '''  Train Generator   '''
        netG.train()

        # generate fake images
        ## randomly draw batch_size_disc y's from unique_train_labels
        batch_target_labels_raw = np.random.choice(unique_train_labels, size=batch_size_gene, replace=True)
        ## add Gaussian noise; we estimate image distribution conditional on these labels
        batch_epsilons = np.random.normal(0, kernel_sigma, batch_size_gene)
        batch_target_labels = batch_target_labels_raw + batch_epsilons
        batch_target_labels[batch_target_labels<0] = batch_target_labels[batch_target_labels<0] + 1
        batch_target_labels[batch_target_labels>1] = batch_target_labels[batch_target_labels>1] - 1
        batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(device)

        z = torch.randn(batch_size_gene, dim_gan, dtype=torch.float).to(device)
        batch_fake_samples = netG(z, batch_target_labels)

        # loss
        dis_out = netD(batch_fake_samples, batch_target_labels)
        g_loss = - torch.mean(torch.log(dis_out+1e-20))

        # backward
        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()

        # print loss
        if niter%100 == 0:
            print ("CcGAN: [Iter %d/%d] [D loss: %.4e] [G loss: %.4e] [real prob: %.3f] [fake prob: %.3f] [Time: %.4f]" % (niter+1, niters, d_loss.item(), g_loss.item(), real_dis_out.mean().item(), fake_dis_out.mean().item(), timeit.default_timer()-start_time))

        # output fake images during training
        if plot_in_train and (niter+1)%100 == 0:
            netG.eval()
            assert save_images_folder is not None
            z = torch.randn(samples_tar_eval.shape[0], dim_gan, dtype=torch.float).to(device)

            labels = np.random.choice(angle_grid_eval/(2*np.pi), size=samples_tar_eval.shape[0], replace=True)
            labels = torch.from_numpy(labels).type(torch.float).to(device)
            prop_samples = netG(z, labels).cpu().detach().numpy()
            filename = save_images_folder + '/{}.png'.format(niter+1)
            ScatterPoints(samples_tar_eval, prop_samples, filename, fig_size=fig_size, point_size=point_size)

            labels = np.random.choice(angle_grid_eval/(2*np.pi), size=1, replace=True)
            labels = np.repeat(labels, samples_tar_eval.shape[0])
            labels = torch.from_numpy(labels).type(torch.float).to(device)
            prop_samples = netG(z, labels).cpu().detach().numpy()
            filename = save_images_folder + '/{}_{}.png'.format(niter+1, labels[0]*(2*np.pi))
            ScatterPoints(samples_tar_eval, prop_samples, filename, fig_size=fig_size, point_size=point_size)

        if save_models_folder is not None and ((niter+1) % save_niters_freq == 0 or (niter+1) == niters):
            save_file = save_models_folder + "/CcGAN_checkpoint_intrain/CcGAN_checkpoint_niters_{}.pth".format(niter+1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)
    #end for niter

    return netG, netD



def SampCcGAN_given_label(netG, label, path=None, NFAKE = 10000, batch_size = 500, num_features=2):
    '''
    label: normalized label in [0,1]
    '''
    if batch_size>NFAKE:
        batch_size = NFAKE
    fake_samples = np.zeros((NFAKE+batch_size, num_features), dtype=np.float)
    netG=netG.to(device)
    netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < NFAKE:
            z = torch.randn(batch_size, dim_gan, dtype=torch.float).to(device)
            y = np.ones(batch_size) * label
            y = torch.from_numpy(y).type(torch.float).view(-1,1).to(device)
            batch_fake_samples = netG(z, y)
            fake_samples[tmp:(tmp+batch_size)] = batch_fake_samples.cpu().detach().numpy()
            tmp += batch_size

    #remove extra entries
    fake_samples = fake_samples[0:NFAKE]
    fake_angles = np.ones(NFAKE) * label #use assigned label

    if path is not None:
        raw_fake_samples = (fake_samples*0.5+0.5)*255.0
        raw_fake_samples = raw_fake_samples.astype(np.uint8)
        for i in range(NFAKE):
            filename = path + '/' + str(i) + '.jpg'
            im = Image.fromarray(raw_fake_samples[i][0], mode='L')
            im = im.save(filename)

    return fake_samples, fake_angles
