
import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import os
import timeit

from utils import *
from opts import parse_opts

''' Settings '''
args = parse_opts()
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# some parameters in opts
loss_type = args.loss_type_gan
niters = args.niters_gan
resume_niters = args.resume_niters_gan
dim_gan = args.dim_gan
lr_g = args.lr_g_gan
lr_d = args.lr_d_gan
save_niters_freq = args.save_niters_freq
batch_size = min(args.batch_size_disc, args.batch_size_gene)
num_classes = args.cGAN_num_classes

NC = args.num_channels
IMG_SIZE = args.img_size


def train_cGAN(images, labels, netG, netD, save_images_folder, save_models_folder = None):

    netG = netG.to(device)
    netD = netD.to(device)

    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))

    trainset = IMGs_dataset(images, labels, normalize=True)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    unique_labels = np.sort(np.array(list(set(labels)))).astype(np.int)

    if save_models_folder is not None and resume_niters>0:
        save_file = save_models_folder + "/cGAN_checkpoint_intrain/cGAN_checkpoint_niters_{}.pth".format(resume_niters)
        checkpoint = torch.load(save_file)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if

    n_row=10
    z_fixed = torch.randn(n_row**2, dim_gan, dtype=torch.float).to(device)
    unique_labels = np.sort(unique_labels)
    selected_labels = np.zeros(n_row)
    indx_step_size = len(unique_labels)//n_row
    for i in range(n_row):
        indx = i*indx_step_size
        selected_labels[i] = unique_labels[indx]
    y_fixed = np.zeros(n_row**2)
    for i in range(n_row):
        curr_label = selected_labels[i]
        for j in range(n_row):
            y_fixed[i*n_row+j] = curr_label
    y_fixed = torch.from_numpy(y_fixed).type(torch.long).to(device)


    batch_idx = 0
    dataloader_iter = iter(train_dataloader)

    start_time = timeit.default_timer()
    for niter in range(resume_niters, niters):

        if batch_idx+1 == len(train_dataloader):
            dataloader_iter = iter(train_dataloader)
            batch_idx = 0

        # training images
        batch_train_images, batch_train_labels = dataloader_iter.next()
        assert batch_size == batch_train_images.shape[0]
        batch_train_images = batch_train_images.type(torch.float).to(device)
        batch_train_labels = batch_train_labels.type(torch.long).to(device)


        '''

        Train Generator: maximize log(D(G(z)))

        '''
        netG.train()

        # Sample noise and labels as generator input
        z = torch.randn(batch_size, dim_gan, dtype=torch.float).to(device)

        #generate fake images
        batch_fake_images = netG(z, batch_train_labels)

        # Loss measures generator's ability to fool the discriminator
        dis_out = netD(batch_fake_images, batch_train_labels)

        if loss_type == "vanilla":
            dis_out = torch.nn.Sigmoid()(dis_out)
            g_loss = - torch.mean(torch.log(dis_out+1e-20))
        elif loss_type == "hinge":
            g_loss = - dis_out.mean()

        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()

        '''

        Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))

        '''

        # Measure discriminator's ability to classify real from generated samples
        real_dis_out = netD(batch_train_images, batch_train_labels)
        fake_dis_out = netD(batch_fake_images.detach(), batch_train_labels.detach())
        if loss_type == "vanilla":
            real_dis_out = torch.nn.Sigmoid()(real_dis_out)
            fake_dis_out = torch.nn.Sigmoid()(fake_dis_out)
            d_loss_real = - torch.log(real_dis_out+1e-20)
            d_loss_fake = - torch.log(1-fake_dis_out+1e-20)
        elif loss_type == "hinge":
            d_loss_real = torch.nn.ReLU()(1.0 - real_dis_out)
            d_loss_fake = torch.nn.ReLU()(1.0 + fake_dis_out)
        d_loss = (d_loss_real + d_loss_fake).mean()

        optimizerD.zero_grad()
        d_loss.backward()
        optimizerD.step()

        batch_idx+=1

        if (niter+1)%20 == 0:
            print ("cGAN: [Iter %d/%d] [D loss: %.4f] [G loss: %.4f] [D out real:%.4f] [D out fake:%.4f] [Time: %.4f]" % (niter+1, niters, d_loss.item(), g_loss.item(), real_dis_out.mean().item(),fake_dis_out.mean().item(), timeit.default_timer()-start_time))


        if (niter+1) % 100 == 0:
            netG.eval()
            with torch.no_grad():
                gen_imgs = netG(z_fixed, y_fixed)
                gen_imgs = gen_imgs.detach()
            save_image(gen_imgs.data, save_images_folder +'/{}.png'.format(niter+1), nrow=n_row, normalize=True)

        if save_models_folder is not None and ((niter+1) % save_niters_freq == 0 or (niter+1) == niters):
            save_file = save_models_folder + "/cGAN_checkpoint_intrain/cGAN_checkpoint_niters_{}.pth".format(niter+1)
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


def SampcGAN_given_label(netG, given_label, class_cutoff_points, NFAKE = 10000, batch_size = 500):
    '''
    given_label: a scalar; raw label without any normalization; not class label
    class_cutoff_points: the cutoff points to determine the membership of a give label
    '''
    class_cutoff_points = np.array(class_cutoff_points)
    num_classes = len(class_cutoff_points)-1

    diff_tmp = class_cutoff_points - given_label
    indx_nonneg = np.where(diff_tmp>=0)[0]
    if len(indx_nonneg)==1: #the last element of diff_tmp is non-negative
        given_class_label = num_classes-1
        assert indx_nonneg[0] == num_classes
    elif len(indx_nonneg)>1:
        if diff_tmp[indx_nonneg[0]]>0:
            given_class_label = indx_nonneg[0] - 1
        else:
            given_class_label = indx_nonneg[0]

    if batch_size>NFAKE:
        batch_size = NFAKE
    fake_images = np.zeros((NFAKE+batch_size, NC, IMG_SIZE, IMG_SIZE))
    netG=netG.to(device)
    netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < NFAKE:
            z = torch.randn(batch_size, dim_gan, dtype=torch.float).to(device)
            labels = torch.from_numpy(given_class_label*np.ones(batch_size)).type(torch.long).to(device)
            batch_fake_images = netG(z, labels)
            fake_images[tmp:(tmp+batch_size)] = batch_fake_images.detach().cpu().numpy()
            tmp += batch_size

    #remove extra entries
    fake_images = fake_images[0:NFAKE]
    raw_fake_labels = np.ones(NFAKE) * given_label #use assigned label

    return fake_images, raw_fake_labels
