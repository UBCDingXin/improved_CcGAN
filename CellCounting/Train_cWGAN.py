"""
Train WGAN and WGAN-GP and their conditional versions
WITH their samplers

"""

import torch
from torchvision.utils import save_image
import numpy as np
from torch import autograd
import os


NC=1
IMG_SIZE=64

MIN_COUNT=74
MAX_COUNT=317

############################################################################################
# Train WGANs
############################################################################################
#-------------------------------------------------------------------------------------------
#  Train WGAN-GP

## function for computing gradient penalty
def calc_gradient_penalty_WGAN(netD, real_data, fake_data, batch_train_counts_data, LAMBDA=10, device="cuda"):
    #LAMBDA: Gradient penalty lambda hyperparameter
    # print "real_data: ", real_data.size(), fake_data.size()
    alpha = torch.rand(real_data.size(0), 1)
    alpha = alpha.expand(real_data.size(0), NC*IMG_SIZE*IMG_SIZE)
    alpha = alpha.view(real_data.size(0), NC, IMG_SIZE, IMG_SIZE)
    alpha = alpha.to(device)

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.to(device)

    interpolates = autograd.Variable(interpolates, requires_grad=True)
    batch_train_counts = autograd.Variable(batch_train_counts_data, requires_grad=True)

    disc_interpolates = netD(interpolates, batch_train_counts)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device) ,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

def train_WGANGP(EPOCHS_GAN, GAN_Latent_Length, trainloader, netG, netD, optimizerG, optimizerD, save_GANimages_folder, LAMBDA = 10, CRITIC_ITERS=5, save_models_folder = None, ResumeEpoch = 0, device="cuda", mean_count=0, std_count=1):

    netG = netG.to(device)
    netD = netD.to(device)

    if save_models_folder is not None and ResumeEpoch>0:
        save_file = save_models_folder + "/cWGANGP_checkpoint_intrain/cWGANGP_checkpoint_epoch" + str(ResumeEpoch) + ".pth"
        checkpoint = torch.load(save_file)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
        gen_iterations = checkpoint['gen_iterations']
    else:
        gen_iterations = 0
    #end if

    n_row=4
    z_fixed = torch.randn(n_row**2, GAN_Latent_Length, dtype=torch.float).to(device)
    y_fixed = np.random.randint(MIN_COUNT, MAX_COUNT, n_row**2)
    y_fixed = torch.from_numpy(y_fixed).type(torch.float).view(-1,1).to(device)
    y_fixed = (y_fixed-mean_count)/std_count

    for epoch in range(ResumeEpoch, EPOCHS_GAN):

        data_iter = iter(trainloader)
        batch_idx = 0
        while (batch_idx < len(trainloader)):
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update

            for batch_idx_G in range(CRITIC_ITERS):

                if batch_idx == len(trainloader):
                    break

                (batch_train_images,batch_train_counts) = data_iter.next()
                batch_idx += 1

                BATCH_SIZE = batch_train_images.shape[0]
                batch_train_images = batch_train_images.type(torch.float).to(device)
                batch_train_counts = batch_train_counts.type(torch.float).to(device)

                netD.zero_grad()
                # Train D on real
                D_real = netD(batch_train_images, batch_train_counts)
                D_real = D_real.mean()
                # Generate fake images
                z = torch.randn(BATCH_SIZE, GAN_Latent_Length, dtype=torch.float).to(device)
                gen_imgs = netG(z, batch_train_counts)
                # Train D on fake
                D_fake = netD(gen_imgs, batch_train_counts)
                D_fake = D_fake.mean()
                # train with gradient penalty
                gradient_penalty = calc_gradient_penalty_WGAN(netD, batch_train_images.data, gen_imgs.data, batch_train_counts.data, LAMBDA, device=device)

                D_cost = D_fake - D_real + gradient_penalty
                D_cost.backward()
                optimizerD.step()

                Wasserstein_D = D_real.cpu().item() - D_fake.cpu().item()

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()
            # Generate fake images
            z = torch.randn(BATCH_SIZE, GAN_Latent_Length, dtype=torch.float).to(device)
            gen_imgs = netG(z)
            G_cost = - netD(gen_imgs).mean()
            G_cost.backward()
            optimizerG.step()
            gen_iterations += 1

            print("WGAN: [Step %d/%d] [Epoch %d/%d] [G_iter %d] [D loss: %.4f] [G loss: %.4f][W Dist: %.4f]" % (batch_idx, len(trainloader), epoch +1, EPOCHS_GAN, gen_iterations, D_cost.item(), G_cost.item(), Wasserstein_D))

#            if batch_idx%20 == 0:
#                print("WGAN: [Step %d/%d] [Epoch %d/%d] [G_iter %d] [D loss: %.4f] [G loss: %.4f][W Dist: %.4f]" % (batch_idx+1, len(trainloader), epoch +1, EPOCHS_GAN, gen_iterations, D_cost.item(), G_cost.item(), Wasserstein_D))

            if gen_iterations % 100 == 0:
                with torch.no_grad():
                    gen_imgs = netG(z_fixed)
                    gen_imgs = gen_imgs.detach()
                save_image(gen_imgs.data, save_GANimages_folder +'%d.png' % gen_iterations, nrow=n_row, normalize=True)
        #end for batch_idx

        if save_models_folder is not None and (epoch+1) % 50 == 0:
            save_file = save_models_folder + "/cWGANGP_checkpoint_intrain/"
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_file = save_file + "cWGANGP_checkpoint_epoch" + str(epoch+1) + ".pth"
            torch.save({
                    'gen_iterations': gen_iterations,
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)

    #end for epoch

    return netG, netD, optimizerG, optimizerD

#--------------------------------------------------------------------------------
# Sample WGAN and WGAN-GP
def SampcWGAN(netG, GAN_Latent_Length = 128, NFAKE = 10000, batch_size = 500, device="cuda", mean_count=0, std_count=1):
    raw_fake_images = np.zeros((NFAKE+batch_size, NC, IMG_SIZE, IMG_SIZE))
    raw_fake_counts = np.zeros(NFAKE+batch_size, dtype=np.float)
    netG=netG.to(device)
    netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < NFAKE:
            z = torch.randn(batch_size, GAN_Latent_Length, dtype=torch.float).to(device)
            y = np.random.randint(MIN_COUNT, MAX_COUNT, n_row**2)
            y = torch.from_numpy(y).type(torch.float).view(-1,1).to(device)
            y = (y - mean_count)/std_count
            batch_fake_images = netG(z, y)
            raw_fake_images[tmp:(tmp+batch_size)] = batch_fake_images.cpu().detach().numpy()
            raw_fake_counts[tmp:(tmp+batch_size)] = y.cpu().view(-1).detach().numpy()
            tmp += batch_size

    #remove unused entry and extra samples
    raw_fake_images = raw_fake_images[0:NFAKE]
    raw_fake_counts = raw_fake_counts[0:NFAKE]

    return raw_fake_images, raw_fake_counts.reshape(-1)
