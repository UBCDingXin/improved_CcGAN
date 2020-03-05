"""
Train a regression DCGAN

"""

import torch
from torchvision.utils import save_image
import numpy as np
import os

NC=1
IMG_SIZE=64

MIN_LABEL=74
MAX_LABEL=317


############################################################################################
# Train DCGAN

def train_cDCGAN(EPOCHS_GAN, GAN_Latent_Length, trainloader, netG, netD, optimizerG, optimizerD, criterion, save_cDCGANimages_folder, save_models_folder = None, ResumeEpoch = 0, device="cuda", shift_label = 0, max_label = 1):


    netG = netG.to(device)
    netD = netD.to(device)

    if save_models_folder is not None and ResumeEpoch>0:
        save_file = save_models_folder + "/cDCGAN_checkpoint_intrain/cDCGAN_checkpoint_epoch" + str(ResumeEpoch) + ".pth"
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

    n_row=8
    z_fixed = torch.randn(n_row**2, GAN_Latent_Length, dtype=torch.float).to(device)
    y_fixed = np.random.randint(MIN_LABEL, MAX_LABEL, n_row**2).astype(np.float)
    y_fixed += shift_label
    y_fixed /= max_label
    y_fixed = torch.from_numpy(y_fixed).type(torch.float).view(-1,1).to(device)

    for epoch in range(ResumeEpoch, EPOCHS_GAN):
        for batch_idx, (batch_train_images, batch_train_counts) in enumerate(trainloader):

            BATCH_SIZE = batch_train_images.shape[0]
            batch_train_images = batch_train_images.type(torch.float).to(device)
            batch_train_counts = batch_train_counts.type(torch.float).to(device)

            # Adversarial ground truths
            GAN_real = torch.ones(BATCH_SIZE,1).to(device)
            GAN_fake = torch.zeros(BATCH_SIZE,1).to(device)

            '''

            Train Generator: maximize log(D(G(z)))

            '''
            optimizerG.zero_grad()
            # Sample noise and labels as generator input
            z = torch.randn(BATCH_SIZE, GAN_Latent_Length, dtype=torch.float).to(device)

            #generate images
            batch_fake_images = netG(z, batch_train_counts)

            # Loss measures generator's ability to fool the discriminator
            dis_out = netD(batch_fake_images, batch_train_counts)

            #generator try to let disc believe gen_imgs are real
            g_loss = criterion(dis_out, GAN_real)
            #final g_loss consists of two parts one from generator's and the other one is from validity loss

            g_loss.backward()
            optimizerG.step()

            '''

            Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))

            '''
            #train discriminator once and generator several times
            optimizerD.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            prob_real = netD(batch_train_images, batch_train_counts)
            prob_fake = netD(batch_fake_images.detach(), batch_train_counts.detach())
            real_loss = criterion(prob_real, GAN_real)
            fake_loss = criterion(prob_fake, GAN_fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizerD.step()
            gen_iterations += 1

            if batch_idx%20 == 0:
                print ("cDCGAN: [Iter %d] [Epoch %d/%d] [D loss: %.4f] [G loss: %.4f] [D prob real:%.4f] [D prob fake:%.4f]" % (gen_iterations, epoch + 1, EPOCHS_GAN, d_loss.item(), g_loss.item(), prob_real.mean().item(),prob_fake.mean().item()))

            if gen_iterations % 100 == 0:
                with torch.no_grad():
                    gen_imgs = netG(z_fixed, y_fixed)
                    gen_imgs = gen_imgs.detach()
                save_image(gen_imgs.data, save_cDCGANimages_folder +'%d.png' % gen_iterations, nrow=n_row, normalize=True)

        if save_models_folder is not None and (epoch+1) % 100 == 0:
            save_file = save_models_folder + "/cDCGAN_checkpoint_intrain/"
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_file = save_file + "cDCGAN_checkpoint_epoch" + str(epoch+1) + ".pth"
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


def SampcDCGAN(netG, GAN_Latent_Length = 128, NFAKE = 10000, batch_size = 500, device="cuda", shift_label = 0, max_label = 1):
    #netD: whether assign weights to fake images via inversing f function (the f in f-GAN)
    if batch_size>NFAKE:
        batch_size = NFAKE
    raw_fake_images = np.zeros((NFAKE+batch_size, NC, IMG_SIZE, IMG_SIZE), dtype=np.float)
    raw_fake_counts = np.zeros(NFAKE+batch_size, dtype=np.float)
    netG=netG.to(device)
    netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < NFAKE:
            z = torch.randn(batch_size, GAN_Latent_Length, dtype=torch.float).to(device)
            y = np.random.randint(MIN_LABEL, MAX_LABEL, n_row**2)
            y = torch.from_numpy(y).type(torch.float).view(-1,1).to(device)
            y += shift_label
            y /= max_label
            batch_fake_images = netG(z, y)
            raw_fake_images[tmp:(tmp+batch_size)] = batch_fake_images.cpu().detach().numpy()
            raw_fake_counts[tmp:(tmp+batch_size)] = y.cpu().view(-1).detach().numpy()
            tmp += batch_size

    #remove extra entries
    raw_fake_images = raw_fake_images[0:NFAKE]
    raw_fake_counts = raw_fake_counts[0:NFAKE]

    return raw_fake_images, raw_fake_counts.reshape(-1)
