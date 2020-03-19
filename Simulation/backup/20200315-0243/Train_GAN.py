"""
Train DCGAN and condtional DCGAN
WITH their samplers

"""

import torch
import numpy as np
import os
from utils import *




############################################################################################
# Train DCGAN

def train_GAN(EPOCHS_GAN, GAN_Latent_Length, trainloader, netG, netD, optimizerG, optimizerD, criterion, save_models_folder = None, ResumeEpoch = 0, device="cuda", plot_in_train=False, save_images_folder = None, samples_tar_eval = None, num_features = 2):


    netG = netG.to(device)
    netD = netD.to(device)

    if save_models_folder is not None and ResumeEpoch>0:
        save_file = save_models_folder + "/GAN_checkpoint_intrain/GAN_checkpoint_epoch" + str(ResumeEpoch) + ".pth"
        checkpoint = torch.load(save_file)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        gen_iterations = checkpoint['gen_iterations']
    else:
        gen_iterations = 0
    #end if

    for epoch in range(ResumeEpoch, EPOCHS_GAN):
        for batch_idx, (batch_train_samples,_) in enumerate(trainloader):

            BATCH_SIZE = batch_train_samples.shape[0]
            batch_train_samples = batch_train_samples.type(torch.float).to(device)

            # Adversarial ground truths
            GAN_real = torch.ones(BATCH_SIZE,1).to(device)
            GAN_fake = torch.zeros(BATCH_SIZE,1).to(device)

            '''

            Train Generator: maximize log(D(G(z)))

            '''
            netG.train()
            optimizerG.zero_grad()
            # Sample noise and labels as generator input
            z = torch.randn(BATCH_SIZE, GAN_Latent_Length, dtype=torch.float).to(device)

            #generate samples
            gen_samples = netG(z)

            # Loss measures generator's ability to fool the discriminator
            dis_out = netD(gen_samples)

            #generator try to let disc believe gen_samples are real
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
            prob_real = netD(batch_train_samples)
            prob_fake = netD(gen_samples.detach())
            real_loss = criterion(prob_real, GAN_real)
            fake_loss = criterion(prob_fake, GAN_fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizerD.step()
            gen_iterations += 1

            if batch_idx%20 == 0:
                print ("GAN: [Iter %d] [Epoch %d/%d] [D loss: %.4f] [G loss: %.4f] [D prob real:%.4f] [D prob fake:%.4f]" % (gen_iterations, epoch + 1, EPOCHS_GAN, d_loss.item(), g_loss.item(), prob_real.mean().item(),prob_fake.mean().item()))
        #end for batch_idx

        if plot_in_train and (epoch+1)%50==0:
            netG.eval()
            assert save_images_folder is not None
            z = torch.randn(samples_tar_eval.shape[0], GAN_Latent_Length, dtype=torch.float).to(device)
            prop_samples = netG(z).cpu().detach().numpy()
            filename = save_images_folder + '/' + str(epoch+1) + '.png'
            ScatterPoints(samples_tar_eval, prop_samples, filename)



        if save_models_folder is not None and (epoch+1) % 100 == 0:
            save_file = save_models_folder + "/GAN_checkpoint_intrain/"
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_file = save_file + "GAN_checkpoint_epoch" + str(epoch+1) + ".pth"
            torch.save({
                    'gen_iterations': gen_iterations,
                    'netG_state_dict': netG.state_dict(),
                    'netD_state_dict': netD.state_dict(),
                    'optimizerG_state_dict': optimizerG.state_dict(),
                    'optimizerD_state_dict': optimizerD.state_dict()
            }, save_file)
    #end for epoch

    return netG, netD, optimizerG, optimizerD


def SampGAN(netG, GAN_Latent_Length = 2, NFAKE = 10000, batch_size = 500, device="cuda", num_features = 2):
    #netD: whether assign weights to fake samples via inversing f function (the f in f-GAN)
    raw_fake_samples = np.zeros((NFAKE+batch_size, num_features))
    netG=netG.to(device)
    netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < NFAKE:
            z = torch.randn(batch_size, GAN_Latent_Length, dtype=torch.float).to(device)
            batch_fake_samples = netG(z)
            raw_fake_samples[tmp:(tmp+batch_size)] = batch_fake_samples.cpu().numpy()
            tmp += batch_size

    #remove extra entries
    raw_fake_samples = raw_fake_samples[0:NFAKE]

    return raw_fake_samples
