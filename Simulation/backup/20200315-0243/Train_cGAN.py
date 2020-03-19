

import torch
import numpy as np
import os
from utils import *

def train_cGAN(EPOCHS_GAN, GAN_Latent_Length, trainloader, netG, netD, optimizerG, optimizerD, criterion, save_models_folder = None, ResumeEpoch = 0, device="cuda", plot_in_train=False, save_images_folder = None, samples_tar_eval = None, angle_grid_eval = None, num_classes=None, num_features = 2, unique_labels = None, label2class = None):
    netG = netG.to(device)
    netD = netD.to(device)

    if save_models_folder is not None and ResumeEpoch>0:
        save_file = save_models_folder + "/cGAN_checkpoint_intrain/cGAN_checkpoint_epoch" + str(ResumeEpoch) + ".pth"
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
        for batch_idx, (batch_train_samples, batch_train_labels) in enumerate(trainloader):

            BATCH_SIZE = batch_train_samples.shape[0]
            batch_train_samples = batch_train_samples.type(torch.float).to(device)
            batch_train_labels = batch_train_labels.type(torch.long).to(device)

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
            batch_fake_samples = netG(z, batch_train_labels)

            # Loss measures generator's ability to fool the discriminator
            dis_out = netD(batch_fake_samples, batch_train_labels)

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
            prob_real = netD(batch_train_samples, batch_train_labels)
            prob_fake = netD(batch_fake_samples.detach(), batch_train_labels.detach())
            real_loss = criterion(prob_real, GAN_real)
            fake_loss = criterion(prob_fake, GAN_fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizerD.step()
            gen_iterations += 1

            if batch_idx%20 == 0:
                print ("cGAN: [Iter %d] [Epoch %d/%d] [D loss: %.4f] [G loss: %.4f] [D prob real:%.4f] [D prob fake:%.4f]" % (gen_iterations, epoch + 1, EPOCHS_GAN, d_loss.item(), g_loss.item(), prob_real.mean().item(),prob_fake.mean().item()))

        #end for batch_idx

        if plot_in_train and (epoch+1)%50==0:
            # netG.eval()
            # assert save_images_folder is not None
            # z = torch.randn(samples_tar_eval.shape[0], GAN_Latent_Length, dtype=torch.float).to(device)
            # labels = np.random.choice(np.arange(num_classes), size=samples_tar_eval.shape[0], replace=True)
            # labels = torch.from_numpy(labels).type(torch.long).to(device)
            # prop_samples = netG(z, labels).cpu().detach().numpy()
            # filename = save_images_folder + '/' + str(epoch+1) + '.png'
            # ScatterPoints(samples_tar_eval, prop_samples, filename)

            assert save_images_folder is not None
            n_samp_per_gaussian_eval = samples_tar_eval.shape[0]//len(angle_grid_eval)

            for j_tmp in range(len(angle_grid_eval)):
                angle = angle_grid_eval[j_tmp]
                angle = (angle*2*np.pi).astype(np.single) #back to original scale of angle [0, 2*pi]
                prop_samples_curr, _ = SampcGAN_given_label(netG, angle, unique_labels, label2class, GAN_Latent_Length, n_samp_per_gaussian_eval, batch_size = 100, device=device)
                if j_tmp == 0:
                    prop_samples = prop_samples_curr
                else:
                    prop_samples = np.concatenate((prop_samples, prop_samples_curr), axis=0)

            filename = save_images_folder + '/' + str(epoch+1) + '.png'
            ScatterPoints(samples_tar_eval, prop_samples, filename)



        if save_models_folder is not None and (epoch+1) % 1000 == 0:
            save_file = save_models_folder + "/cGAN_checkpoint_intrain/"
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_file = save_file + "cGAN_checkpoint_epoch" + str(epoch+1) + ".pth"
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



def SampcGAN(netG, class2label, GAN_Latent_Length = 2, NFAKE = 10000, batch_size = 500, num_classes = 100, num_features = 2, device="cuda"):
    '''
    class2label: convert class label to raw label without normalization
    '''
    if batch_size>NFAKE:
        batch_size = NFAKE
    fake_samples = np.zeros((NFAKE+batch_size, num_features))
    raw_fake_labels = np.zeros(NFAKE+batch_size)
    netG=netG.to(device)
    netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < NFAKE:
            z = torch.randn(batch_size, GAN_Latent_Length, dtype=torch.float).to(device)
            labels = np.random.choice(np.arange(num_classes),size=batch_size,replace=True)
            raw_fake_labels[tmp:(tmp+batch_size)] = labels
            labels = torch.from_numpy(labels).type(torch.long).to(device)
            batch_fake_samples = netG(z, labels)
            fake_samples[tmp:(tmp+batch_size)] = batch_fake_samples.cpu().detach().numpy()
            tmp += batch_size

    #remove extra entries
    fake_samples = fake_samples[0:NFAKE]
    raw_fake_labels = raw_fake_labels[0:NFAKE]
    raw_fake_labels = raw_fake_labels.astype(np.float)

    #convert class labels to raw labels
    raw_fake_labels = np.array([class2label[raw_fake_labels[i]] for i in range(NFAKE)])

    return fake_samples, raw_fake_labels


def SampcGAN_given_label(netG, given_label, unique_labels, label2class, GAN_Latent_Length = 2, NFAKE = 10000, batch_size = 500, num_features=2, device="cuda"):
    '''
    given_label: raw label without any normalization; not class label
    unique_labels: unique raw labels
    label2class: convert a raw label to a class label
    '''

    #given label may not in unique_labels, so find the closest unique label
    dis_all = np.array([(given_label-unique_labels[i])**2 for i in range(len(unique_labels))])
    dis_sorted = np.sort(dis_all)
    dis_argsorted = np.argsort(dis_all)

    closest_unique_label1 = unique_labels[dis_argsorted[0]]
    if dis_sorted[0]==dis_sorted[1]: #two closest unique labels
        closest_unique_label2 = unique_labels[dis_argsorted[1]]
    else:
        closest_unique_label2 = None

    #netD: whether assign weights to fake samples via inversing f function (the f in f-GAN)
    if batch_size>NFAKE:
        batch_size = NFAKE
    fake_samples = np.zeros((NFAKE+batch_size, num_features))
    netG=netG.to(device)
    netG.eval()
    with torch.no_grad():
        tmp = 0
        if closest_unique_label2 is None: #only one closest unique label
            while tmp < NFAKE:
                z = torch.randn(batch_size, GAN_Latent_Length, dtype=torch.float).to(device)
                labels = torch.from_numpy(label2class[closest_unique_label1]*np.ones(batch_size)).type(torch.long).to(device)
                batch_fake_samples = netG(z, labels)
                fake_samples[tmp:(tmp+batch_size)] = batch_fake_samples.detach().cpu().numpy()
                tmp += batch_size
        else: #two closest unique labels
            while tmp < NFAKE//2:
                z = torch.randn(batch_size, GAN_Latent_Length, dtype=torch.float).to(device)
                labels = torch.from_numpy(label2class[closest_unique_label1]*np.ones(batch_size)).type(torch.long).to(device)
                batch_fake_samples = netG(z, labels)
                fake_samples[tmp:(tmp+batch_size)] = batch_fake_samples.detach().cpu().numpy()
                tmp += batch_size

            while tmp < NFAKE-NFAKE//2:
                z = torch.randn(batch_size, GAN_Latent_Length, dtype=torch.float).to(device)
                labels = torch.from_numpy(label2class[closest_unique_label2]*np.ones(batch_size)).type(torch.long).to(device)
                batch_fake_samples = netG(z, labels)
                fake_samples[tmp:(tmp+batch_size)] = batch_fake_samples.detach().cpu().numpy()
                tmp += batch_size

    #remove extra entries
    fake_samples = fake_samples[0:NFAKE]
    raw_fake_labels = np.ones(NFAKE) * given_label #use assigned label

    return fake_samples, raw_fake_labels
