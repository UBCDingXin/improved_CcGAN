"""
Train a regression DCGAN

"""

import torch
import numpy as np
import os
import timeit

from utils import *




############################################################################################
# Train Continuous cGAN
def train_CcGAN(kernel_sigma, threshold_type, kappa, epoch_GAN, dim_GAN, trainloader, netG, netD, optimizerG, optimizerD, save_images_folder, save_models_folder = None, ResumeEpoch = 0, device="cuda", plot_in_train=False, samples_tar_eval = None, angle_grid_eval = None, fig_size=5, point_size=None):
    '''

    kernel_sigma: the sigma in the Guassian kernel (a real value)
    threshold_type: hard or soft threshold ('hard', 'soft')

    '''

    def sample_Gaussian(n, dim, mean=0, sigma=1):
        samples = np.random.normal(mean, sigma, n*dim)
        return samples.reshape((n, dim))


    # traning GAN model
    netG = netG.to(device)
    netD = netD.to(device)

    if save_models_folder is not None and ResumeEpoch>0:
        save_file = save_models_folder + "/CcGAN_checkpoint_intrain/CcGAN_checkpoint_epoch" + str(ResumeEpoch) + ".pth"
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


    start_tmp = timeit.default_timer()
    for epoch in range(ResumeEpoch, epoch_GAN):
        for batch_idx, (batch_train_samples, batch_train_angles) in enumerate(trainloader):

            # if batch_idx == len(trainloader)-1:
            #     break

            # samples and angles are split into two parts evenly.
            # only 50% of the batch samples are used but all angles are used
            BATCH_SIZE = batch_train_samples.shape[0]//2
            batch_train_samples_1 = batch_train_samples[0:BATCH_SIZE].type(torch.float).to(device) #real x_j's
            batch_train_angles_1 = batch_train_angles[0:BATCH_SIZE].type(torch.float).to(device) #y_j's
            batch_train_angles_2 = batch_train_angles[BATCH_SIZE:].type(torch.float).to(device) #y_i's

            # generate Gaussian noise which are added to y_i (batch_train_angles_2)
            batch_epsilons_1 = np.random.normal(0, kernel_sigma, BATCH_SIZE)
            batch_epsilons_tensor_1 = torch.from_numpy(batch_epsilons_1).type(torch.float).to(device)

            batch_epsilons_2 = np.random.normal(0, kernel_sigma, BATCH_SIZE)
            batch_epsilons_tensor_2 = torch.from_numpy(batch_epsilons_2).type(torch.float).to(device)

            '''

            Train Generator: maximize log(D(G(z)))

            '''
            netG.train()
            optimizerG.zero_grad()

            # sample noise as generator's input; generate fake samples with length BATCH_SIZE
            z_2 = torch.from_numpy(sample_Gaussian(BATCH_SIZE, dim_GAN)).type(torch.float).to(device)
            # batch_fake_samples_2 = netG(z_2, batch_train_angles_2 + batch_epsilons_tensor_2)
            # batch_comb_labels_2 = torch.clamp(batch_train_angles_2 + batch_epsilons_tensor_2, 0, 1)
            batch_comb_labels_2 = batch_train_angles_2 + batch_epsilons_tensor_2
            indx_smaller = batch_comb_labels_2<0
            indx_bigger = batch_comb_labels_2>1
            batch_comb_labels_2[indx_smaller] = batch_comb_labels_2[indx_smaller] + 1
            batch_comb_labels_2[indx_bigger] = batch_comb_labels_2[indx_bigger] - 1

            batch_fake_samples_2 = netG(z_2, batch_comb_labels_2)

            # Loss measures generator's ability to fool the discriminator
            # dis_out = netD(batch_fake_samples_2, batch_train_angles_2 + batch_epsilons_tensor_2)
            dis_out = netD(batch_fake_samples_2, batch_comb_labels_2)

            # no weights
            g_loss = - torch.mean(torch.log(dis_out+1e-20))

            g_loss.backward()
            optimizerG.step()


            '''

            Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))

            '''
            #train discriminator once and generator several times
            optimizerD.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            # real_dis_out = netD(batch_train_samples_1, batch_train_angles_2 + batch_epsilons_tensor_2)
            real_dis_out = netD(batch_train_samples_1, batch_comb_labels_2)

            # z_1 = torch.randn(BATCH_SIZE, dim_GAN, dtype=torch.float).to(device)
            z_1 = torch.from_numpy(sample_Gaussian(BATCH_SIZE, dim_GAN)).type(torch.float).to(device)
            # batch_fake_samples_1 = netG(z_1, batch_train_angles_1 + batch_epsilons_tensor_1)
            # fake_dis_out = netD(batch_fake_samples_1.detach(), batch_train_angles_2 + batch_epsilons_tensor_2)

            # batch_comb_labels_1 = torch.clamp(batch_train_angles_1 + batch_epsilons_tensor_1, 0, 1)
            batch_comb_labels_1 = batch_train_angles_1 + batch_epsilons_tensor_1
            indx_smaller = batch_comb_labels_1<0
            indx_bigger = batch_comb_labels_1>1
            batch_comb_labels_1[indx_smaller] = batch_comb_labels_1[indx_smaller] + 1
            batch_comb_labels_1[indx_bigger] = batch_comb_labels_1[indx_bigger] - 1


            batch_fake_samples_1 = netG(z_1, batch_comb_labels_1)
            fake_dis_out = netD(batch_fake_samples_1.detach(), batch_comb_labels_2)


            # compute weight for x_j when it is used to learn p(x|y_i+epsilon)
            if threshold_type == "soft":
                # real_weights_x_j = np.clip(np.exp(-kappa*(batch_train_angles_1.cpu().numpy()-batch_train_angles_2.cpu().numpy()-batch_epsilons_2)**2), 1e-20, 1e+20)
                # fake_weights_x_j = np.clip(np.exp(-kappa*(batch_train_angles_1.cpu().numpy()+batch_epsilons_1-batch_train_angles_2.cpu().numpy()-batch_epsilons_2)**2), 1e-20, 1e+20)

                # real_weights_x_j = np.clip(np.exp(-kappa*(batch_train_angles_1.cpu().numpy()-batch_comb_labels_2.cpu().numpy())**2), 0, 1e+20)
                # fake_weights_x_j = np.clip(np.exp(-kappa*(batch_comb_labels_1.cpu().numpy()-batch_comb_labels_2.cpu().numpy())**2), 0, 1e+20)

                real_weights_x_j = np.exp(-kappa*(batch_train_angles_1.cpu().numpy()-batch_comb_labels_2.cpu().numpy())**2)
                fake_weights_x_j = np.exp(-kappa*(batch_comb_labels_1.cpu().numpy()-batch_comb_labels_2.cpu().numpy())**2)
            else:
                # real_weights_x_j = np.zeros(BATCH_SIZE)
                # indx = np.where(np.abs(batch_train_angles_1.cpu().numpy()-batch_train_angles_2.cpu().numpy()-batch_epsilons_2) <= kappa)[0]
                # real_weights_x_j[indx] = 1
                #
                # fake_weights_x_j = np.zeros(BATCH_SIZE)
                # indx = np.where(np.abs(batch_train_angles_1.cpu().numpy()+batch_epsilons_1-batch_train_angles_2.cpu().numpy()-batch_epsilons_2) <= kappa)[0]
                # fake_weights_x_j[indx] = 1

                real_weights_x_j = np.zeros(BATCH_SIZE)
                indx = np.where(np.abs(batch_train_angles_1.cpu().numpy()-batch_comb_labels_2.cpu().numpy()) <= kappa)[0]
                real_weights_x_j[indx] = 1

                fake_weights_x_j = np.zeros(BATCH_SIZE)
                indx = np.where(np.abs(batch_comb_labels_1.cpu().numpy()-batch_comb_labels_2.cpu().numpy()) <= kappa)[0]
                fake_weights_x_j[indx] = 1

            real_weights_x_j = torch.from_numpy(real_weights_x_j).type(torch.float).to(device)
            fake_weights_x_j = torch.from_numpy(fake_weights_x_j).type(torch.float).to(device)

            d_loss = - torch.mean(real_weights_x_j * torch.log(real_dis_out+1e-20)) - torch.mean(fake_weights_x_j * torch.log(1 - fake_dis_out+1e-20))

            d_loss.backward()
            optimizerD.step()

            gen_iterations += 1

            if batch_idx%20 == 0:
                print ("CcGAN: [Iter %d] [Epoch %d/%d] [D loss: %.4e] [G loss: %.4e] [real prob: %.3f] [fake prob: %.3f] [Time: %.4f]" % (gen_iterations, epoch + 1, epoch_GAN, d_loss.item(), g_loss.item(), real_dis_out.mean().item(), fake_dis_out.mean().item(), timeit.default_timer()-start_tmp))
        # for batch_idx


        if plot_in_train and (epoch+1) % 50 == 0:
            netG.eval()
            assert save_images_folder is not None
            z = torch.randn(samples_tar_eval.shape[0], dim_GAN, dtype=torch.float).to(device)

            labels = np.random.choice(angle_grid_eval/(2*np.pi), size=samples_tar_eval.shape[0], replace=True)
            labels = torch.from_numpy(labels).type(torch.float).to(device)
            prop_samples = netG(z, labels).cpu().detach().numpy()
            filename = save_images_folder + '/' + str(epoch+1) + '.png'
            ScatterPoints(samples_tar_eval, prop_samples, filename, fig_size=fig_size, point_size=point_size)

            labels = np.random.choice(angle_grid_eval/(2*np.pi), size=1, replace=True)
            labels = np.repeat(labels, samples_tar_eval.shape[0])
            labels = torch.from_numpy(labels).type(torch.float).to(device)
            prop_samples = netG(z, labels).cpu().detach().numpy()
            filename = save_images_folder + '/{}_{}.png'.format(epoch+1, labels[0]*(2*np.pi))
            ScatterPoints(samples_tar_eval, prop_samples, filename, fig_size=fig_size, point_size=point_size)


        if save_models_folder is not None and (epoch+1) % 1000 == 0:
            save_file = save_models_folder + "/CcGAN_checkpoint_intrain/"
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_file = save_file + "CcGAN_checkpoint_epoch" + str(epoch+1) + ".pth"
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


def SampCcGAN_given_label(netG, label, path=None, dim_GAN = 2, NFAKE = 10000, batch_size = 500, num_features=2, device="cuda"):
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
            z = torch.randn(batch_size, dim_GAN, dtype=torch.float).to(device)
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
