"""
Train a regression DCGAN

"""

import torch
from torchvision.utils import save_image
import numpy as np
import os
import timeit
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

NC=1
IMG_SIZE=64


############################################################################################
# Train Continuous cDCGAN
def train_Continuous_cDCGAN(train_labels, kernel_sigma, threshold_type, kappa, epoch_GAN, dim_GAN, trainloader, netG, netD, optimizerG, optimizerD, save_images_folder, save_models_folder = None, ResumeEpoch = 0, device="cuda", tfboard_writer=None):
    '''

    train_labels: training labels (numpy array)
    kernel_sigma: the sigma in the Guassian kernel (a real value)
    threshold_type: hard or soft threshold ('hard', 'soft')

    '''

    # kernels stuffs: assume Gaussian kernel
    train_labels = train_labels.reshape(-1) #do not shuffle train_labels because its order matters
    n_train = len(train_labels)

    # std_count = np.std(train_labels)
    def sample_Gaussian(n, dim, mean=0, sigma=1):
        samples = np.random.normal(mean, sigma, n*dim)
        return samples.reshape((n, dim))


    # traning GAN model
    netG = netG.to(device)
    netD = netD.to(device)

    if save_models_folder is not None and ResumeEpoch>0:
        save_file = save_models_folder + "/CcDCGAN_checkpoint_intrain/CcDCGAN_checkpoint_epoch" + str(ResumeEpoch) + ".pth"
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

    n_row=10; n_col = n_row
    # z_fixed = torch.randn(n_row*n_col, dim_GAN, dtype=torch.float).to(device)
    z_fixed = torch.from_numpy(sample_Gaussian(n_row*n_col, dim_GAN)).type(torch.float).to(device)

    # min_label = np.min(train_labels)
    # max_label = np.max(train_labels)
    # selected_labels = np.linspace(min_label, max_label, num=n_row+2)
    # selected_labels = selected_labels[1:(n_row+1)] #remove the minimum and maximum

    # printed images with labels between the 5-th quantile and 95-th quantile of training labels
    start_label = np.quantile(train_labels, 0.05)
    end_label = np.quantile(train_labels, 0.95)
    selected_labels = np.linspace(start_label, end_label, num=n_row)
    y_fixed = np.zeros(n_row*n_col)
    for i in range(n_row):
        curr_label = selected_labels[i]
        for j in range(n_col):
            y_fixed[i*n_col+j] = curr_label
    print(y_fixed)
    y_fixed = torch.from_numpy(y_fixed).type(torch.float).view(-1,1).to(device)


    start_tmp = timeit.default_timer()
    for epoch in range(ResumeEpoch, epoch_GAN):
        for batch_idx, (batch_train_images, batch_train_labels) in enumerate(trainloader):

            # images and labels are split into two parts evenly.
            # only 50% of the batch images are used but all labels are used
            BATCH_SIZE = int(batch_train_images.shape[0]/2)
            batch_train_images_1 = batch_train_images[0:BATCH_SIZE].type(torch.float).to(device) #real x_j's
            batch_train_labels_1 = batch_train_labels[0:BATCH_SIZE].type(torch.float).to(device) #y_j's
            batch_train_labels_2 = batch_train_labels[BATCH_SIZE:].type(torch.float).to(device) #y_i's

            # generate Gaussian noise which are added to y_i (batch_train_labels_2)
            batch_epsilons_1 = np.random.normal(0, kernel_sigma, BATCH_SIZE)
            batch_epsilons_tensor_1 = torch.from_numpy(batch_epsilons_1).type(torch.float).to(device)

            batch_epsilons_2 = np.random.normal(0, kernel_sigma, BATCH_SIZE)
            batch_epsilons_tensor_2 = torch.from_numpy(batch_epsilons_2).type(torch.float).to(device)

            '''

            Train Generator: maximize log(D(G(z)))

            '''
            netG.train()
            optimizerG.zero_grad()

            # sample noise as generator's input; generate fake images with length BATCH_SIZE
            # z_2 = torch.randn(BATCH_SIZE, dim_GAN, dtype=torch.float).to(device)
            # batch_fake_images_2 = netG(z_2, batch_train_labels_2 + batch_epsilons_tensor_2 + batch_epsilons_tensor_1)

            z_2 = torch.from_numpy(sample_Gaussian(BATCH_SIZE, dim_GAN)).type(torch.float).to(device)
            batch_fake_images_2 = netG(z_2, batch_train_labels_2 + batch_epsilons_tensor_2)



            # Loss measures generator's ability to fool the discriminator
            dis_out = netD(batch_fake_images_2, batch_train_labels_2 + batch_epsilons_tensor_2)
            # dis_out = netD(batch_fake_images_2, batch_train_labels_2 + batch_epsilons_tensor_2 + batch_epsilons_tensor_1)

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
            real_dis_out = netD(batch_train_images_1, batch_train_labels_2 + batch_epsilons_tensor_2)

            # z_1 = torch.randn(BATCH_SIZE, dim_GAN, dtype=torch.float).to(device)
            z_1 = torch.from_numpy(sample_Gaussian(BATCH_SIZE, dim_GAN)).type(torch.float).to(device)
            batch_fake_images_1 = netG(z_1, batch_train_labels_1 + batch_epsilons_tensor_1)
            fake_dis_out = netD(batch_fake_images_1.detach(),batch_train_labels_2 + batch_epsilons_tensor_2)

            # compute weight for x_j when it is used to learn p(x|y_i+epsilon)
            if threshold_type == "soft":
                # real_weights_x_j = np.clip(np.exp(-kernel_sigma*(batch_train_labels_1.cpu().numpy()-batch_train_labels_2.cpu().numpy()-batch_epsilons_2)**2), 1e-20, 1e+20)
                # fake_weights_x_j = np.clip(np.exp(-kernel_sigma*(batch_train_labels_1.cpu().numpy()+batch_epsilons_1-batch_train_labels_2.cpu().numpy()-batch_epsilons_2)**2), 1e-20, 1e+20)
                real_weights_x_j = np.clip(np.exp(-kappa*(batch_train_labels_1.cpu().numpy()-batch_train_labels_2.cpu().numpy()-batch_epsilons_2)**2), 1e-20, 1e+20)
                fake_weights_x_j = np.clip(np.exp(-kappa*(batch_train_labels_1.cpu().numpy()+batch_epsilons_1-batch_train_labels_2.cpu().numpy()-batch_epsilons_2)**2), 1e-20, 1e+20)
            else:
                real_weights_x_j = np.zeros(BATCH_SIZE)
                indx = np.where(np.abs(batch_train_labels_1.cpu().numpy()-batch_train_labels_2.cpu().numpy()-batch_epsilons_2) <= kappa)[0]
                real_weights_x_j[indx] = 1

                fake_weights_x_j = np.zeros(BATCH_SIZE)
                indx = np.where(np.abs(batch_train_labels_1.cpu().numpy()+batch_epsilons_1-batch_train_labels_2.cpu().numpy()-batch_epsilons_2) <= kappa)[0]
                fake_weights_x_j[indx] = 1

            real_weights_x_j = torch.from_numpy(real_weights_x_j).type(torch.float).to(device)
            fake_weights_x_j = torch.from_numpy(fake_weights_x_j).type(torch.float).to(device)


            d_loss = - torch.mean(real_weights_x_j * torch.log(real_dis_out+1e-20)) - torch.mean(fake_weights_x_j * torch.log(1 - fake_dis_out+1e-20))


            d_loss.backward()
            optimizerD.step()


            gen_iterations += 1


            tfboard_writer.add_scalar('D loss', d_loss.item(), gen_iterations)
            tfboard_writer.add_scalar('G loss', g_loss.item(), gen_iterations)


            if batch_idx%20 == 0:
                print ("CcDCGAN: [Iter %d] [Epoch %d/%d] [D loss: %.4e] [G loss: %.4e] [real prob: %.3f] [fake prob: %.3f] [Time: %.4f]" % (gen_iterations, epoch + 1, epoch_GAN, d_loss.item(), g_loss.item(), real_dis_out.mean().item(), fake_dis_out.mean().item(), timeit.default_timer()-start_tmp))

            if gen_iterations % 100 == 0:
                netG.eval()
                with torch.no_grad():
                    gen_imgs = netG(z_fixed, y_fixed)
                    gen_imgs = gen_imgs.detach().cpu()
                    save_image(gen_imgs.data, save_images_folder +'%d.png' % gen_iterations, nrow=n_row, normalize=True)



        if save_models_folder is not None and (epoch+1) % 1000 == 0:
            save_file = save_models_folder + "/CcDCGAN_checkpoint_intrain/"
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_file = save_file + "CcDCGAN_checkpoint_epoch" + str(epoch+1) + ".pth"
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


def SampCcDCGAN_given_label(netG, label, path=None, dim_GAN = 128, NFAKE = 10000, batch_size = 500, device="cuda"):
    '''
    label: normalized label in [0,1]
    '''
    if batch_size>NFAKE:
        batch_size = NFAKE
    fake_images = np.zeros((NFAKE+batch_size, NC, IMG_SIZE, IMG_SIZE), dtype=np.float)
    netG=netG.to(device)
    netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < NFAKE:
            z = torch.randn(batch_size, dim_GAN, dtype=torch.float).to(device)
            y = np.ones(batch_size) * label
            y = torch.from_numpy(y).type(torch.float).view(-1,1).to(device)
            batch_fake_images = netG(z, y)
            fake_images[tmp:(tmp+batch_size)] = batch_fake_images.cpu().detach().numpy()
            tmp += batch_size

    #remove extra entries
    fake_images = fake_images[0:NFAKE]
    fake_labels = np.ones(NFAKE) * label #use assigned label

    if path is not None:
        raw_fake_images = (fake_images*0.5+0.5)*255.0
        raw_fake_images = raw_fake_images.astype(np.uint8)
        for i in range(NFAKE):
            filename = path + '/' + str(i) + '.jpg'
            im = Image.fromarray(raw_fake_images[i][0], mode='L')
            im = im.save(filename)

    return fake_images, fake_labels
