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
from torch import autograd

NC=1
IMG_SIZE=64

#MIN_LABEL=74
#MAX_LABEL=317


############################################################################################
# Train Continuous cWGANGP


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

def train_Continuous_cWGANGP(train_labels, kernel_sigma, threshold_type, kappa, epoch_GAN, dim_GAN, trainloader, netG, netD, optimizerG, optimizerD, save_images_folder, LAMBDA = 10, CRITIC_ITERS=5, save_models_folder = None, ResumeEpoch = 0, device="cuda", tfboard_writer=None):
    '''

    train_labels: training labels (numpy array)
    kernel_sigma: the sigma in the Guassian kernel (a real value)
    threshold_type: hard or soft threshold ('hard', 'soft')

    '''

    # kernels stuffs: assume Gaussian kernel
    train_labels = train_labels.reshape(-1) #do not shuffle train_labels because its order matters
    n_train = len(train_labels)

    std_count = np.std(train_labels)
    def sample_Gaussian(n, dim, mean=0, sigma=1):
        samples = np.random.normal(mean, sigma, n*dim)
        return samples.reshape((n, dim))


    # traning GAN model
    netG = netG.to(device)
    netD = netD.to(device)

    if save_models_folder is not None and ResumeEpoch>0:
        save_file = save_models_folder + "/cWGANGP_checkpoint_intrain/cDCGAN_checkpoint_epoch" + str(ResumeEpoch) + ".pth"
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
    # z_fixed = torch.randn(n_row**2, dim_GAN, dtype=torch.float).to(device)
    z_fixed = torch.from_numpy(sample_Gaussian(n_row**2, dim_GAN)).type(torch.float).to(device)

    unique_labels = np.array(list(set(train_labels)))
    unique_labels = np.sort(unique_labels)
    assert len(unique_labels) >= n_row
    y_fixed = np.zeros(n_row**2)
    for i in range(n_row):
        if i == 0:
            curr_label = np.min(unique_labels)
        else:
            if np.max(unique_labels)<=1:
                if curr_label+0.1 <= 1:
                    curr_label += 0.1
            else:
                curr_label += 10
        for j in range(n_row):
            y_fixed[i*n_row+j] = curr_label
    print(y_fixed)
    y_fixed = torch.from_numpy(y_fixed).type(torch.float).view(-1,1).to(device)



    start_tmp = timeit.default_timer()

    for epoch in range(ResumeEpoch, epoch_GAN):

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

                (batch_train_images, batch_train_labels) = data_iter.next()
                batch_idx += 1

                BATCH_SIZE = int(batch_train_images.shape[0]/2)
                batch_train_images_1 = batch_train_images[0:BATCH_SIZE].type(torch.float).to(device) #real x_j's
                batch_train_labels_1 = batch_train_labels[0:BATCH_SIZE].type(torch.float).to(device) #y_j's
                batch_train_labels_2 = batch_train_labels[BATCH_SIZE:].type(torch.float).to(device) #y_i's

                # generate Gaussian noise which are added to y_i (batch_train_labels_2)
                batch_epsilons_1 = np.random.normal(0, kernel_sigma, BATCH_SIZE)
                batch_epsilons_tensor_1 = torch.from_numpy(batch_epsilons_1).type(torch.float).to(device)

                batch_epsilons_2 = np.random.normal(0, kernel_sigma, BATCH_SIZE)
                batch_epsilons_tensor_2 = torch.from_numpy(batch_epsilons_2).type(torch.float).to(device)

                netD.zero_grad()

                real_dis_out = netD(batch_train_images_1, batch_train_labels_2 + batch_epsilons_tensor_2)

                # z_1 = torch.randn(BATCH_SIZE, dim_GAN, dtype=torch.float).to(device)
                z_1 = torch.from_numpy(sample_Gaussian(BATCH_SIZE, dim_GAN)).type(torch.float).to(device)
                batch_fake_images_1 = netG(z_1, batch_train_labels_1 + batch_epsilons_tensor_1)
                fake_dis_out = netD(batch_fake_images_1.detach(),batch_train_labels_2 + batch_epsilons_tensor_2)

                gradient_penalty = calc_gradient_penalty_WGAN(netD, batch_train_images_1.data, batch_fake_images_1.data, (batch_train_labels_2 + batch_epsilons_tensor_2).data, LAMBDA=LAMBDA, device=device)

                # compute weight for x_j when it is used to learn p(x|y_i+epsilon)
                if threshold_type == "soft":
                    real_weights_x_j = np.clip(np.exp(-kappa*(batch_train_labels_1.cpu().numpy()-batch_train_labels_2.cpu().numpy()-batch_epsilons_2)**2), 1e-20, 1e+20)
                    fake_weights_x_j = np.clip(np.exp(-kappa*(batch_train_labels_1.cpu().numpy()+batch_epsilons_1-batch_train_labels_2.cpu().numpy()-batch_epsilons_2)**2), 1e-20, 1e+20)
                else:
                    real_weights_x_j = np.zeros(BATCH_SIZE)
                    indx = np.where(np.abs(batch_train_labels_1.cpu().numpy()-batch_train_labels_2.cpu().numpy()-batch_epsilons_2) < kappa)[0]
                    real_weights_x_j[indx] = 1

                    fake_weights_x_j = np.zeros(BATCH_SIZE)
                    indx = np.where(np.abs(batch_train_labels_1.cpu().numpy()+batch_epsilons_1-batch_train_labels_2.cpu().numpy()-batch_epsilons_2) < kappa)[0]
                    fake_weights_x_j[indx] = 1

                real_weights_x_j = torch.from_numpy(real_weights_x_j).type(torch.float).to(device)
                fake_weights_x_j = torch.from_numpy(fake_weights_x_j).type(torch.float).to(device)

                d_loss = - torch.mean(real_weights_x_j * real_dis_out) - torch.mean(fake_weights_x_j * (1 - fake_dis_out)) + gradient_penalty

                d_loss.backward()
                optimizerD.step()


                with torch.no_grad():
                    real_correct_dis_out = netD(batch_train_images_1, batch_train_labels_1)
                    fake_correct_dis_out = netD(batch_fake_images_1, batch_train_labels_1 + batch_epsilons_tensor_1)
                    Wasserstein_D = torch.mean(real_correct_dis_out).cpu().item() - torch.mean(fake_correct_dis_out).cpu().item()

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()

            batch_epsilons_2 = np.random.normal(0, kernel_sigma, BATCH_SIZE)
            batch_epsilons_tensor_2 = torch.from_numpy(batch_epsilons_2).type(torch.float).to(device)

            # z_2 = torch.randn(BATCH_SIZE, dim_GAN, dtype=torch.float).to(device)
            z_2 = torch.from_numpy(sample_Gaussian(BATCH_SIZE, dim_GAN)).type(torch.float).to(device)
            batch_fake_images_2 = netG(z_2, batch_train_labels_2 + batch_epsilons_tensor_2)

            dis_out = netD(batch_fake_images_2, batch_train_labels_2 + batch_epsilons_tensor_2)

            g_loss = - dis_out.mean()
            g_loss.backward()
            optimizerG.step()

            gen_iterations += 1


            tfboard_writer.add_scalar('D loss', d_loss.item(), gen_iterations)
            tfboard_writer.add_scalar('G loss', g_loss.item(), gen_iterations)
            tfboard_writer.add_scalar('W distance', Wasserstein_D, gen_iterations)

            if batch_idx%20 == 0:
                print ("CcWGANGP: [Iter %d] [Epoch %d/%d] [D loss: %.4e] [G loss: %.4e] [W Dist: %.4f] [Time: %.4f]" % (gen_iterations, epoch + 1, epoch_GAN, d_loss.item(), g_loss.item(), Wasserstein_D, timeit.default_timer()-start_tmp))

            if gen_iterations % 100 == 0:
                with torch.no_grad():
                    gen_imgs = netG(z_fixed, y_fixed)
                    gen_imgs = gen_imgs.detach()
                save_image(gen_imgs.data, save_images_folder +'%d.png' % gen_iterations, nrow=n_row, normalize=True)


        if save_models_folder is not None and (epoch+1) % 500 == 0:
            save_file = save_models_folder + "/CcWGANGP_checkpoint_intrain/"
            if not os.path.exists(save_file):
                os.makedirs(save_file)
            save_file = save_file + "CcWGANGP_checkpoint_epoch" + str(epoch+1) + ".pth"
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





def SampCcGAN_given_label(netG, label, path=None, dim_GAN = 128, NFAKE = 10000, batch_size = 500, device="cuda"):
    #netD: whether assign weights to fake images via inversing f function (the f in f-GAN)
    if batch_size>NFAKE:
        batch_size = NFAKE
    raw_fake_images = np.zeros((NFAKE+batch_size, NC, IMG_SIZE, IMG_SIZE), dtype=np.float)
    netG=netG.to(device)
    netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < NFAKE:
            z = torch.randn(batch_size, dim_GAN, dtype=torch.float).to(device)
            y = np.ones(batch_size) * label
            y = torch.from_numpy(y).type(torch.float).view(-1,1).to(device)
            batch_fake_images = netG(z, y)
            raw_fake_images[tmp:(tmp+batch_size)] = batch_fake_images.cpu().detach().numpy()
            tmp += batch_size

    #remove extra entries
    raw_fake_images = raw_fake_images[0:NFAKE]

    raw_fake_images_renorm = (raw_fake_images*0.5+0.5)*255.0
    raw_fake_images_renorm = raw_fake_images_renorm.astype(np.uint8)

    if path is not None:
        for i in range(NFAKE):
            filename = path + '/' + str(i) + '.jpg'
            im = Image.fromarray(raw_fake_images_renorm[i][0], mode='L')
            im = im.save(filename)

    return raw_fake_images
