"""
Train a regression DCGAN

"""

import torch
from torchvision.utils import save_image
import numpy as np
import os
import timeit
from torch.utils.tensorboard import SummaryWriter

NC=1
IMG_SIZE=64

MIN_LABEL=74
MAX_LABEL=317


############################################################################################
# Train Continuous cDCGAN
def train_Continuous_cDCGAN(train_labels, kernel_sigma, threshold_type, epoch_GAN, dim_GAN, trainloader, netG, netD, optimizerG, optimizerD, save_images_folder, save_models_folder = None, ResumeEpoch = 0, device="cuda", tfboard_writer=None):
    '''

    train_labels: training labels (numpy array)
    kernel_sigma: the sigma in the Guassian kernel (a real value)
    threshold_type: hard or soft threshold ('hard', 'soft')

    '''

    # kernels stuffs: assume Gaussian kernel
    train_labels = train_labels.reshape(-1) #do not shuffle train_labels because its order matters
    n_train = len(train_labels)

    def label_to_index(label): #label to its index in train_labels according to its distance to each element of train_labels
        diff_abs = np.abs(train_labels-label)
        return np.argsort(diff_abs)[0]

    if threshold_type == "soft":
        exp_y_diff_square = np.identity(n_train) #in sum(Kernel(y_l-y_i-epsilon)); exp(-(y_l-y_i)**2/(2*kernel_sigma**2))
        for l in range(n_train):
            for i in range(l+1, n_train):
                exp_y_diff_square[l,i] = np.clip(np.exp(-kernel_sigma*(train_labels[l]-train_labels[i])**2), 1e-20, 1e+20)
        i_lower = np.tril_indices(n_train, -1)
        exp_y_diff_square[i_lower] = exp_y_diff_square.T[i_lower] #copy the upper triangle to the lower triangle

        exp_y_diff_square_dict = dict() #key: index of y_i; value: an array of exp(-(y_l-y_i)**2/(2*kernel_sigma**2)), l=1,...,n_train
        for i in range(n_train):
            exp_y_diff_square_dict[i] = exp_y_diff_square[:,i]

        exp_y_diff = np.identity(n_train) #in sum(Kernel(y_l-y_i-epsilon)); exp((y_l-y_i)/(sigma^2))
        for l in range(n_train):
            for i in range(n_train):
                if l!=i:
                    exp_y_diff[l,i] = np.clip(np.exp(2*kernel_sigma*(train_labels[l]-train_labels[i])), 1e-20, 1e+20)

        exp_y_diff_dict = dict() #key: index of y_i; value: an array of exp((y_l-y_i)/(kernel_sigma**2)), l=1,...,n_train
        for i in range(n_train):
            exp_y_diff_dict[i] = exp_y_diff[:,i]

        def weights_yi(y_j, y_i, epsilon): # for fixed y_i and epsilon
            numerator = 1 + np.clip(np.exp(-kernel_sigma*(y_j-y_i-epsilon)**2))
            denominator = np.sum(1 + exp_y_diff_square_dict[label_to_index(y_i)] * (exp_y_diff_dict[label_to_index(y_i)]**epsilon) * np.clip(np.exp(-kernel_sigma*epsilon**2), 1e-20, 1e+20))
            return numerator/denominator

    # traning GAN model
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

    n_row=10
    z_fixed = torch.randn(n_row**2, dim_GAN, dtype=torch.float).to(device)
    y_fixed = train_labels[0:n_row**2].astype(np.float)
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
            optimizerG.zero_grad()

            # sample noise as generator's input; generate fake images with length BATCH_SIZE
            z = torch.randn(BATCH_SIZE, dim_GAN, dtype=torch.float).to(device)
            batch_fake_images_2 = netG(z, batch_train_labels_2 + batch_epsilons_tensor_2)


            # Loss measures generator's ability to fool the discriminator
            dis_out = netD(batch_fake_images_2, batch_train_labels_2 + batch_epsilons_tensor_2)

            # no weights
            g_loss = - torch.mean(torch.log(dis_out))

            g_loss.backward()
            optimizerG.step()


            '''

            Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))

            '''

            #train discriminator once and generator several times
            optimizerD.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_dis_out = netD(batch_train_images_1, batch_train_labels_2 + batch_epsilons_tensor_2)

            batch_fake_images_1 = netG(z, batch_train_labels_1 + batch_epsilons_tensor_1)
            fake_dis_out = netD(batch_fake_images_1.detach(),batch_train_labels_2 + batch_epsilons_tensor_2)

            # compute weight for x_j when it is used to learn p(x|y_i+epsilon)
            real_weights_x_j = np.zeros(BATCH_SIZE)
            for j in range(BATCH_SIZE):
                if threshold_type == "soft":
                    real_weights_x_j[j] = weights_yi(batch_train_labels_1[j].item(), batch_train_labels_2[j].item(), batch_epsilons_2[j])
                else:
                    if np.abs(batch_train_labels_1[j].item()-batch_train_labels_2[j].item()-batch_epsilons_2[j]) < kernel_sigma:
                        real_weights_x_j[j] = 1
            real_weights_x_j = torch.from_numpy(real_weights_x_j).type(torch.float).to(device)


            fake_weights_x_j = np.zeros(BATCH_SIZE)
            for j in range(BATCH_SIZE):
                if threshold_type == "soft":
                    fake_weights_x_j[j] = weights_yi(batch_train_labels_1[j].item(), batch_train_labels_2[j].item(), batch_epsilons_2[j]-batch_epsilons_1[j])
                else:
                    if np.abs(batch_train_labels_1[j].item()+batch_epsilons_1[j]-batch_train_labels_2[j].item()-batch_epsilons_2[j]) < kernel_sigma:
                        fake_weights_x_j[j] = 1
            fake_weights_x_j = torch.from_numpy(fake_weights_x_j).type(torch.float).to(device)

            d_loss = - torch.mean(real_weights_x_j * torch.log(real_dis_out)) - torch.mean(fake_weights_x_j * torch.log(1 - fake_dis_out)) #only add weights to real out

            d_loss.backward()
            optimizerD.step()


            gen_iterations += 1


            tfboard_writer.add_scalar('D loss', d_loss.item(), gen_iterations)
            tfboard_writer.add_scalar('G loss', g_loss.item(), gen_iterations)


            if batch_idx%20 == 0:
                print ("CcDCGAN: [Iter %d] [Epoch %d/%d] [D loss: %.4e] [G loss: %.4e] [real prob: %.3f] [fake prob: %.3f] [Time: %.4f]" % (gen_iterations, epoch + 1, epoch_GAN, d_loss.item(), g_loss.item(), real_dis_out.mean().item(), fake_dis_out.mean().item(), timeit.default_timer()-start_tmp))

            if gen_iterations % 100 == 0:
                with torch.no_grad():
                    gen_imgs = netG(z_fixed, y_fixed)
                    gen_imgs = gen_imgs.detach()
                save_image(gen_imgs.data, save_images_folder +'%d.png' % gen_iterations, nrow=n_row, normalize=True)

        if save_models_folder is not None and (epoch+1) % 500 == 0:
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


def SampcDCGAN(netG, dim_GAN = 128, NFAKE = 10000, batch_size = 500, device="cuda", normalize_count = False, shift_label = 0, max_label = 1):
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
            z = torch.randn(batch_size, dim_GAN, dtype=torch.float).to(device)
            y = np.random.randint(MIN_LABEL, MAX_LABEL, n_row**2)
            y = torch.from_numpy(y).type(torch.float).view(-1,1).to(device)
            if normalize_count:
                y += shift_label
                y /= max_label
                y = (y-0.5)/0.5

            batch_fake_images = netG(z, y)
            raw_fake_images[tmp:(tmp+batch_size)] = batch_fake_images.cpu().detach().numpy()
            raw_fake_counts[tmp:(tmp+batch_size)] = y.cpu().view(-1).detach().numpy()
            tmp += batch_size

    #remove extra entries
    raw_fake_images = raw_fake_images[0:NFAKE]
    raw_fake_counts = raw_fake_counts[0:NFAKE]

    return raw_fake_images, raw_fake_counts.reshape(-1)
