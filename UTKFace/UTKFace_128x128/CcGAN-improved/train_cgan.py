
import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import os
import timeit

from utils import IMGs_dataset, SimpleProgressBar
from opts import parse_opts
from DiffAugment_pytorch import DiffAugment

''' Settings '''
args = parse_opts()

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
gan_arch = args.GAN_arch
num_D_steps = args.num_D_steps

visualize_freq = args.visualize_freq

num_workers = args.num_workers

NC = args.num_channels
IMG_SIZE = args.img_size

use_DiffAugment = args.gan_DiffAugment
policy = args.gan_DiffAugment_policy


## horizontal flip images
# def hflip_images(batch_images):
#     ''' for numpy arrays '''
#     uniform_threshold = np.random.uniform(0,1,len(batch_images))
#     indx_gt = np.where(uniform_threshold>0.5)[0]
#     batch_images[indx_gt] = np.flip(batch_images[indx_gt], axis=3)
#     return batch_images
def hflip_images(batch_images):
    ''' for torch tensors '''
    uniform_threshold = np.random.uniform(0,1,len(batch_images))
    indx_gt = np.where(uniform_threshold>0.5)[0]
    batch_images[indx_gt] = torch.flip(batch_images[indx_gt], dims=[3])
    return batch_images


def train_cgan(images, labels, netG, netD, save_images_folder, save_models_folder = None):

    netG = netG.cuda()
    netD = netD.cuda()

    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))

    trainset = IMGs_dataset(images, labels, normalize=True)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    unique_labels = np.sort(np.array(list(set(labels)))).astype(np.int)

    if save_models_folder is not None and resume_niters>0:
        save_file = save_models_folder + "/cGAN_{}_nDsteps_{}_checkpoint_intrain/cGAN_checkpoint_niters_{}.pth".format(gan_arch, num_D_steps, resume_niters)
        checkpoint = torch.load(save_file)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if

    n_row=10
    z_fixed = torch.randn(n_row**2, dim_gan, dtype=torch.float).cuda()
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
    y_fixed = torch.from_numpy(y_fixed).type(torch.long).cuda()


    batch_idx = 0
    dataloader_iter = iter(train_dataloader)

    start_time = timeit.default_timer()
    for niter in range(resume_niters, niters):

        if batch_idx+1 == len(train_dataloader):
            dataloader_iter = iter(train_dataloader)
            batch_idx = 0

        '''

        Train Generator: maximize log(D(G(z)))

        '''

        netG.train()

        # get training images
        _, batch_train_labels = dataloader_iter.next()
        assert batch_size == batch_train_labels.shape[0]
        batch_train_labels = batch_train_labels.type(torch.long).cuda()
        batch_idx+=1

        # Sample noise and labels as generator input
        z = torch.randn(batch_size, dim_gan, dtype=torch.float).cuda()

        #generate fake images
        batch_fake_images = netG(z, batch_train_labels)

        # Loss measures generator's ability to fool the discriminator
        if use_DiffAugment:
            dis_out = netD(DiffAugment(batch_fake_images, policy=policy), batch_train_labels)
        else:
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

        for _ in range(num_D_steps):

            if batch_idx+1 == len(train_dataloader):
                dataloader_iter = iter(train_dataloader)
                batch_idx = 0

            # get training images
            batch_train_images, batch_train_labels = dataloader_iter.next()
            assert batch_size == batch_train_images.shape[0]
            batch_train_images = batch_train_images.type(torch.float).cuda()
            batch_train_images = hflip_images(batch_train_images)
            batch_train_labels = batch_train_labels.type(torch.long).cuda()
            batch_idx+=1

            # Measure discriminator's ability to classify real from generated samples
            if use_DiffAugment:
                real_dis_out = netD(DiffAugment(batch_train_images, policy=policy), batch_train_labels)
                fake_dis_out = netD(DiffAugment(batch_fake_images.detach(), policy=policy), batch_train_labels.detach())
            else:
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

        

        if (niter+1)%20 == 0:
            print ("cGAN-%s: [Iter %d/%d] [D loss: %.4f] [G loss: %.4f] [D out real:%.4f] [D out fake:%.4f] [Time: %.4f]" % (gan_arch, niter+1, niters, d_loss.item(), g_loss.item(), real_dis_out.mean().item(),fake_dis_out.mean().item(), timeit.default_timer()-start_time))


        if (niter+1) % visualize_freq == 0:
            netG.eval()
            with torch.no_grad():
                gen_imgs = netG(z_fixed, y_fixed)
                gen_imgs = gen_imgs.detach()
            save_image(gen_imgs.data, save_images_folder +'/{}.png'.format(niter+1), nrow=n_row, normalize=True)

        if save_models_folder is not None and ((niter+1) % save_niters_freq == 0 or (niter+1) == niters):
            save_file = save_models_folder + "/cGAN_{}_nDsteps_{}_checkpoint_intrain/cGAN_checkpoint_niters_{}.pth".format(gan_arch, num_D_steps, niter+1)
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


def sample_cgan_given_labels(netG, given_labels, class_cutoff_points, batch_size = 200, denorm=True, to_numpy=True, verbose=True):
    '''
    given_labels: a numpy array; raw label without any normalization; not class label
    class_cutoff_points: the cutoff points to determine the membership of a give label
    '''

    class_cutoff_points = np.array(class_cutoff_points)
    num_classes = len(class_cutoff_points)-1

    nfake = len(given_labels)
    given_class_labels = np.zeros(nfake)
    for i in range(nfake):
        curr_given_label = given_labels[i]
        diff_tmp = class_cutoff_points - curr_given_label
        indx_nonneg = np.where(diff_tmp>=0)[0]
        if len(indx_nonneg)==1: #the last element of diff_tmp is non-negative
            curr_given_class_label = num_classes-1
            assert indx_nonneg[0] == num_classes
        elif len(indx_nonneg)>1:
            if diff_tmp[indx_nonneg[0]]>0:
                curr_given_class_label = indx_nonneg[0] - 1
            else:
                curr_given_class_label = indx_nonneg[0]
        given_class_labels[i] = curr_given_class_label
    given_class_labels = np.concatenate((given_class_labels, given_class_labels[0:batch_size]))

    if batch_size>nfake:
        batch_size = nfake
    fake_images = []
    netG=netG.cuda()
    netG.eval()
    with torch.no_grad():
        if verbose:
            pb = SimpleProgressBar()
        tmp = 0
        while tmp < nfake:
            z = torch.randn(batch_size, dim_gan, dtype=torch.float).cuda()
            labels = torch.from_numpy(given_class_labels[tmp:(tmp+batch_size)]).type(torch.long).cuda()
            if labels.max().item()>num_classes:
                print("Error: max label {}".format(labels.max().item()))
            batch_fake_images = netG(z, labels)
            if denorm: #denorm imgs to save memory
                assert batch_fake_images.max().item()<=1.0 and batch_fake_images.min().item()>=-1.0
                batch_fake_images = batch_fake_images*0.5+0.5
                batch_fake_images = batch_fake_images*255.0
                batch_fake_images = batch_fake_images.type(torch.uint8)
                # assert batch_fake_images.max().item()>1
            fake_images.append(batch_fake_images.detach().cpu())
            tmp += batch_size
            if verbose:
                pb.update(min(float(tmp)/nfake, 1)*100)

    fake_images = torch.cat(fake_images, dim=0)
    #remove extra entries
    fake_images = fake_images[0:nfake]

    if to_numpy:
        fake_images = fake_images.numpy()

    return fake_images, given_labels