import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import os
import timeit

from utils import IMGs_dataset, SimpleProgressBar
from opts import parse_opts

''' Settings '''
args = parse_opts()

# some parameters in opts
gan_arch = args.GAN_arch
niters = args.niters_gan
resume_niters = args.resume_niters_gan
dim_z = args.dim_z
lr_g = args.lr_g
lr_d = args.lr_d
save_niters_freq = args.save_niters_freq
visualize_freq = args.visualize_freq
batch_size = args.batch_size

num_channels = args.num_channels
img_size = args.img_size
max_label = args.max_label
num_workers = args.num_workers


def train_cgan(train_images, train_labels, netG, netD, save_images_folder, save_models_folder = None):
    
    netG = netG.cuda()
    netD = netD.cuda()

    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))

    trainset = IMGs_dataset(train_images, train_labels, normalize=True)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    unique_labels = np.sort(np.array(list(set(train_labels)))).astype(np.int)

    if save_models_folder is not None and resume_niters>0:
        save_file = save_models_folder + "/{}_checkpoint_intrain/{}_checkpoint_niters_{}.pth".format(gan_arch,gan_arch,resume_niters)
        checkpoint = torch.load(save_file)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if

    # printed images with labels between the 5-th quantile and 95-th quantile of training labels
    n_row=10; n_col = n_row
    z_fixed = torch.randn(n_row*n_col, dim_z, dtype=torch.float).cuda()
    start_label = np.quantile(train_labels, 0.05)
    end_label = np.quantile(train_labels, 0.95)
    selected_labels = np.linspace(start_label, end_label, num=n_row)
    y_fixed = np.zeros(n_row*n_col)
    for i in range(n_row):
        curr_label = selected_labels[i]
        for j in range(n_col):
            y_fixed[i*n_col+j] = curr_label
    print(y_fixed)
    y_fixed = torch.from_numpy(y_fixed).type(torch.float).view(-1,1).cuda()


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
        batch_train_images = batch_train_images.type(torch.float).cuda()
        batch_train_labels = batch_train_labels.type(torch.long).cuda()

        # Adversarial ground truths
        GAN_real = torch.ones(batch_size,1).cuda()
        GAN_fake = torch.zeros(batch_size,1).cuda()


        '''

        Train Generator: maximize log(D(G(z)))

        '''
        netG.train()

        # Sample noise and labels as generator input
        z = torch.randn(batch_size, dim_z, dtype=torch.float).cuda()

        #generate fake images
        batch_fake_images = netG(z, batch_train_labels)

        # Loss measures generator's ability to fool the discriminator
        dis_out = netD(batch_fake_images, batch_train_labels)

        #generator try to let disc believe gen_imgs are real
        g_loss = criterion(dis_out, GAN_real)

        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()


        '''

        Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))

        '''

        # Measure discriminator's ability to classify real from generated samples
        prob_real = netD(batch_train_images, batch_train_labels)
        prob_fake = netD(batch_fake_images.detach(), batch_train_labels.detach())
        real_loss = criterion(prob_real, GAN_real)
        fake_loss = criterion(prob_fake, GAN_fake)
        d_loss = (real_loss + fake_loss) / 2

        optimizerD.zero_grad()
        d_loss.backward()
        optimizerD.step()

        batch_idx+=1

        if (niter+1)%20 == 0:
            print ("%s-concat: [Iter %d/%d] [D loss: %.4f] [G loss: %.4f] [D prob real:%.4f] [D prob fake:%.4f] [Time: %.4f]" % (gan_arch, niter+1, niters, d_loss.item(), g_loss.item(), prob_real.mean().item(),prob_fake.mean().item(), timeit.default_timer()-start_time))


        if (niter+1) % visualize_freq == 0:
            netG.eval()
            with torch.no_grad():
                gen_imgs = netG(z_fixed, y_fixed)
                gen_imgs = gen_imgs.detach()
            save_image(gen_imgs.data, save_images_folder +'/{}.png'.format(niter+1), nrow=n_row, normalize=True)

        if save_models_folder is not None and ((niter+1) % save_niters_freq == 0 or (niter+1) == niters):
            save_file = save_models_folder + "/{}_checkpoint_intrain/{}_checkpoint_niters_{}.pth".format(gan_arch,gan_arch,niter+1)
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



def sample_cgan_given_labels(netG, given_labels, batch_size = 500):
    '''
    netG: pretrained generator network
    given_labels: float. unnormalized labels. we need to convert them to values in [0,1]. 
    '''

    ## num of fake images will be generated
    nfake = len(given_labels)

    ## normalize regression
    labels = given_labels/max_label

    ## generate images
    if batch_size>nfake:
        batch_size = nfake

    netG=netG.cuda()
    netG.eval()

    ## concat to avoid out of index errors
    labels = np.concatenate((labels, labels[0:batch_size]), axis=0)

    fake_images = []

    with torch.no_grad():
        # pb = SimpleProgressBar()
        tmp = 0
        while tmp < nfake:
            z = torch.randn(batch_size, dim_z, dtype=torch.float).cuda()
            c = torch.from_numpy(labels[tmp:(tmp+batch_size)]).type(torch.float).cuda()
            batch_fake_images = netG(z, c)
            fake_images.append(batch_fake_images.detach().cpu().numpy())
            tmp += batch_size
            # pb.update(min(float(tmp)/nfake, 1)*100)

    fake_images = np.concatenate(fake_images, axis=0)
    #remove extra images
    fake_images = fake_images[0:nfake]

    #denomarlized fake images
    if fake_images.max()<=1.0:
        fake_images = fake_images*0.5+0.5
        fake_images = (fake_images*255.0).astype(np.uint8)
        
    return fake_images, given_labels