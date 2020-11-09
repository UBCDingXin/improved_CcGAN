
import torch
import torch.nn as nn
import numpy as np
import os
import timeit

from utils import *
from opts import parse_opts

''' Settings '''
args = parse_opts()
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# some parameters in opts
niters = args.niters_gan
resume_niters = args.resume_niters_gan
dim_gan = args.dim_gan
lr = args.lr_gan
save_niters_freq = args.save_niters_freq
batch_size = min(args.batch_size_disc, args.batch_size_gene)

def train_cGAN(train_samples, train_labels, netG, netD, save_models_folder = None, plot_in_train=False, save_images_folder = None, samples_tar_eval = None, angle_grid_eval = None, num_classes=None, num_features = 2, unique_labels = None, label2class = None, fig_size=5, point_size=None):

    train_dataset = custom_dataset(train_samples, train_labels)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

    netG = netG.to(device)
    netD = netD.to(device)

    criterion = nn.BCELoss()
    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

    if save_models_folder is not None and resume_niters>0:
        save_file = save_models_folder + "/cGAN_checkpoint_intrain/cGAN_checkpoint_niters_{}.pth".format(resume_niters)
        checkpoint = torch.load(save_file)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
    #end if

    batch_idx = 0
    dataloader_iter = iter(train_dataloader)

    start_time = timeit.default_timer()
    for niter in range(resume_niters, niters):

        if batch_idx+1 == len(train_dataloader):
            dataloader_iter = iter(train_dataloader)
            batch_idx = 0

        # sample
        batch_train_samples, batch_train_labels = dataloader_iter.next()
        assert batch_size == batch_train_samples.shape[0]
        batch_train_samples = batch_train_samples.type(torch.float).to(device)
        batch_train_labels = batch_train_labels.type(torch.long).to(device)

        # Adversarial ground truths
        GAN_real = torch.ones(batch_size,1).to(device)
        GAN_fake = torch.zeros(batch_size,1).to(device)

        '''

        Train Generator: maximize log(D(G(z)))

        '''
        netG.train()

        # Sample noise and labels as generator input
        z = torch.randn(batch_size, dim_gan, dtype=torch.float).to(device)

        #generate samples
        batch_fake_samples = netG(z, batch_train_labels)

        # Loss measures generator's ability to fool the discriminator
        dis_out = netD(batch_fake_samples, batch_train_labels)

        #generator try to let disc believe gen_imgs are real
        g_loss = criterion(dis_out, GAN_real)
        #final g_loss consists of two parts one from generator's and the other one is from validity loss

        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()

        '''

        Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))

        '''

        # Measure discriminator's ability to classify real from generated samples
        prob_real = netD(batch_train_samples, batch_train_labels)
        prob_fake = netD(batch_fake_samples.detach(), batch_train_labels.detach())
        real_loss = criterion(prob_real, GAN_real)
        fake_loss = criterion(prob_fake, GAN_fake)
        d_loss = (real_loss + fake_loss) / 2

        optimizerD.zero_grad()
        d_loss.backward()
        optimizerD.step()

        batch_idx+=1

        if (niter+1)%100 == 0:
            print ("cGAN: [Iter %d/%d] [D loss: %.4f] [G loss: %.4f] [D prob real:%.4f] [D prob fake:%.4f] [Time: %.4f]" % (niter+1, niters, d_loss.item(), g_loss.item(), prob_real.mean().item(),prob_fake.mean().item(), timeit.default_timer()-start_time))


        if plot_in_train and (niter+1)%100==0:

            assert save_images_folder is not None
            n_samp_per_gaussian_eval = samples_tar_eval.shape[0]//len(angle_grid_eval)

            for j_tmp in range(len(angle_grid_eval)):
                angle = np.single(angle_grid_eval[j_tmp])
                # angle = (angle*2*np.pi).astype(np.single) #back to original scale of angle [0, 2*pi]
                prop_samples_curr, _ = SampcGAN_given_label(netG, angle, unique_labels, label2class, n_samp_per_gaussian_eval, batch_size = n_samp_per_gaussian_eval, num_features=num_features)

                if j_tmp == 0:
                    prop_samples = prop_samples_curr
                else:
                    prop_samples = np.concatenate((prop_samples, prop_samples_curr), axis=0)

            filename = save_images_folder + '/{}.png'.format(niter+1)
            ScatterPoints(samples_tar_eval, prop_samples, filename, fig_size=fig_size, point_size=point_size)

        if save_models_folder is not None and ((niter+1) % save_niters_freq == 0 or (niter+1) == niters):
            save_file = save_models_folder + "/cGAN_checkpoint_intrain/cGAN_checkpoint_niters_{}.pth".format(niter+1)
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



def SampcGAN(netG, class2label, NFAKE = 10000, batch_size = 500, num_classes = 100, num_features = 2):
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
            z = torch.randn(batch_size, dim_gan, dtype=torch.float).to(device)
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


def SampcGAN_given_label(netG, given_label, unique_labels, label2class, NFAKE = 10000, batch_size = 500, num_features=2):
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

    if batch_size>NFAKE:
        batch_size = NFAKE
    fake_samples = np.zeros((NFAKE+batch_size, num_features))
    netG=netG.to(device)
    netG.eval()
    with torch.no_grad():
        tmp = 0
        if closest_unique_label2 is None: #only one closest unique label
            while tmp < NFAKE:
                z = torch.randn(batch_size, dim_gan, dtype=torch.float).to(device)
                labels = torch.from_numpy(label2class[closest_unique_label1]*np.ones(batch_size)).type(torch.long).to(device)
                batch_fake_samples = netG(z, labels)
                fake_samples[tmp:(tmp+batch_size)] = batch_fake_samples.detach().cpu().numpy()
                tmp += batch_size
        else: #two closest unique labels
            while tmp < NFAKE//2:
                z = torch.randn(batch_size, dim_gan, dtype=torch.float).to(device)
                labels = torch.from_numpy(label2class[closest_unique_label1]*np.ones(batch_size)).type(torch.long).to(device)
                batch_fake_samples = netG(z, labels)
                fake_samples[tmp:(tmp+batch_size)] = batch_fake_samples.detach().cpu().numpy()
                tmp += batch_size

            while tmp < NFAKE-NFAKE//2:
                z = torch.randn(batch_size, dim_gan, dtype=torch.float).to(device)
                labels = torch.from_numpy(label2class[closest_unique_label2]*np.ones(batch_size)).type(torch.long).to(device)
                batch_fake_samples = netG(z, labels)
                fake_samples[tmp:(tmp+batch_size)] = batch_fake_samples.detach().cpu().numpy()
                tmp += batch_size

    #remove extra entries
    fake_samples = fake_samples[0:NFAKE]
    raw_fake_labels = np.ones(NFAKE) * given_label #use assigned label

    return fake_samples, raw_fake_labels
