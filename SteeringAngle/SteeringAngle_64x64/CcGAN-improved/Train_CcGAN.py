import torch
import numpy as np
import os
import timeit
from PIL import Image
from torchvision.utils import save_image

from utils import *
from opts import parse_opts

''' Settings '''
args = parse_opts()
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# some parameters in opts
loss_type = args.loss_type_gan
niters = args.niters_gan
resume_niters = args.resume_niters_gan
dim_gan = args.dim_gan
lr_g = args.lr_g_gan
lr_d = args.lr_d_gan
save_niters_freq = args.save_niters_freq
batch_size_disc = args.batch_size_disc
batch_size_gene = args.batch_size_gene
batch_size_max = max(batch_size_disc, batch_size_gene)

threshold_type = args.threshold_type
nonzero_soft_weight_threshold = args.nonzero_soft_weight_threshold

NC = args.num_channels
IMG_SIZE = args.img_size

def train_CcGAN(kernel_sigma, kappa, train_images, train_labels, netG, netD, net_y2h, save_images_folder, save_models_folder = None, clip_label=False):

    '''
    Note that train_images are not normalized to [-1,1]
    '''

    netG = netG.to(device)
    netD = netD.to(device)
    net_y2h = net_y2h.to(device)
    net_y2h.eval()

    optimizerG = torch.optim.Adam(netG.parameters(), lr=lr_g, betas=(0.5, 0.999))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=lr_d, betas=(0.5, 0.999))

    if save_models_folder is not None and resume_niters>0:
        save_file = save_models_folder + "/CcGAN_{}_checkpoint_intrain/CcGAN_checkpoint_niters_{}.pth".format(threshold_type, resume_niters)
        checkpoint = torch.load(save_file)
        netG.load_state_dict(checkpoint['netG_state_dict'])
        netD.load_state_dict(checkpoint['netD_state_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG_state_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if

    #################
    unique_train_labels = np.sort(np.array(list(set(train_labels))))
    min_label = np.min(unique_train_labels)
    max_label = np.max(unique_train_labels)

    def find_outliers(input_labels):
        bool_larger = (input_labels>max_label).reshape(-1,1)
        bool_less = (input_labels<min_label).reshape(-1,1)
        bool_cat = np.concatenate((bool_larger, bool_less), axis=1)
        indx_clip = np.where(np.any(bool_cat, axis=1)==True)[0]
        return indx_clip

    # printed images with labels between the 5-th quantile and 95-th quantile of training labels
    n_row=10; n_col = n_row
    z_fixed = torch.randn(n_row*n_col, dim_gan, dtype=torch.float).to(device)
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


    start_time = timeit.default_timer()
    for niter in range(resume_niters, niters):

        ## randomly draw batch_size_disc y's from unique_train_labels
        batch_target_labels_in_dataset = np.random.choice(unique_train_labels, size=batch_size_max, replace=True)
        ## add Gaussian noise; we estimate image distribution conditional on these labels
        batch_epsilons = np.random.normal(0, kernel_sigma, batch_size_max)
        batch_target_labels_with_epsilon = batch_target_labels_in_dataset + batch_epsilons
        if clip_label:
            # batch_target_labels_with_epsilon = np.clip(batch_target_labels_with_epsilon, 0.0, 1.0)
            indx_clip = find_outliers(batch_target_labels_with_epsilon)
            while len(indx_clip)>1:
                batch_epsilons = np.random.normal(0, kernel_sigma, len(indx_clip))
                batch_target_labels_with_epsilon[indx_clip] = batch_target_labels_in_dataset[indx_clip] + batch_epsilons
                indx_clip = find_outliers(batch_target_labels_with_epsilon)

        '''  Train Discriminator   '''
        batch_target_labels = batch_target_labels_with_epsilon[0:batch_size_disc]

        ## find index of real images with labels in the vicinity of batch_target_labels
        ## generate labels for fake image generation; these labels are also in the vicinity of batch_target_labels
        batch_real_indx = np.zeros(batch_size_disc, dtype=int) #index of images in the datata; the labels of these images are in the vicinity
        batch_fake_labels = np.zeros(batch_size_disc)

        for j in range(batch_size_disc):
            ## index for real images
            if threshold_type == "hard":
                indx_real_in_vicinity = np.where(np.abs(train_labels-batch_target_labels[j])<= kappa)[0]
            else:
                # reverse the weight function for SVDL
                indx_real_in_vicinity = np.where((train_labels-batch_target_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold)/kappa)[0]

            ## if the max gap between two consecutive ordered unique labels is large, it is possible that len(indx_real_in_vicinity)<1
            while len(indx_real_in_vicinity)<1:
                batch_epsilons_j = np.random.normal(0, kernel_sigma, 1)
                batch_target_labels[j] = batch_target_labels_in_dataset[j] + batch_epsilons_j
                if clip_label:
                    # batch_target_labels = np.clip(batch_target_labels, 0.0, 1.0)
                    while batch_target_labels[j]>max_label or batch_target_labels[j]<min_label:
                        batch_epsilons_j = np.random.normal(0, kernel_sigma, 1)
                        batch_target_labels[j] = batch_target_labels_in_dataset[j] + batch_epsilons_j
                ## index for real images
                if threshold_type == "hard":
                    indx_real_in_vicinity = np.where(np.abs(train_labels-batch_target_labels[j])<= kappa)[0]
                else:
                    # reverse the weight function for SVDL
                    indx_real_in_vicinity = np.where((train_labels-batch_target_labels[j])**2 <= -np.log(nonzero_soft_weight_threshold)/kappa)[0]
            #end while len(indx_real_in_vicinity)<1

            assert len(indx_real_in_vicinity)>=1

            batch_real_indx[j] = np.random.choice(indx_real_in_vicinity, size=1)[0]

            ## labels for fake images generation
            if threshold_type == "hard":
                lb = batch_target_labels[j] - kappa
                ub = batch_target_labels[j] + kappa
            else:
                lb = batch_target_labels[j] - np.sqrt(-np.log(nonzero_soft_weight_threshold)/kappa)
                ub = batch_target_labels[j] + np.sqrt(-np.log(nonzero_soft_weight_threshold)/kappa)
            lb = max(0.0, lb); ub = min(ub, 1.0)
            assert lb<=ub
            assert lb>=0 and ub>=0
            assert lb<=1 and ub<=1
            batch_fake_labels[j] = np.random.uniform(lb, ub, size=1)[0]
        #end for j

        ## draw the real image batch from the training set
        batch_real_images = train_images[batch_real_indx]
        assert batch_real_images.max()>1
        batch_real_labels = train_labels[batch_real_indx]
        batch_real_labels = torch.from_numpy(batch_real_labels).type(torch.float).to(device)


        ## normalize real images
        trainset = IMGs_dataset(batch_real_images, labels=None, normalize=True)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_disc, shuffle=False)
        train_dataloader = iter(train_dataloader)
        batch_real_images = train_dataloader.next()
        assert len(batch_real_images) == batch_size_disc
        batch_real_images = batch_real_images.type(torch.float).to(device)
        assert batch_real_images.max().item()<=1


        ## generate the fake image batch
        batch_fake_labels = torch.from_numpy(batch_fake_labels).type(torch.float).to(device)
        z = torch.randn(batch_size_disc, dim_gan, dtype=torch.float).to(device)
        batch_fake_images = netG(z, net_y2h(batch_fake_labels))

        ## target labels on gpu
        batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(device)

        ## weight vector
        if threshold_type == "soft":
            real_weights = torch.exp(-kappa*(batch_real_labels-batch_target_labels)**2).to(device)
            fake_weights = torch.exp(-kappa*(batch_fake_labels-batch_target_labels)**2).to(device)
        else:
            real_weights = torch.ones(batch_size_disc, dtype=torch.float).to(device)
            fake_weights = torch.ones(batch_size_disc, dtype=torch.float).to(device)
        #end if threshold type

        # forward pass
        real_dis_out = netD(batch_real_images, net_y2h(batch_target_labels))
        fake_dis_out = netD(batch_fake_images.detach(), net_y2h(batch_target_labels))

        if loss_type == "vanilla":
            real_dis_out = torch.nn.Sigmoid()(real_dis_out)
            fake_dis_out = torch.nn.Sigmoid()(fake_dis_out)
            d_loss_real = - torch.log(real_dis_out+1e-20)
            d_loss_fake = - torch.log(1-fake_dis_out+1e-20)
        elif loss_type == "hinge":
            d_loss_real = torch.nn.ReLU()(1.0 - real_dis_out)
            d_loss_fake = torch.nn.ReLU()(1.0 + fake_dis_out)

        d_loss = torch.mean(real_weights.view(-1) * d_loss_real.view(-1)) + torch.mean(fake_weights.view(-1) * d_loss_fake.view(-1))

        optimizerD.zero_grad()
        d_loss.backward()
        optimizerD.step()


        '''  Train Generator   '''
        netG.train()

        # generate fake images
        batch_target_labels = batch_target_labels_with_epsilon[0:batch_size_gene]
        batch_target_labels = torch.from_numpy(batch_target_labels).type(torch.float).to(device)

        z = torch.randn(batch_size_gene, dim_gan, dtype=torch.float).to(device)
        batch_fake_images = netG(z, net_y2h(batch_target_labels))

        # loss
        dis_out = netD(batch_fake_images, net_y2h(batch_target_labels))
        if loss_type == "vanilla":
            dis_out = torch.nn.Sigmoid()(dis_out)
            g_loss = - torch.mean(torch.log(dis_out+1e-20))
        elif loss_type == "hinge":
            g_loss = - dis_out.mean()

        # backward
        optimizerG.zero_grad()
        g_loss.backward()
        optimizerG.step()

        # print loss
        if (niter+1) % 20 == 0:
            print ("CcGAN: [Iter %d/%d] [D loss: %.4e] [G loss: %.4e] [real prob: %.3f] [fake prob: %.3f] [Time: %.4f]" % (niter+1, niters, d_loss.item(), g_loss.item(), real_dis_out.mean().item(), fake_dis_out.mean().item(), timeit.default_timer()-start_time))

        if (niter+1) % 100 == 0:
            netG.eval()
            with torch.no_grad():
                gen_imgs = netG(z_fixed, net_y2h(y_fixed))
                gen_imgs = gen_imgs.detach().cpu()
                save_image(gen_imgs.data, save_images_folder + '/{}.png'.format(niter+1), nrow=n_row, normalize=True)

        if save_models_folder is not None and ((niter+1) % save_niters_freq == 0 or (niter+1) == niters):
            save_file = save_models_folder + "/CcGAN_{}_checkpoint_intrain/CcGAN_checkpoint_niters_{}.pth".format(threshold_type, niter+1)
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




def SampCcGAN_given_labels(netG, net_y2h, labels, path=None, NFAKE = 10000, batch_size = 500):
    '''
    labels: a numpy array; normalized label in [0,1]
    '''
    assert len(labels) == NFAKE
    if batch_size>NFAKE:
        batch_size = NFAKE
    fake_images = np.zeros((NFAKE+batch_size, NC, IMG_SIZE, IMG_SIZE), dtype=np.float)
    fake_labels = np.concatenate((labels, labels[0:batch_size]))
    netG=netG.to(device)
    netG.eval()

    with torch.no_grad():
        pb = SimpleProgressBar()
        tmp = 0
        while tmp < NFAKE:
            z = torch.randn(batch_size, dim_gan, dtype=torch.float).to(device)
            y = torch.from_numpy(fake_labels[tmp:(tmp+batch_size)]).type(torch.float).view(-1,1).to(device)
            batch_fake_images = netG(z, net_y2h(y))
            fake_images[tmp:(tmp+batch_size)] = batch_fake_images.cpu().detach().numpy()
            tmp += batch_size
            pb.update(min(float(tmp)/NFAKE, 1)*100)

    #remove extra entries
    fake_images = fake_images[0:NFAKE]
    fake_labels = fake_labels[0:NFAKE]

    if path is not None:
        raw_fake_images = (fake_images*0.5+0.5)*255.0
        raw_fake_images = raw_fake_images.astype(np.uint8)
        for i in range(NFAKE):
            filename = path + '/' + str(i) + '.jpg'
            im = Image.fromarray(raw_fake_images[i][0], mode='L')
            im = im.save(filename)

    return fake_images, fake_labels

def SampCcGAN_given_label(netG, net_y2h, label, path=None, NFAKE = 10000, batch_size = 500):
    '''
    label: a scalar; normalized label in [0,1]
    '''
    if batch_size>NFAKE:
        batch_size = NFAKE
    fake_images = np.zeros((NFAKE+batch_size, NC, IMG_SIZE, IMG_SIZE), dtype=np.float)
    netG=netG.to(device)
    netG.eval()

    with torch.no_grad():
        tmp = 0
        while tmp < NFAKE:
            z = torch.randn(batch_size, dim_gan, dtype=torch.float).to(device)
            y = np.ones(batch_size) * label
            y = torch.from_numpy(y).type(torch.float).view(-1,1).to(device)
            batch_fake_images = netG(z, net_y2h(y))
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
