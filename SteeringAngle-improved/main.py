print("\n===================================================================================================")

import argparse
import copy
import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import h5py
import os
import random
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import timeit
from PIL import Image

from opts import parse_opts
args = parse_opts()
wd = args.root_path
os.chdir(wd)

from utils import *
from models import *
from Train_cGAN import *
from Train_CcGAN import *
from Train_net_for_label_embed import train_net_embed, train_net_y2h
from eval_metrics import cal_FID, cal_labelscore

#-----------------------------
# images
NC = args.num_channels #number of channels
IMG_SIZE = args.img_size

#--------------------------------
# system
NGPU = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NCPU = 8


#-------------------------------
# Embedding
base_lr_x2y = 0.01
base_lr_y2h = 0.01

# -------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

#-------------------------------
# output folders
save_models_folder = wd + '/output/saved_models'
os.makedirs(save_models_folder, exist_ok=True)
save_images_folder = wd + '/output/saved_images'
os.makedirs(save_images_folder, exist_ok=True)

#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
# data loader
data_filename = args.data_path + '/SteeringAngle_{}x{}.h5'.format(IMG_SIZE, IMG_SIZE)
hf = h5py.File(data_filename, 'r')
labels = hf['labels'][:]
labels = labels.astype(np.float)
images = hf['images'][:]
hf.close()

# remove too small angles and too large angles
q1 = args.min_label
q2 = args.max_label
indx = np.where((labels>q1)*(labels<q2)==True)[0]
labels = labels[indx]
images = images[indx]
assert len(labels)==len(images)

raw_images = copy.deepcopy(images) #backup images;
raw_labels = copy.deepcopy(labels) #backup raw labels; we may normalize labels later

### show some real  images
if args.show_real_imgs:
    unique_labels_show = np.array(sorted(list(set(labels))))
    indx_show = np.arange(0, len(unique_labels_show), len(unique_labels_show)//9)
    unique_labels_show = unique_labels_show[indx_show]
    nrow = len(unique_labels_show); ncol = 1
    # images_show = np.zeros((nrow*ncol, images.shape[1], images.shape[2], images.shape[3]))
    sel_labels_indx = []
    for i in range(nrow):
        curr_label = unique_labels_show[i]
        indx_curr_label = np.where(labels==curr_label)[0]
        np.random.shuffle(indx_curr_label)
        indx_curr_label = indx_curr_label[0:ncol]
        sel_labels_indx.extend(list(indx_curr_label))
    sel_labels_indx = np.array(sel_labels_indx)
    images_show = images[sel_labels_indx]
    print(images_show.mean())
    images_show = (images_show/255.0-0.5)/0.5
    images_show = torch.from_numpy(images_show)
    save_image(images_show.data, save_images_folder +'/real_images_grid_{}x{}.png'.format(nrow, ncol), nrow=ncol, normalize=True)

# for each angle, take no more than args.max_num_img_per_label images
image_num_threshold = args.max_num_img_per_label
print("\n Original set has {} images; For each angle, take no more than {} images>>>".format(len(images), image_num_threshold))
unique_labels_tmp = np.sort(np.array(list(set(labels))))
for i in tqdm(range(len(unique_labels_tmp))):
    indx_i = np.where(labels == unique_labels_tmp[i])[0]
    if len(indx_i)>image_num_threshold:
        np.random.shuffle(indx_i)
        indx_i = indx_i[0:image_num_threshold]
    if i == 0:
        sel_indx = indx_i
    else:
        sel_indx = np.concatenate((sel_indx, indx_i))
images = images[sel_indx]
labels = labels[sel_indx]
print("{} images left and there are {} unique labels".format(len(images), len(set(labels))))

## print number of images for each label
unique_labels_tmp = np.sort(np.array(list(set(labels))))
num_img_per_label_all = np.zeros(len(unique_labels_tmp))
for i in range(len(unique_labels_tmp)):
    indx_i = np.where(labels==unique_labels_tmp[i])[0]
    num_img_per_label_all[i] = len(indx_i)
#print(list(num_img_per_label_all))
data_csv = np.concatenate((unique_labels_tmp.reshape(-1,1), num_img_per_label_all.reshape(-1,1)), 1)
np.savetxt(wd + '/label_dist.csv', data_csv, delimiter=',')


## replicate minority samples to alleviate the imbalance issue
max_num_img_per_label_after_replica = args.max_num_img_per_label_after_replica
if max_num_img_per_label_after_replica>1:
    unique_labels_replica = np.sort(np.array(list(set(labels))))
    num_labels_replicated = 0
    print("Start replicating minority samples >>>")
    for i in tqdm(range(len(unique_labels_replica))):
        curr_label = unique_labels_replica[i]
        indx_i = np.where(labels == curr_label)[0]
        if len(indx_i) < max_num_img_per_label_after_replica:
            num_img_less = max_num_img_per_label_after_replica - len(indx_i)
            indx_replica = np.random.choice(indx_i, size = num_img_less, replace=True)
            if num_labels_replicated == 0:
                images_replica = images[indx_replica]
                labels_replica = labels[indx_replica]
            else:
                images_replica = np.concatenate((images_replica, images[indx_replica]), axis=0)
                labels_replica = np.concatenate((labels_replica, labels[indx_replica]))
            num_labels_replicated+=1
    #end for i
    images = np.concatenate((images, images_replica), axis=0)
    labels = np.concatenate((labels, labels_replica))
    print("We replicate {} images and labels \n".format(len(images_replica)))
    del images_replica, labels_replica; gc.collect()
# hist_filename = wd + "/histogram_replica_angle_" + str(args.img_size) + 'x' + str(args.img_size)
# num_bins = 500
# plt.figure()
# plt.hist(labels, num_bins, facecolor='blue', density=False)
# plt.savefig(hist_filename)


# plot the histogram of unnormalized labels
hist_filename = wd + "/histogram_unnormalized_angle_" + str(args.img_size) + 'x' + str(args.img_size)
num_bins = 500
plt.figure()
plt.hist(labels, num_bins, facecolor='blue', density=False)
plt.savefig(hist_filename)


# normalize labels
print("\n Range of unnormalized labels: ({},{})".format(np.min(labels), np.max(labels)))
min_label_before_shift = np.min(labels)
max_label_after_shift = np.max(labels+np.abs(min_label_before_shift))

if args.GAN == "cGAN": #treated as classification; convert angles to class labels
    unique_labels = np.sort(np.array(list(set(labels))))
    num_unique_labels = len(unique_labels)
    print("{} unique labels are split into {} classes".format(num_unique_labels, args.cGAN_num_classes))

    ## convert steering angles to class labels and vice versa
    ### step 1: prepare two dictionaries
    label2class = dict()
    class2label = dict()
    num_labels_per_class = num_unique_labels//args.cGAN_num_classes
    class_cutoff_points = [unique_labels[0]] #the cutoff points on [min_label, max_label] to determine classes
    curr_class = 0
    for i in range(num_unique_labels):
        label2class[unique_labels[i]]=curr_class
        if (i+1)%num_labels_per_class==0 and (curr_class+1)!=args.cGAN_num_classes:
            curr_class += 1
            class_cutoff_points.append(unique_labels[i+1])
    class_cutoff_points.append(unique_labels[-1])
    assert len(class_cutoff_points)-1 == args.cGAN_num_classes

    for i in range(args.cGAN_num_classes):
        class2label[i] = (class_cutoff_points[i]+class_cutoff_points[i+1])/2

    ### step 2: convert angles to class labels
    labels_new = -1*np.ones(len(labels))
    for i in range(len(labels)):
        labels_new[i] = label2class[labels[i]]
    assert np.sum(labels_new<0)==0
    labels = labels_new
    del labels_new; gc.collect()
    unique_labels = np.sort(np.array(list(set(labels)))).astype(np.int)
    assert len(unique_labels) == args.cGAN_num_classes

else:
    # normalize labels to [0,1]
    labels += np.abs(min_label_before_shift)
    labels /= max_label_after_shift
    print("min_label_before_shift: {}; max_label_after_shift: {}".format(min_label_before_shift, max_label_after_shift))

    # plot the histogram of normalized labels
    hist_filename = wd + "/histogram_normalized_angle_" + str(args.img_size) + 'x' + str(args.img_size)
    num_bins = 500
    plt.figure()
    plt.hist(labels, num_bins, facecolor='blue', density=False)
    plt.savefig(hist_filename)

    print("\n Range of normalized labels: ({},{})".format(np.min(labels), np.max(labels)))

    unique_labels_norm = np.sort(np.array(list(set(labels))))

    print("\n There are {} unique labels.".format(len(unique_labels_norm)))

    if args.kernel_sigma<0:
        std_label = np.std(labels)
        args.kernel_sigma = 1.06*std_label*(len(labels))**(-1/5)

        print("\n Use rule-of-thumb formula to compute kernel_sigma >>>")
        print("\n The std of {} labels is {} so the kernel sigma is {}".format(len(labels), std_label, args.kernel_sigma))

    if args.kappa<0:
        n_unique = len(unique_labels_norm)

        diff_list = []
        for i in range(1,n_unique):
            diff_list.append(unique_labels_norm[i] - unique_labels_norm[i-1])
        kappa_base = np.abs(args.kappa)*np.max(np.array(diff_list))

        if args.threshold_type=="hard":
            args.kappa = kappa_base
        else:
            args.kappa = 1/kappa_base**2
## end if args.GAN

#######################################################################################
'''               Pre-trained CNN and GAN for label embedding                       '''
#######################################################################################
if args.GAN == "CcGAN":
    net_embed_filename_ckpt = save_models_folder + '/ckpt_{}_epoch_{}_seed_{}.pth'.format(args.net_embed, args.epoch_cnn_embed, args.seed)
    net_y2h_filename_ckpt = save_models_folder + '/ckpt_net_y2h_epoch_{}_seed_{}.pth'.format(args.epoch_net_y2h, args.seed)

    labels_train_embed = raw_labels + np.abs(min_label_before_shift)
    labels_train_embed /= max_label_after_shift
    unique_labels_norm_embed = np.sort(np.array(list(set(labels_train_embed))))
    print("\n labels_train_embed: min={}, max={}".format(np.min(labels_train_embed), np.max(labels_train_embed)))
    trainset_embed = IMGs_dataset(raw_images, labels_train_embed, normalize=True)
    trainloader_embed_net = torch.utils.data.DataLoader(trainset_embed, batch_size=args.batch_size_embed, shuffle=True, num_workers=8) #use data before replication

    if args.net_embed == "ResNet18_embed":
        net_embed = ResNet18_embed(dim_embed=args.dim_embed, ngpu = NGPU)
    elif args.net_embed == "ResNet34_embed":
        net_embed = ResNet34_embed(dim_embed=args.dim_embed, ngpu = NGPU)
    elif args.net_embed == "ResNet50_embed":
        net_embed = ResNet50_embed(dim_embed=args.dim_embed, ngpu = NGPU)
    net_embed = net_embed.to(device)

    net_y2h = model_y2h(dim_embed=args.dim_embed)
    net_y2h = net_y2h.to(device)

    ## (1). Train net_embed first: x2h+h2y
    if not os.path.isfile(net_embed_filename_ckpt):
        print("\n Start training CNN for label embedding >>>")
        optimizer_net_embed = torch.optim.SGD(net_embed.parameters(), lr = base_lr_x2y, momentum= 0.9, weight_decay=1e-4)
        net_embed = train_net_embed(trainloader_embed_net, None, net_embed, optimizer_net_embed, epochs=args.epoch_cnn_embed, base_lr=base_lr_x2y, save_models_folder = save_models_folder, resumeepoch = args.resumeepoch_cnn_embed)
        # save model
        torch.save({
        'net_state_dict': net_embed.state_dict(),
        }, net_embed_filename_ckpt)
    else:
        print("\n net_embed ckpt already exists")
        print("\n Loading...")
        checkpoint = torch.load(net_embed_filename_ckpt)
        net_embed.load_state_dict(checkpoint['net_state_dict'])
    #end not os.path.isfile

    ## (2). Train y2h
    #train a net which maps a label back to the embedding space
    if not os.path.isfile(net_y2h_filename_ckpt):
        print("\n Start training net_y2h >>>")
        optimizer_net_y2h = torch.optim.SGD(net_y2h.parameters(), lr = base_lr_y2h, momentum = 0.9, weight_decay=1e-4)
        net_y2h = train_net_y2h(unique_labels_norm_embed, net_y2h, net_embed, optimizer_net_y2h, epochs=args.epoch_net_y2h, base_lr=base_lr_y2h, batch_size=128)
        # save model
        torch.save({
        'net_state_dict': net_y2h.state_dict(),
        }, net_y2h_filename_ckpt)
    else:
        print("\n net_y2h ckpt already exists")
        print("\n Loading...")
        checkpoint = torch.load(net_y2h_filename_ckpt)
        net_y2h.load_state_dict(checkpoint['net_state_dict'])
    #end not os.path.isfile

    ##some simple test
    indx_tmp = np.arange(len(unique_labels_norm_embed))
    np.random.shuffle(indx_tmp)
    indx_tmp = indx_tmp[:10]
    labels_tmp = unique_labels_norm_embed[indx_tmp].reshape(-1,1)
    labels_tmp = torch.from_numpy(labels_tmp).type(torch.float).to(device)
    epsilons_tmp = np.random.normal(0, 0.2, len(labels_tmp))
    epsilons_tmp = torch.from_numpy(epsilons_tmp).view(-1,1).type(torch.float).to(device)
    labels_noise_tmp = torch.clamp(labels_tmp+epsilons_tmp, 0.0, 1.0)
    net_embed.eval()
    net_h2y = net_embed.h2y
    net_y2h.eval()
    with torch.no_grad():
        labels_hidden_tmp = net_y2h(labels_tmp)
        labels_noise_hidden_tmp = net_y2h(labels_noise_tmp)
        labels_rec_tmp = net_h2y(labels_hidden_tmp).cpu().numpy().reshape(-1,1)
        labels_noise_rec_tmp = net_h2y(labels_noise_hidden_tmp).cpu().numpy().reshape(-1,1)
        labels_hidden_tmp = labels_hidden_tmp.cpu().numpy()
        labels_noise_hidden_tmp = labels_noise_hidden_tmp.cpu().numpy()
    labels_tmp = labels_tmp.cpu().numpy()
    labels_noise_tmp = labels_noise_tmp.cpu().numpy()
    results1 = np.concatenate((labels_tmp, labels_rec_tmp), axis=1)
    print("\n labels vs reconstructed labels")
    print(results1)

    labels_diff = (labels_tmp-labels_noise_tmp)**2
    hidden_diff = np.mean((labels_hidden_tmp-labels_noise_hidden_tmp)**2, axis=1, keepdims=True)
    results2 = np.concatenate((labels_diff, hidden_diff), axis=1)
    print("\n labels diff vs hidden diff")
    print(results2)


#######################################################################################
'''                                    GAN training                                 '''
#######################################################################################
print("{}, Sigma is {}, Kappa is {}".format(args.threshold_type, args.kernel_sigma, args.kappa))

if args.GAN == 'CcGAN':
    save_GANimages_InTrain_folder = save_images_folder + '/{}_{}_{}_{}_InTrain'.format(args.GAN, args.threshold_type, args.kernel_sigma, args.kappa)
else:
    save_GANimages_InTrain_folder = save_images_folder + '/{}_nclass_{}_InTrain'.format(args.GAN, args.cGAN_num_classes)
os.makedirs(save_GANimages_InTrain_folder, exist_ok=True)

start = timeit.default_timer()
print("\n Begin Training %s:" % args.GAN)
#----------------------------------------------
# cGAN: treated as a classification dataset
if args.GAN == "cGAN":
    Filename_GAN = save_models_folder + '/ckpt_{}_niters_{}_nclasses_{}_seed_{}.pth'.format(args.GAN, args.niters_gan, args.cGAN_num_classes, args.seed)
    if not os.path.isfile(Filename_GAN):
        netG = cond_cnn_generator(args.dim_gan, num_classes=args.cGAN_num_classes)
        netD = cond_cnn_discriminator(num_classes=args.cGAN_num_classes)
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

        # Start training
        netG, netD = train_cGAN(images, labels, netG, netD, save_images_folder=save_GANimages_InTrain_folder, save_models_folder = save_models_folder)

        # store model
        torch.save({
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
        }, Filename_GAN)
    else:
        print("Loading pre-trained generator >>>")
        checkpoint = torch.load(Filename_GAN)
        netG = cond_cnn_generator(args.dim_gan, num_classes=args.cGAN_num_classes).to(device)
        netG = nn.DataParallel(netG)
        netG.load_state_dict(checkpoint['netG_state_dict'])

    # function for sampling from a trained GAN
    def fn_sampleGAN_given_label(nfake, label, batch_size):
        label = label*max_label_after_shift - np.abs(min_label_before_shift) #back to original scale
        fake_images, fake_labels = SampcGAN_given_label(netG, label, class_cutoff_points=class_cutoff_points, NFAKE = nfake, batch_size = batch_size)
        fake_labels = (fake_labels + np.abs(min_label_before_shift)) / max_label_after_shift
        return fake_images, fake_labels

    def fn_sampleGAN_given_labels(nfake, labels, batch_size):
        labels = labels*max_label_after_shift - np.abs(min_label_before_shift) #back to original scale
        fake_images, fake_labels = SampcGAN_given_labels(netG, labels, class_cutoff_points=class_cutoff_points, NFAKE = nfake, batch_size = batch_size)
        fake_labels = (fake_labels + np.abs(min_label_before_shift)) / max_label_after_shift
        return fake_images, fake_labels

#----------------------------------------------
# Concitnuous cGAN
if args.GAN == "CcGAN":
    Filename_GAN = save_models_folder + '/ckpt_{}_niters_{}_seed_{}_{}_{}_{}.pth'.format(args.GAN, args.niters_gan, args.seed, args.threshold_type, args.kernel_sigma, args.kappa)

    if not os.path.isfile(Filename_GAN):
        netG = cont_cond_cnn_generator(nz=args.dim_gan, img_size=IMG_SIZE, dim_embed=args.dim_embed).to(device)
        netD = cont_cond_cnn_discriminator(img_size=IMG_SIZE, dim_embed=args.dim_embed).to(device)
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

        # Start training
        netG, netD = train_CcGAN(args.kernel_sigma, args.kappa, images, labels, netG, netD, net_y2h, save_images_folder=save_GANimages_InTrain_folder, save_models_folder = save_models_folder)

        # store model
        torch.save({
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
        }, Filename_GAN)

    else:
        print("Loading pre-trained generator >>>")
        checkpoint = torch.load(Filename_GAN)
        netG = cont_cond_cnn_generator(nz=args.dim_gan, img_size=IMG_SIZE, dim_embed=args.dim_embed).to(device)
        netG = nn.DataParallel(netG)
        netG.load_state_dict(checkpoint['netG_state_dict'])

    def fn_sampleGAN_given_label(nfake, label, batch_size):
        fake_images, fake_labels = SampCcGAN_given_label(netG, net_y2h, label, path=None, NFAKE = nfake, batch_size = batch_size)
        return fake_images, fake_labels

    def fn_sampleGAN_given_labels(nfake, labels, batch_size):
        fake_images, fake_labels = SampCcGAN_given_labels(netG, net_y2h, labels, path=None, NFAKE = nfake, batch_size = batch_size)
        return fake_images, fake_labels

stop = timeit.default_timer()
print("GAN training finished; Time elapses: {}s".format(stop - start))


#######################################################################################
'''                                  Evaluation                                     '''
#######################################################################################
if args.comp_FID:
    # for FID
    PreNetFID = encoder(dim_bottleneck=512).to(device)
    PreNetFID = nn.DataParallel(PreNetFID)
    Filename_PreCNNForEvalGANs = save_models_folder + '/ckpt_AE_epoch_200_seed_2020_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
    PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])

    # Diversity: entropy of predicted races within each eval center
    PreNetDiversity = ResNet34_class(num_classes=5, ngpu = NGPU).to(device) #49 chair types
    Filename_PreCNNForEvalGANs_Diversity = save_models_folder + '/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_20_seed_2020_classify_5_scenes_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_Diversity)
    PreNetDiversity.load_state_dict(checkpoint_PreNet['net_state_dict'])

    # for LS
    PreNetLS = ResNet34_regre(ngpu = NGPU).to(device)
    Filename_PreCNNForEvalGANs = save_models_folder + '/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
    PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])

    #####################
    # generate nfake images
    print("Start sampling {} fake images per label from GAN >>>".format(args.nfake_per_label))
    eval_labels = np.linspace(np.min(raw_labels), np.max(raw_labels), args.num_eval_labels) #not normalized
    eval_labels_norm = (eval_labels + np.abs(min_label_before_shift)) / max_label_after_shift

    for i in range(args.num_eval_labels):
        curr_label = eval_labels_norm[i]
        if i == 0:
            fake_labels_assigned = np.ones(args.nfake_per_label)*curr_label
        else:
            fake_labels_assigned = np.concatenate((fake_labels_assigned, np.ones(args.nfake_per_label)*curr_label))
    fake_images, _ = fn_sampleGAN_given_labels(args.nfake_per_label*args.num_eval_labels, fake_labels_assigned, args.samp_batch_size)
    assert len(fake_images) == args.nfake_per_label*args.num_eval_labels
    assert len(fake_labels_assigned) == args.nfake_per_label*args.num_eval_labels
    print("End sampling! We got {} fake images.".format(len(fake_images)))

    ## dump fake images for evaluation: NIQE
    if args.GAN == "cGAN":
        dump_fake_images_folder = wd + "/dump_fake_data/fake_images_cGAN_nclass_{}_nsamp_{}".format(args.cGAN_num_classes, len(fake_images))
    else:
        if args.kernel_sigma>1e-30:
            dump_fake_images_folder = wd + "/dump_fake_data/fake_images_CcGAN_{}_nsamp_{}".format(args.threshold_type, len(fake_images))
        else:
            dump_fake_images_folder = wd + "/dump_fake_data/fake_images_CcGAN_limit_nsamp_{}".format(len(fake_images))
    for i in tqdm(range(len(fake_images))):
        label_i = fake_labels_assigned[i]*max_label_after_shift - np.abs(min_label_before_shift)
        filename_i = dump_fake_images_folder + "/{}_{}.png".format(i, label_i)
        os.makedirs(os.path.dirname(filename_i), exist_ok=True)
        image_i = fake_images[i]
        image_i = ((image_i*0.5+0.5)*255.0).astype(np.uint8)
        image_i_pil = Image.fromarray(image_i.transpose(1,2,0))
        image_i_pil.save(filename_i)
    #end for i

    #####################
    # normalize real images and labels
    real_images = (raw_images/255.0-0.5)/0.5
    real_labels = (raw_labels + np.abs(min_label_before_shift)) / max_label_after_shift
    nfake_all = len(fake_images)
    nreal_all = len(real_images)

    #####################
    # Evaluate FID within a sliding window with a radius R on the label's range (not normalized range, i.e., [min_label,max_label]). The center of the sliding window locate on [min_label+R,...,max_label-R].
    center_start = np.min(raw_labels)+args.FID_radius
    center_stop = np.max(raw_labels)-args.FID_radius
    centers_loc = np.linspace(center_start, center_stop, args.FID_num_centers) #not normalized

    # output center locations for computing NIQE
    filename_centers = wd + '/steering_angle_centers_loc_for_NIQE.txt'
    np.savetxt(filename_centers, centers_loc)

    labelscores_over_centers = np.zeros(len(centers_loc)) #label score at each center
    FID_over_centers = np.zeros(len(centers_loc))
    entropies_over_centers = np.zeros(len(centers_loc)) # entropy at each center
    num_realimgs_over_centers = np.zeros(len(centers_loc))
    for i in range(len(centers_loc)):
        center = centers_loc[i]
        interval_start = (center - args.FID_radius + np.abs(min_label_before_shift)) / max_label_after_shift
        interval_stop = (center + args.FID_radius + np.abs(min_label_before_shift)) / max_label_after_shift
        indx_real = np.where((real_labels>=interval_start)*(real_labels<=interval_stop)==True)[0]
        np.random.shuffle(indx_real)
        real_images_curr = real_images[indx_real]
        num_realimgs_over_centers[i] = len(real_images_curr)
        indx_fake = np.where((fake_labels_assigned>=interval_start)*(fake_labels_assigned<=interval_stop)==True)[0]
        np.random.shuffle(indx_fake)
        fake_images_curr = fake_images[indx_fake]
        fake_labels_assigned_curr = fake_labels_assigned[indx_fake]
        ## FID
        FID_over_centers[i] = cal_FID(PreNetFID, real_images_curr, fake_images_curr, batch_size = 500, resize = None)
        ## Entropy of predicted class labels
        predicted_class_labels = predict_class_labels(PreNetDiversity, fake_images_curr, batch_size=500)
        entropies_over_centers[i] = compute_entropy(predicted_class_labels)
        ## Label score
        labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr, fake_labels_assigned_curr, min_label_before_shift=min_label_before_shift, max_label_after_shift=max_label_after_shift, batch_size = 200, resize = None)
        ## print
        print("\n [{}/{}] Center:{}; Real:{}; Fake:{}; FID:{}; LS:{}; ET:{}.".format(i+1, len(centers_loc), center, len(real_images_curr), len(fake_images_curr), FID_over_centers[i], labelscores_over_centers[i], entropies_over_centers[i]))
    # end for i
    # average over all centers
    print("\n {} SFID: {}({}); min/max: {}/{}.".format(args.GAN, np.mean(FID_over_centers), np.std(FID_over_centers), np.min(FID_over_centers), np.max(FID_over_centers)))
    print("\n {} LS over centers: {}({}); min/max: {}/{}.".format(args.GAN, np.mean(labelscores_over_centers), np.std(labelscores_over_centers), np.min(labelscores_over_centers), np.max(labelscores_over_centers)))
    print("\n {} entropy over centers: {}({}); min/max: {}/{}.".format(args.GAN, np.mean(entropies_over_centers), np.std(entropies_over_centers), np.min(entropies_over_centers), np.max(entropies_over_centers)))

    # dump FID versus number of samples (for each center) to npy
    if args.GAN == "cGAN":
        dump_fid_ls_entropy_over_centers_filename = wd + "/cGAN_nclass_{}_fid_ls_entropy_over_centers".format(args.cGAN_num_classes)
    else:
        if args.kernel_sigma>1e-30:
            dump_fid_ls_entropy_over_centers_filename = wd + "/CcGAN_{}_fid_ls_entropy_over_centers".format(args.threshold_type)
        else:
            dump_fid_ls_entropy_over_centers_filename = wd + "/CcGAN_limit_fid_ls_entropy_over_centers"
    np.savez(dump_fid_ls_entropy_over_centers_filename, fids=FID_over_centers, labelscores=labelscores_over_centers, entropies=entropies_over_centers, nrealimgs=num_realimgs_over_centers, centers=centers_loc)

    #####################
    # FID: Evaluate FID on all fake images
    indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
    indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
    FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size = 500, resize = None)
    print("\n {}: FID of {} fake images: {}.".format(args.GAN, nfake_all, FID))

    #####################
    # Overall LS: abs(y_assigned - y_predicted)
    ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, fake_images, fake_labels_assigned, min_label_before_shift=min_label_before_shift, max_label_after_shift=max_label_after_shift, batch_size = 200, resize = None)
    print("\n {}: overall LS of {} fake images: {}({}).".format(args.GAN, nfake_all, ls_mean_overall, ls_std_overall))

#######################################################################################
'''               Visualize fake images of the trained GAN                          '''
#######################################################################################
if args.visualize_fake_images:
    # First, visualize conditional generation
    ## 10 rows; 3 columns (3 samples for each age)
    n_row = 10
    n_col = 3
    displayed_normalized_labels = np.linspace(0.05, 0.95, n_row)
    ### output fake images from a trained GAN
    if args.GAN == 'CcGAN':
        filename_fake_images = save_images_folder + '/{}_{}_sigma_{}_kappa_{}_fake_images_grid_{}x{}.png'.format(args.GAN, args.threshold_type, args.kernel_sigma, args.kappa, n_row, n_col)
    else:
        filename_fake_images = save_images_folder + '/{}_nclass_{}_fake_images_grid_{}x{}.png'.format(args.GAN, args.cGAN_num_classes, n_row, n_col)
    images_show = np.zeros((n_row*n_col, NC, IMG_SIZE, IMG_SIZE))
    for i_row in range(n_row):
        curr_label = displayed_normalized_labels[i_row]
        for j_col in range(n_col):
            curr_image, _ = fn_sampleGAN_given_label(1, curr_label, 1)
            images_show[i_row*n_col+j_col,:,:,:] = curr_image
    images_show = torch.from_numpy(images_show)
    save_image(images_show.data, filename_fake_images, nrow=n_col, normalize=True)


    ### output some real images as baseline
    filename_real_images = save_images_folder + '/real_images_grid_{}x{}.png'.format(n_row, n_col)
    if not os.path.isfile(filename_real_images):
        images_show = np.zeros((n_row*n_col, NC, IMG_SIZE, IMG_SIZE))
        for i_row in range(n_row):
            # generate 3 real images from each interval
            label_lower = displayed_normalized_labels[i_row]*max_label_after_shift-np.abs(min_label_before_shift)-1
            label_upper = displayed_normalized_labels[i_row]*max_label_after_shift-np.abs(min_label_before_shift)+1

            for j_col in range(n_col):
                indx_curr_label = np.where((raw_labels>=label_lower)*(raw_labels<label_upper)==True)[0]
                np.random.shuffle(indx_curr_label)
                indx_curr_label = indx_curr_label[0]
                images_show[i_row*n_col+j_col] = raw_images[indx_curr_label]
        images_show = (images_show/255.0-0.5)/0.5
        images_show = torch.from_numpy(images_show)
        save_image(images_show.data, filename_real_images, nrow=n_col, normalize=True)


    # Second, fix z but increase y; check whether there is a continuous change, only for CcGAN
    if args.GAN == 'CcGAN':
        n_continuous_labels = 10
        normalized_continuous_labels = np.linspace(0.05, 0.95, n_continuous_labels)
        z = torch.randn(1, args.dim_gan, dtype=torch.float).to(device)
        continuous_images_show = torch.zeros(n_continuous_labels, NC, IMG_SIZE, IMG_SIZE, dtype=torch.float)

        netG.eval()
        with torch.no_grad():
            for i in range(n_continuous_labels):
                y = np.ones(1) * normalized_continuous_labels[i]
                y = torch.from_numpy(y).type(torch.float).view(-1,1).to(device)
                fake_image_i = netG(z, net_y2h(y))
                continuous_images_show[i,:,:,:] = fake_image_i.cpu()

        filename_continous_fake_images = save_images_folder + '/{}_{}_sigma_{}_kappa_{}_continuous_fake_images_grid.png'.format(args.GAN, args.threshold_type, args.kernel_sigma, args.kappa)
        save_image(continuous_images_show.data, filename_continous_fake_images, nrow=n_continuous_labels, normalize=True)

        print("Continuous ys: ", (normalized_continuous_labels*max_label_after_shift-np.abs(min_label_before_shift)))

print("\n===================================================================================================")
