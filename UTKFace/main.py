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
from Train_CcGAN_limit import train_CcGAN_limit
from eval_metrics import cal_FID, cal_labelscore


#######################################################################################
'''                                   Settings                                      '''
#######################################################################################

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
save_traincurves_folder = wd + '/output/training_loss_fig'
os.makedirs(save_traincurves_folder, exist_ok=True)


#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
# data loader
data_filename = args.data_path + '/UTKFace_{}x{}.h5'.format(IMG_SIZE, IMG_SIZE)
hf = h5py.File(data_filename, 'r')
labels = hf['labels'][:]
labels = labels.astype(np.float)
images = hf['images'][:]
hf.close()

# subset of UTKFace
selected_labels = np.arange(args.min_age, args.max_age+1)
for i in range(len(selected_labels)):
    curr_label = selected_labels[i]
    index_curr_label = np.where(labels==curr_label)[0]
    if i == 0:
        images_subset = images[index_curr_label]
        labels_subset = labels[index_curr_label]
    else:
        images_subset = np.concatenate((images_subset, images[index_curr_label]), axis=0)
        labels_subset = np.concatenate((labels_subset, labels[index_curr_label]))
# for i
images = images_subset
labels = labels_subset
del images_subset, labels_subset; gc.collect()

raw_images = copy.deepcopy(images)
raw_labels = copy.deepcopy(labels)

### show some real  images
if args.show_real_imgs:
    unique_labels_show = sorted(list(set(labels)))
    nrow = len(unique_labels_show); ncol = 10
    images_show = np.zeros((nrow*ncol, images.shape[1], images.shape[2], images.shape[3]))
    for i in range(nrow):
        curr_label = unique_labels_show[i]
        indx_curr_label = np.where(labels==curr_label)[0]
        np.random.shuffle(indx_curr_label)
        indx_curr_label = indx_curr_label[0:ncol]
        for j in range(ncol):
            images_show[i*ncol+j,:,:,:] = images[indx_curr_label[j]]
    print(images_show.shape)
    images_show = (images_show/255.0-0.5)/0.5
    images_show = torch.from_numpy(images_show)
    save_image(images_show.data, save_images_folder +'/real_images_grid_{}x{}.png'.format(nrow, ncol), nrow=ncol, normalize=True)


# for each age, take no more than args.max_num_img_per_label images
image_num_threshold = args.max_num_img_per_label
print("\n Original set has {} images; For each age, take no more than {} images>>>".format(len(images), image_num_threshold))
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
print("{} images left.".format(len(images)))


hist_filename = wd + "/histogram_before_replica_unnormalized_age_" + str(args.img_size) + 'x' + str(args.img_size)
num_bins = len(list(set(labels)))
plt.figure()
plt.hist(labels, num_bins, facecolor='blue', density=False)
plt.savefig(hist_filename)


## replicate minority samples to alleviate the imbalance
max_num_img_per_label_after_replica = np.min([args.max_num_img_per_label_after_replica, args.max_num_img_per_label])
if max_num_img_per_label_after_replica>1:
    unique_labels_replica = np.sort(np.array(list(set(labels))))
    num_labels_replicated = 0
    print("Start replicating monority samples >>>")
    for i in tqdm(range(len(unique_labels_replica))):
        # print((i, num_labels_replicated))
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
# hist_filename = wd + "/histogram_replica_age_" + str(args.img_size) + 'x' + str(args.img_size)
# num_bins = len(list(set(labels)))
# plt.figure()
# plt.hist(labels, num_bins, facecolor='blue', density=False)
# plt.savefig(hist_filename)

# plot the histogram of unnormalized labels
hist_filename = wd + "/histogram_after_replica_unnormalized_age_" + str(args.img_size) + 'x' + str(args.img_size)
num_bins = len(list(set(labels)))
plt.figure()
plt.hist(labels, num_bins, facecolor='blue', density=False)
plt.savefig(hist_filename)


# normalize labels
print("\n Range of unnormalized labels: ({},{})".format(np.min(labels), np.max(labels)))
max_label = np.max(labels)
assert max_label == args.max_age
if args.GAN == "cGAN": #treated as classification; convert ages to class labels
    unique_labels = np.sort(np.array(list(set(labels))))
    num_unique_labels = len(unique_labels)
    print("{} unique labels are split into {} classes".format(num_unique_labels, args.cGAN_num_classes))

    ## convert ages to class labels and vice versa
    ### step 1: prepare two dictionaries
    label2class = dict()
    class2label = dict()
    num_labels_per_class = num_unique_labels//args.cGAN_num_classes
    class_cutoff_points = [unique_labels[0]] #the cutoff points on [min_label, max_label] to determine classes; each interval is a class
    curr_class = 0
    for i in range(num_unique_labels):
        label2class[unique_labels[i]]=curr_class
        if (i+1)%num_labels_per_class==0 and (curr_class+1)!=args.cGAN_num_classes:
            curr_class += 1
            class_cutoff_points.append(unique_labels[i+1])
    class_cutoff_points.append(unique_labels[-1])
    assert len(class_cutoff_points)-1 == args.cGAN_num_classes

    ### the cell label of each interval equals to the average of the two end points
    for i in range(args.cGAN_num_classes):
        class2label[i] = (class_cutoff_points[i]+class_cutoff_points[i+1])/2

    ### step 2: convert ages to class labels
    labels_new = -1*np.ones(len(labels))
    for i in range(len(labels)):
        labels_new[i] = label2class[labels[i]]
    assert np.sum(labels_new<0)==0
    labels = labels_new
    del labels_new; gc.collect()
    unique_labels = np.sort(np.array(list(set(labels)))).astype(np.int)
else:
    labels /= args.max_age #normalize to [0,1]
    # labels /= max_label #normalize to [0,1]

    # plot the histogram of normalized labels
    hist_filename = wd + "/histogram_normalized_age_" + str(args.img_size) + 'x' + str(args.img_size)
    num_bins = len(list(set(labels)))
    plt.figure()
    plt.hist(labels, num_bins, facecolor='blue', density=False)
    plt.savefig(hist_filename)

    print("\n Range of normalized labels: ({},{})".format(np.min(labels), np.max(labels)))

    unique_labels_norm = np.sort(np.array(list(set(labels))))

    if args.kernel_sigma<0:
        std_label = np.std(labels)
        args.kernel_sigma =1.06*std_label*(len(labels))**(-1/5)
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
# if args.GAN

#######################################################################################
'''                                    GAN training                                 '''
#######################################################################################
print("{}, Sigma is {}, Kappa is {}".format(args.threshold_type, args.kernel_sigma, args.kappa))

if args.GAN == 'CcGAN':
    save_GANimages_InTrain_folder = save_images_folder + '/{}_{}_{}_{}_InTrain'.format(args.GAN, args.threshold_type, args.kernel_sigma, args.kappa)
else:
    save_GANimages_InTrain_folder = save_images_folder + '/{}_InTrain'.format(args.GAN)
os.makedirs(save_GANimages_InTrain_folder, exist_ok=True)

start = timeit.default_timer()
print("\n Begin Training %s:" % args.GAN)
#----------------------------------------------
# cGAN: treated as a classification dataset
if args.GAN == "cGAN":
    Filename_GAN = save_models_folder + '/ckpt_{}_niters_{}_nclass_{}_seed_{}.pth'.format(args.GAN, args.niters_gan, args.cGAN_num_classes, args.seed)
    print(Filename_GAN)
    if not os.path.isfile(Filename_GAN):
        print("There are {} unique labels".format(len(unique_labels)))

        netG = cond_cnn_generator(nz=args.dim_gan, num_classes=args.cGAN_num_classes)
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
        fake_labels = np.ones(nfake) * label #normalized labels
        label = int(label * max_label) #back to original scale
        fake_images, _ = SampcGAN_given_label(netG, label, class_cutoff_points=class_cutoff_points, NFAKE = nfake, batch_size = batch_size)
        return fake_images, fake_labels

#----------------------------------------------
# Concitnuous cGAN
elif args.GAN == "CcGAN":
    Filename_GAN = save_models_folder + '/ckpt_{}_niters_{}_seed_{}_{}_{}_{}.pth'.format(args.GAN, args.niters_gan, args.seed, args.threshold_type, args.kernel_sigma, args.kappa)
    print(Filename_GAN)

    if not os.path.isfile(Filename_GAN):
        netG = cont_cond_cnn_generator(nz=args.dim_gan)
        netD = cont_cond_cnn_discriminator()
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

        # Start training
        if args.kernel_sigma>1e-30:
            netG, netD = train_CcGAN(args.kernel_sigma, args.kappa, images, labels, netG, netD, save_images_folder=save_GANimages_InTrain_folder, save_models_folder = save_models_folder)
        else:
            print("\n Limiting mode...")
            netG, netD = train_CcGAN_limit(images, labels, netG, netD, save_images_folder=save_GANimages_InTrain_folder, save_models_folder = save_models_folder)

        # store model
        torch.save({
            'netG_state_dict': netG.state_dict(),
            'netD_state_dict': netD.state_dict(),
        }, Filename_GAN)

    else:
        print("Loading pre-trained generator >>>")
        checkpoint = torch.load(Filename_GAN)
        netG = cont_cond_cnn_generator(args.dim_gan).to(device)
        netG = nn.DataParallel(netG)
        netG.load_state_dict(checkpoint['netG_state_dict'])

    def fn_sampleGAN_given_label(nfake, label, batch_size):
        fake_images, fake_labels = SampCcGAN_given_label(netG, label, path=None, NFAKE = nfake, batch_size = batch_size)
        return fake_images, fake_labels

stop = timeit.default_timer()
print("GAN training finished; Time elapses: {}s".format(stop - start))



#######################################################################################
'''                                  Evaluation                                     '''
#######################################################################################
if args.comp_FID:

    #for FID
    PreNetFID = encoder(dim_bottleneck=512).to(device)
    PreNetFID = nn.DataParallel(PreNetFID)
    Filename_PreCNNForEvalGANs = save_models_folder + '/ckpt_AE_epoch_200_seed_2020_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
    PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])

    # Diversity: entropy of predicted races within each eval center
    PreNetDiversity = ResNet34_class(num_classes=5, ngpu = NGPU).to(device) #5 races
    Filename_PreCNNForEvalGANs_Diversity = save_models_folder + '/ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_5_races_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_Diversity)
    PreNetDiversity.load_state_dict(checkpoint_PreNet['net_state_dict'])

    # for LS
    PreNetLS = ResNet34_regre(ngpu = NGPU).to(device)
    Filename_PreCNNForEvalGANs_LS = save_models_folder + '/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_LS)
    PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])

    #####################
    # generate nfake images
    print("Start sampling {} fake images per label from GAN >>>".format(args.nfake_per_label))

    eval_labels_norm = np.arange(1, max_label+1) / max_label # normalized labels for evaluation
    num_eval_labels = len(eval_labels_norm)

    ## wo dump
    for i in tqdm(range(num_eval_labels)):
        curr_label = eval_labels_norm[i]
        curr_fake_images, curr_fake_labels = fn_sampleGAN_given_label(args.nfake_per_label, curr_label, args.samp_batch_size)

        if i == 0:
            fake_images = curr_fake_images
            fake_labels_assigned = curr_fake_labels.reshape(-1)
        else:
            fake_images = np.concatenate((fake_images, curr_fake_images), axis=0)
            fake_labels_assigned = np.concatenate((fake_labels_assigned, curr_fake_labels.reshape(-1)))
    assert len(fake_images) == args.nfake_per_label*num_eval_labels
    assert len(fake_labels_assigned) == args.nfake_per_label*num_eval_labels


    ## dump fake images for evaluation: NIQE
    if args.GAN == "cGAN":
        dump_fake_images_folder = wd + "/dump_fake_data/fake_images_cGAN_nclass_{}_nsamp_{}".format(args.cGAN_num_classes, len(fake_images))
    else:
        if args.kernel_sigma>1e-30:
            dump_fake_images_folder = wd + "/dump_fake_data/fake_images_CcGAN_{}_nsamp_{}".format(args.threshold_type, len(fake_images))
        else:
            dump_fake_images_folder = wd + "/dump_fake_data/fake_images_CcGAN_limit_nsamp_{}".format(len(fake_images))
    for i in tqdm(range(len(fake_images))):
        label_i = int(fake_labels_assigned[i]*max_label)
        filename_i = dump_fake_images_folder + "/{}_{}.png".format(i, label_i)
        os.makedirs(os.path.dirname(filename_i), exist_ok=True)
        image_i = fake_images[i]
        image_i = ((image_i*0.5+0.5)*255.0).astype(np.uint8)
        image_i_pil = Image.fromarray(image_i.transpose(1,2,0))
        image_i_pil.save(filename_i)
    #end for i

    print("End sampling!")
    print("\n We got {} fake images.".format(len(fake_images)))

    #####################
    # normalize real images and labels
    real_images = (raw_images/255.0-0.5)/0.5
    real_labels = raw_labels/max_label
    nfake_all = len(fake_images)
    nreal_all = len(real_images)

    #####################
    # Evaluate FID within a sliding window with a radius R on the label's range (i.e., [1,max_label]). The center of the sliding window locate on [R+1,2,3,...,max_label-R].
    center_start = 1+args.FID_radius
    center_stop = max_label-args.FID_radius
    centers_loc = np.arange(center_start, center_stop+1)
    FID_over_centers = np.zeros(len(centers_loc))
    entropies_over_centers = np.zeros(len(centers_loc)) # entropy at each center
    labelscores_over_centers = np.zeros(len(centers_loc)) #label score at each center
    num_realimgs_over_centers = np.zeros(len(centers_loc))
    for i in range(len(centers_loc)):
        center = centers_loc[i]
        interval_start = (center - args.FID_radius)/max_label
        interval_stop = (center + args.FID_radius)/max_label
        indx_real = np.where((real_labels>=interval_start)*(real_labels<=interval_stop)==True)[0]
        np.random.shuffle(indx_real)
        real_images_curr = real_images[indx_real]
        num_realimgs_over_centers[i] = len(real_images_curr)
        indx_fake = np.where((fake_labels_assigned>=interval_start)*(fake_labels_assigned<=interval_stop)==True)[0]
        np.random.shuffle(indx_fake)
        fake_images_curr = fake_images[indx_fake]
        fake_labels_assigned_curr = fake_labels_assigned[indx_fake]
        # FID
        FID_over_centers[i] = cal_FID(PreNetFID, real_images_curr, fake_images_curr, batch_size = 500, resize = None)
        # Entropy of predicted class labels
        predicted_class_labels = predict_class_labels(PreNetDiversity, fake_images_curr, batch_size=500)
        entropies_over_centers[i] = compute_entropy(predicted_class_labels)
        # Label score
        labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr, fake_labels_assigned_curr, min_label_before_shift=0, max_label_after_shift=args.max_age, batch_size = 500, resize = None)

        print("\r Center:{}; Real:{}; Fake:{}; FID:{}; LS:{}; ET:{}.".format(center, len(real_images_curr), len(fake_images_curr), FID_over_centers[i], labelscores_over_centers[i], entropies_over_centers[i]))

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
    ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, fake_images, fake_labels_assigned, min_label_before_shift=0, max_label_after_shift=args.max_age, batch_size = 500, resize = None)
    print("\n {}: overall LS of {} fake images: {}({}).".format(args.GAN, nfake_all, ls_mean_overall, ls_std_overall))



#######################################################################################
'''               Visualize fake images of the trained GAN                          '''
#######################################################################################
if args.visualize_fake_images:
    # First, visualize conditional generation
    ## 10 rows; 3 columns (3 samples for each age)
    n_row = 10
    n_col = 3
    displayed_labels = (np.linspace(0.05, 0.95, n_row)*max_label).astype(np.int)
    # displayed_labels = np.array([3,9,15,21,27,33,39,45,51,57])
    # displayed_labels = np.array([4,10,16,22,28,34,40,46,52,58])
    displayed_normalized_labels = displayed_labels/max_label
    ### output fake images from a trained GAN
    if args.GAN == 'CcGAN':
        filename_fake_images = save_images_folder + '/{}_{}_sigma_{}_kappa_{}_fake_images_grid_{}x{}.png'.format(args.GAN, args.threshold_type, args.kernel_sigma, args.kappa, n_row, n_col)
    else:
        filename_fake_images = save_images_folder + '/{}_nclass_{}_fake_images_grid_{}x{}.png'.format(args.GAN, args.cGAN_num_classes, n_row, n_col)
    images_show = np.zeros((n_row*n_col, images.shape[1], images.shape[2], images.shape[3]))
    for i_row in range(n_row):
        curr_label = displayed_normalized_labels[i_row]
        for j_col in range(n_col):
            curr_image, _ = fn_sampleGAN_given_label(1, curr_label, 1)
            images_show[i_row*n_col+j_col,:,:,:] = curr_image
    images_show = torch.from_numpy(images_show)
    save_image(images_show.data, filename_fake_images, nrow=n_col, normalize=True)
    print("displayed_labels: ", displayed_labels)

    #----------------------------------------------------------------
    ### output some real images as baseline
    filename_real_images = save_images_folder + '/real_images_grid_{}x{}.png'.format(n_row, n_col)
    if not os.path.isfile(filename_real_images):
        images_show = np.zeros((n_row*n_col, NC, IMG_SIZE, IMG_SIZE))
        for i_row in range(n_row):
            curr_label = displayed_labels[i_row]
            for j_col in range(n_col):
                indx_curr_label = np.where(raw_labels==curr_label)[0]
                np.random.shuffle(indx_curr_label)
                indx_curr_label = indx_curr_label[0]
                images_show[i_row*n_col+j_col] = raw_images[indx_curr_label]
        images_show = (images_show/255.0-0.5)/0.5
        images_show = torch.from_numpy(images_show)
        save_image(images_show.data, filename_real_images, nrow=n_col, normalize=True)

    # Second, fix z but increase y; check whether there is a continuous change, only for CcGAN
    if args.GAN == 'CcGAN':
        # n_continuous_labels = 10
        # normalized_continuous_labels = np.linspace(0.05, 0.95, n_continuous_labels)
        normalized_continuous_labels = displayed_normalized_labels; n_continuous_labels=len(normalized_continuous_labels)
        z = torch.randn(1, args.dim_gan, dtype=torch.float).to(device)
        continuous_images_show = torch.zeros(n_continuous_labels, NC, IMG_SIZE, IMG_SIZE, dtype=torch.float)

        netG.eval()
        with torch.no_grad():
            for i in range(n_continuous_labels):
                y = np.ones(1) * normalized_continuous_labels[i]
                y = torch.from_numpy(y).type(torch.float).view(-1,1).to(device)
                fake_image_i = netG(z, y)
                continuous_images_show[i,:,:,:] = fake_image_i.cpu()

        filename_continous_fake_images = save_images_folder + '/{}_{}_sigma_{}_kappa_{}_continuous_fake_images_grid.png'.format(args.GAN, args.threshold_type, args.kernel_sigma, args.kappa)
        save_image(continuous_images_show.data, filename_continous_fake_images, nrow=n_continuous_labels, normalize=True)

        print("Continuous ys: ", (normalized_continuous_labels*max_label).astype(np.int))
