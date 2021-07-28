print("\n===================================================================================================")

import argparse
import copy
import gc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')
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
from trainer import train_cgan, sample_cgan_given_labels
from eval_metrics import cal_FID, cal_labelscore, inception_score



#######################################################################################
'''                                   Settings                                      '''
#######################################################################################


#-------------------------------
# seeds
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
cudnn.benchmark = False
np.random.seed(args.seed)

#-------------------------------
# output folders
path_to_output = os.path.join(wd, "output/output_arch_{}_dimz_{}_lrg_{}_lrd_{}".format(args.GAN_arch, args.dim_z, args.lr_g, args.lr_d))
os.makedirs(path_to_output, exist_ok=True)
save_models_folder = os.path.join(path_to_output, 'saved_models')
os.makedirs(save_models_folder, exist_ok=True)
save_images_folder = os.path.join(path_to_output, 'saved_images')
os.makedirs(save_images_folder, exist_ok=True)


#######################################################################################
'''                                    Data loader                                 '''
#######################################################################################
# data loader
data_filename = args.data_path + '/UTKFace_{}x{}.h5'.format(args.img_size, args.img_size)
hf = h5py.File(data_filename, 'r')
labels = hf['labels'][:]
labels = labels.astype(float)
images = hf['images'][:]
hf.close()

# subset of UTKFace
selected_labels = np.arange(args.min_label, args.max_label+1)
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

# plot the histogram of unnormalized labels
hist_filename = wd + "/histogram_after_replica_unnormalized_age_" + str(args.img_size) + 'x' + str(args.img_size)
num_bins = len(list(set(labels)))
plt.figure()
plt.hist(labels, num_bins, facecolor='blue', density=False)
plt.savefig(hist_filename)

## normalize labels
labels = labels / args.max_label




#######################################################################################
'''                                    GAN training                                 '''
#######################################################################################

start = timeit.default_timer()
print("\n Begin Training InfoGAN with %s:" % args.GAN_arch)

Filename_GAN = save_models_folder + '/ckpt_{}_dimz_{}_niters_{}_seed_{}.pth'.format(args.GAN_arch, args.dim_z, args.niters_gan, args.seed)
print(Filename_GAN)

save_images_in_train_folder = save_images_folder + '/images_in_train'
os.makedirs(save_images_in_train_folder, exist_ok=True)

if not os.path.isfile(Filename_GAN):
    
        netG = SNGAN_Generator(dim_z=args.dim_z, dim_c=1).cuda()
        netD = SNGAN_Discriminator(dim_c=1).cuda()
        netG = nn.DataParallel(netG)
        netD = nn.DataParallel(netD)

        # Start training
        netG, netD = train_cgan(images, labels, netG, netD, save_images_folder=save_images_in_train_folder, save_models_folder = save_models_folder)

        # store model
        torch.save({
            'netG_state_dict': netG.state_dict(),
        }, Filename_GAN)
else:
    print("Loading pre-trained generator >>>")
    checkpoint = torch.load(Filename_GAN)
    netG = SNGAN_Generator(dim_z=args.dim_z, dim_c=1).cuda()
    netG = nn.DataParallel(netG)
    netG.load_state_dict(checkpoint['netG_state_dict'])

def fn_sampleGAN_given_labels(labels, batch_size):
    labels = labels*args.max_label
    fake_images, fake_labels = sample_cgan_given_labels(netG, given_labels=labels, batch_size = batch_size)
    fake_labels = fake_labels / args.max_label
    return fake_images, fake_labels

stop = timeit.default_timer()
print("cGAN-concat training finished; Time elapses: {}s".format(stop - start))




#######################################################################################
'''                                  Evaluation                                     '''
#######################################################################################
if args.comp_FID:
    #for FID
    PreNetFID = encoder(dim_bottleneck=512).cuda()
    PreNetFID = nn.DataParallel(PreNetFID)
    Filename_PreCNNForEvalGANs = os.path.join(args.eval_ckpt_path, 'ckpt_AE_epoch_200_seed_2020_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
    PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])

    # Diversity: entropy of predicted races within each eval center
    PreNetDiversity = ResNet34_class(num_classes=5, ngpu = torch.cuda.device_count()).cuda() #5 races
    Filename_PreCNNForEvalGANs_Diversity = os.path.join(args.eval_ckpt_path, 'ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_5_races_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_Diversity)
    PreNetDiversity.load_state_dict(checkpoint_PreNet['net_state_dict'])

    # for LS
    PreNetLS = ResNet34_regre(ngpu = torch.cuda.device_count()).cuda()
    Filename_PreCNNForEvalGANs_LS = os.path.join(args.eval_ckpt_path, 'ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_LS)
    PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])


    #####################
    # generate nfake images
    print("Start sampling {} fake images per label from GAN >>>".format(args.nfake_per_label))

    eval_labels_norm = np.arange(1, args.max_label+1) / args.max_label # normalized labels for evaluation
    num_eval_labels = len(eval_labels_norm)

    for i in tqdm(range(num_eval_labels)):
        curr_label = eval_labels_norm[i]
        curr_fake_images, curr_fake_labels = fn_sampleGAN_given_labels(curr_label*np.ones([args.nfake_per_label,1]), args.samp_batch_size)
        if i == 0:
            fake_images = curr_fake_images
            fake_labels_assigned = curr_fake_labels.reshape(-1)
        else:
            fake_images = np.concatenate((fake_images, curr_fake_images), axis=0)
            fake_labels_assigned = np.concatenate((fake_labels_assigned, curr_fake_labels.reshape(-1)))
    assert len(fake_images) == args.nfake_per_label*num_eval_labels
    assert len(fake_labels_assigned) == args.nfake_per_label*num_eval_labels


    ## dump fake images for evaluation: NIQE
    if args.dump_fake_for_NIQE:
        print("\n Dumping fake images for NIQE...")
        dump_fake_images_folder = save_images_folder + '/fake_images_for_NIQE_nfake_{}'.format(len(fake_images))
        os.makedirs(dump_fake_images_folder, exist_ok=True)
        for i in tqdm(range(len(fake_images))):
            label_i = int(fake_labels_assigned[i]*args.max_label)
            filename_i = dump_fake_images_folder + "/{}_{}.png".format(i, label_i)
            os.makedirs(os.path.dirname(filename_i), exist_ok=True)
            image_i = fake_images[i]
            # image_i = ((image_i*0.5+0.5)*255.0).astype(np.uint8)
            image_i_pil = Image.fromarray(image_i.transpose(1,2,0))
            image_i_pil.save(filename_i)
        #end for i
    
    print("End sampling!")
    print("\n We got {} fake images.".format(len(fake_images)))


    #####################
    # normalize images and labels
    real_images = (raw_images/255.0-0.5)/0.5
    real_labels = raw_labels/args.max_label
    nfake_all = len(fake_images)
    nreal_all = len(real_images)
    fake_images = (fake_images/255.0-0.5)/0.5


    #####################
    # Evaluate FID within a sliding window with a radius R on the label's range (i.e., [1,max_label]). The center of the sliding window locate on [R+1,2,3,...,max_label-R].
    center_start = 1+args.FID_radius
    center_stop = args.max_label-args.FID_radius
    centers_loc = np.arange(center_start, center_stop+1)
    FID_over_centers = np.zeros(len(centers_loc))
    entropies_over_centers = np.zeros(len(centers_loc)) # entropy at each center
    labelscores_over_centers = np.zeros(len(centers_loc)) #label score at each center
    num_realimgs_over_centers = np.zeros(len(centers_loc))
    for i in range(len(centers_loc)):
        center = centers_loc[i]
        interval_start = (center - args.FID_radius)/args.max_label
        interval_stop = (center + args.FID_radius)/args.max_label
        indx_real = np.where((real_labels>=interval_start)*(real_labels<=interval_stop)==True)[0]
        np.random.shuffle(indx_real)
        real_images_curr = real_images[indx_real]
        num_realimgs_over_centers[i] = len(real_images_curr)
        indx_fake = np.where((fake_labels_assigned>=interval_start)*(fake_labels_assigned<=interval_stop)==True)[0]
        np.random.shuffle(indx_fake)
        fake_images_curr = fake_images[indx_fake]
        fake_labels_assigned_curr = fake_labels_assigned[indx_fake]
        # FID
        FID_over_centers[i] = cal_FID(PreNetFID, real_images_curr, fake_images_curr, batch_size = 200, resize = None)
        # Entropy of predicted class labels
        predicted_class_labels = predict_class_labels(PreNetDiversity, fake_images_curr, batch_size = 200, num_workers=args.num_workers)
        entropies_over_centers[i] = compute_entropy(predicted_class_labels)
        # Label score
        labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr, fake_labels_assigned_curr, min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size = 200, resize = None, num_workers=args.num_workers)

        print("\r Center:{}; Real:{}; Fake:{}; FID:{}; LS:{}; ET:{}.".format(center, len(real_images_curr), len(fake_images_curr), FID_over_centers[i], labelscores_over_centers[i], entropies_over_centers[i]))

    # average over all centers
    print("\n {} SFID: {}({}); min/max: {}/{}.".format(args.GAN_arch, np.mean(FID_over_centers), np.std(FID_over_centers), np.min(FID_over_centers), np.max(FID_over_centers)))
    print("\n {} LS over centers: {}({}); min/max: {}/{}.".format(args.GAN_arch, np.mean(labelscores_over_centers), np.std(labelscores_over_centers), np.min(labelscores_over_centers), np.max(labelscores_over_centers)))
    print("\n {} entropy over centers: {}({}); min/max: {}/{}.".format(args.GAN_arch, np.mean(entropies_over_centers), np.std(entropies_over_centers), np.min(entropies_over_centers), np.max(entropies_over_centers)))

    # dump FID versus number of samples (for each center) to npy
    dump_fid_ls_entropy_over_centers_filename = os.path.join(path_to_output, 'fid_ls_entropy_over_centers')
    np.savez(dump_fid_ls_entropy_over_centers_filename, fids=FID_over_centers, labelscores=labelscores_over_centers, entropies=entropies_over_centers, nrealimgs=num_realimgs_over_centers, centers=centers_loc)

    #####################
    # Overall LS: abs(y_assigned - y_predicted)
    ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, fake_images, fake_labels_assigned, min_label_before_shift=0, max_label_after_shift=args.max_label, batch_size = 200, resize = None, num_workers=args.num_workers)
    print("\n {}: overall LS of {} fake images: {}({}).".format(args.GAN_arch, nfake_all, ls_mean_overall, ls_std_overall))

    #####################
    # FID: Evaluate FID on all fake images
    indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
    indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
    FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size = 200, resize = None)
    print("\n {}: FID of {} fake images: {}.".format(args.GAN_arch, nfake_all, FID))

    #####################
    # IS: Evaluate IS on all fake images
    indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
    IS, IS_std = inception_score(imgs=fake_images[indx_shuffle_fake], num_classes=5, net=PreNetDiversity, cuda=True, batch_size = 200, splits=10, normalize_img=False)
    print("\n {}: IS of {} fake images: {}({}).".format(args.GAN_arch, nfake_all, IS, IS_std))

    #####################
    # Dump evaluation results
    eval_results_logging_fullpath = os.path.join(path_to_output, 'eval_results_{}.txt'.format(args.GAN_arch))
    if not os.path.isfile(eval_results_logging_fullpath):
        eval_results_logging_file = open(eval_results_logging_fullpath, "w")
        eval_results_logging_file.close()
    with open(eval_results_logging_fullpath, 'a') as eval_results_logging_file:
        eval_results_logging_file.write("\n===================================================================================================")
        eval_results_logging_file.write("\n Radius: {}.  \n".format(args.FID_radius))
        print(args, file=eval_results_logging_file)
        eval_results_logging_file.write("\n SFID: {}({}).".format(np.mean(FID_over_centers), np.std(FID_over_centers)))
        eval_results_logging_file.write("\n LS: {}({}).".format(np.mean(labelscores_over_centers), np.std(labelscores_over_centers)))
        eval_results_logging_file.write("\n Diversity: {}({}).".format(np.mean(entropies_over_centers), np.std(entropies_over_centers)))



#######################################################################################
'''               Visualize fake images of the trained GAN                          '''
#######################################################################################
if args.visualize_fake_images:

    # First, visualize conditional generation; vertical grid
    ## 10 rows; 3 columns (3 samples for each age)
    n_row = 10
    n_col = 3
    displayed_labels = (np.linspace(0.05, 0.95, n_row)*args.max_label).astype(np.int)
    # displayed_labels = np.array([3,9,15,21,27,33,39,45,51,57])
    # displayed_labels = np.array([4,10,16,22,28,34,40,46,52,58])
    displayed_normalized_labels = displayed_labels/args.max_label
    
    ### output fake images from a trained GAN
    filename_fake_images = os.path.join(save_images_folder, 'fake_images_grid_{}x{}.png').format(n_row, n_col)
    fake_labels_assigned = []
    for tmp_i in range(len(displayed_normalized_labels)):
        curr_label = displayed_normalized_labels[tmp_i]
        fake_labels_assigned.append(np.ones(shape=[n_col, 1])*curr_label)
    fake_labels_assigned = np.concatenate(fake_labels_assigned, axis=0)
    images_show, _ = fn_sampleGAN_given_labels(fake_labels_assigned, args.samp_batch_size)
    images_show = (images_show/255.0-0.5)/0.5
    images_show = torch.from_numpy(images_show)
    save_image(images_show.data, filename_fake_images, nrow=n_col, normalize=True)

    #----------------------------------------------------------------
    ### output some real images as baseline
    filename_real_images = save_images_folder + '/real_images_grid_{}x{}.png'.format(n_row, n_col)
    if not os.path.isfile(filename_real_images):
        images_show = np.zeros((n_row*n_col, args.num_channels, args.img_size, args.img_size))
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


print("\n===================================================================================================")