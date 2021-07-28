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
from trainer import *
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
data_filename = args.data_path + '/Cell200_{}x{}.h5'.format(IMG_SIZE, IMG_SIZE)
hf = h5py.File(data_filename, 'r')
counts = hf['CellCounts'][:]
counts = counts.astype(float)
images = hf['IMGs_grey'][:]
hf.close()

raw_images = copy.deepcopy(images)
raw_counts = copy.deepcopy(counts)

##############
### show some real  images
if args.show_real_imgs:
    unique_counts_show = sorted(list(set(counts)))
    nrow = len(unique_counts_show); ncol = 10
    images_show = np.zeros((nrow*ncol, images.shape[1], images.shape[2], images.shape[3]))
    for i in range(nrow):
        curr_label = unique_counts_show[i]
        indx_curr_label = np.where(counts==curr_label)[0][0:ncol]
        for j in range(ncol):
            images_show[i*ncol+j,:,:,:] = images[indx_curr_label[j]]
    print(images_show.shape)
    images_show = (images_show/255.0-0.5)/0.5
    images_show = torch.from_numpy(images_show)
    save_image(images_show.data, save_images_folder +'/real_images_grid_{}x{}.png'.format(nrow, ncol), nrow=ncol, normalize=True)


##############
# images for training GAN
# for each cell count select n_imgs_per_cellcount images
n_imgs_per_cellcount = args.num_imgs_per_count
selected_cellcounts = np.arange(args.start_count, args.end_count+1, args.stepsize_count)
n_unique_cellcount = len(selected_cellcounts)

images_subset = np.zeros((n_imgs_per_cellcount*n_unique_cellcount, NC, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
counts_subset = np.zeros(n_imgs_per_cellcount*n_unique_cellcount)
for i in range(n_unique_cellcount):
    curr_cellcount = selected_cellcounts[i]
    index_curr_cellcount = np.where(counts==curr_cellcount)[0]

    if i == 0:
        images_subset = images[index_curr_cellcount[0:n_imgs_per_cellcount]]
        counts_subset = counts[index_curr_cellcount[0:n_imgs_per_cellcount]]
    else:
        images_subset = np.concatenate((images_subset, images[index_curr_cellcount[0:n_imgs_per_cellcount]]), axis=0)
        counts_subset = np.concatenate((counts_subset, counts[index_curr_cellcount[0:n_imgs_per_cellcount]]))
# for i
images = images_subset
counts = counts_subset
del images_subset, counts_subset; gc.collect()

print("Number of images: %d" % len(images))

unique_counts = np.sort(np.array(list(set(counts)))).astype(np.int)
counts /= args.end_count # normalize to [0,1]




#######################################################################################
'''                                    GAN training                                 '''
#######################################################################################

save_GANimages_InTrain_folder = save_images_folder + '/{}_InTrain'.format(args.GAN)
os.makedirs(save_GANimages_InTrain_folder, exist_ok=True)

start = timeit.default_timer()
print("\n Begin Training %s:" % args.GAN)
Filename_GAN = save_models_folder + '/ckpt_{}_niters_{}_seed_{}.pth'.format(args.GAN, args.niters_gan, args.seed)
print(Filename_GAN)

if not os.path.isfile(Filename_GAN):
    print("There are {} unique cell counts".format(len(unique_counts)))

    netG = cond_cnn_generator(nz=args.dim_gan, dim_c=1)
    netD = cond_cnn_discriminator(dim_c=1)
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)

    # Start training
    netG, netD = train_cGAN(images, counts, netG, netD, save_images_folder=save_GANimages_InTrain_folder, save_models_folder = save_models_folder)

    # store model
    torch.save({
        'netG_state_dict': netG.state_dict(),
        'netD_state_dict': netD.state_dict(),
    }, Filename_GAN)
else:
    print("Loading pre-trained generator >>>")
    checkpoint = torch.load(Filename_GAN)
    netG = cond_cnn_generator(args.dim_gan, dim_c=1).to(device)
    netG = nn.DataParallel(netG)
    netG.load_state_dict(checkpoint['netG_state_dict'])

# function for sampling from a trained GAN
def fn_sampleGAN_given_label(nfake, count, batch_size):
    fake_counts = np.ones(nfake) * count #normalized count
    fake_images, _ = SampcGAN_given_label(netG, count, NFAKE = nfake, batch_size = batch_size)
    return fake_images, fake_counts

stop = timeit.default_timer()
print("GAN training finished; Time elapses: {}s".format(stop - start))



#######################################################################################
'''                                  Evaluation                                     '''
#######################################################################################
if args.comp_FID:
    #for FID
    PreNetFID = encoder(dim_bottleneck=512).to(device)
    PreNetFID = nn.DataParallel(PreNetFID)
    Filename_PreCNNForEvalGANs = save_models_folder + '/ckpt_AE_epoch_50_seed_2020_CVMode_False.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
    PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])

    # for LS
    PreNetLS = ResNet34_regre(ngpu = NGPU).to(device)
    Filename_PreCNNForEvalGANs = save_models_folder + '/ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_Transformation_True_Cell_200.pth'
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
    PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])

    #####################
    # generate nfake images
    print("Start sampling {} fake images per label from GAN >>>".format(args.nfake_per_label))

    eval_labels_norm = np.arange(args.start_count, args.end_count + 1) / args.end_count
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
    print("End sampling!")
    print("\n We got {} fake images.".format(len(fake_images)))

    ## dump fake images for evaluation: NIQE
    if args.dump_fake_for_NIQE:
        dump_fake_images_folder = wd + "/dump_fake_data/fake_images_cGAN-concat_nsamp_{}".format(len(fake_images))
        for i in tqdm(range(len(fake_images))):
            label_i = round(fake_labels_assigned[i]*args.end_count)
            filename_i = dump_fake_images_folder + "/{}_{}.png".format(i, label_i)
            os.makedirs(os.path.dirname(filename_i), exist_ok=True)
            image_i = fake_images[i]
            image_i = ((image_i*0.5+0.5)*255.0).astype(np.uint8)
            image_i_pil = Image.fromarray(image_i[0])
            image_i_pil.save(filename_i)
        #end for i

    print("End sampling {} fake images per label from GAN >>>".format(args.nfake_per_label))


    #####################
    # normalize real images and labels
    real_images = (raw_images/255.0-0.5)/0.5
    real_labels = raw_counts/args.end_count
    nfake_all = len(fake_images)
    nreal_all = len(real_images)

    #####################
    # Evaluate FID within a sliding window with a radius R on the label's range (i.e., [args.start_count,args.end_count]). The center of the sliding window locate on [R+args.start_count,2,3,...,args.end_count-R].
    center_start = args.start_count+args.FID_radius
    center_stop = args.end_count-args.FID_radius
    centers_loc = np.arange(center_start, center_stop+1)
    FID_over_centers = np.zeros(len(centers_loc))
    labelscores_over_centers = np.zeros(len(centers_loc)) #label score at each center
    num_realimgs_over_centers = np.zeros(len(centers_loc))
    for i in range(len(centers_loc)):
        center = centers_loc[i]
        interval_start = (center - args.FID_radius)/args.end_count
        interval_stop = (center + args.FID_radius)/args.end_count
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
        # Label score
        labelscores_over_centers[i], _ = cal_labelscore(PreNetLS, fake_images_curr, fake_labels_assigned_curr, min_label_before_shift=0, max_label_after_shift=args.end_count, batch_size = 200, resize = None)

        print("\r Center:{}; Real:{}; Fake:{}; FID:{}; LS:{}.".format(center, len(real_images_curr), len(fake_images_curr), FID_over_centers[i], labelscores_over_centers[i]))

    # average over all centers
    print("\n {} SFID: {}({}); min/max: {}/{}.".format(args.GAN, np.mean(FID_over_centers), np.std(FID_over_centers), np.min(FID_over_centers), np.max(FID_over_centers)))
    print("\n {} LS over centers: {}({}); min/max: {}/{}.".format(args.GAN, np.mean(labelscores_over_centers), np.std(labelscores_over_centers), np.min(labelscores_over_centers), np.max(labelscores_over_centers)))

    # dump FID versus number of samples (for each center) to npy
    dump_fid_ls_entropy_over_centers_filename = wd + "/cGAN-concat_fid_ls_entropy_over_centers"
    np.savez(dump_fid_ls_entropy_over_centers_filename, fids=FID_over_centers, labelscores=labelscores_over_centers, nrealimgs=num_realimgs_over_centers, centers=centers_loc)

    #####################
    # FID: Evaluate FID on all fake images
    indx_shuffle_real = np.arange(nreal_all); np.random.shuffle(indx_shuffle_real)
    indx_shuffle_fake = np.arange(nfake_all); np.random.shuffle(indx_shuffle_fake)
    FID = cal_FID(PreNetFID, real_images[indx_shuffle_real], fake_images[indx_shuffle_fake], batch_size = 200, resize = None)
    print("\n {}: FID of {} fake images: {}.".format(args.GAN, nfake_all, FID))

    #####################
    # Overall LS: abs(y_assigned - y_predicted)
    ls_mean_overall, ls_std_overall = cal_labelscore(PreNetLS, fake_images, fake_labels_assigned, min_label_before_shift=0, max_label_after_shift=args.end_count, batch_size = 200, resize = None)
    print("\n {}: overall LS of {} fake images: {}({}).".format(args.GAN, nfake_all, ls_mean_overall, ls_std_overall))

#######################################################################################
'''               Visualize fake images of the trained GAN                          '''
#######################################################################################
if args.visualize_fake_images:

    # First, visualize conditional generation on several unseen cell counts
    ## 3 rows (3 samples); 10 columns (10 unseen cell counts; displayed cell count starts from 10)
    n_row = 3
    n_col = 10
    all_unique_cellcounts = np.arange(args.start_count, args.end_count+1)
    unseen_unique_cellcounts = np.sort(np.setdiff1d(all_unique_cellcounts, selected_cellcounts))
    unseen_unique_cellcounts = unseen_unique_cellcounts[unseen_unique_cellcounts>=10]
    stepsize_tmp = len(unseen_unique_cellcounts)//(n_col-1)
    for i_col in range(n_col):
        if i_col==0:
            displayed_cellcounts = [unseen_unique_cellcounts[i_col]]
        else:
            displayed_cellcounts.append(unseen_unique_cellcounts[i_col*stepsize_tmp])
    print("\n Displayed {} cell counts are:".format(len(displayed_cellcounts)), displayed_cellcounts)

    ### output some real images as baseline
    filename_real_images = save_images_folder + '/real_images_grid_{}x{}.png'.format(n_row, n_col)
    if not os.path.isfile(filename_real_images):
        images_show = np.zeros((n_row*n_col, images.shape[1], images.shape[2], images.shape[3]))
        for i_row in range(n_row):
            for j_col in range(n_col):
                curr_label = displayed_cellcounts[j_col]
                indx_curr_label = np.where(raw_counts==curr_label)[0]
                np.random.shuffle(indx_curr_label)
                indx_curr_label = indx_curr_label[0]
                images_show[i_row*n_col+j_col] = raw_images[indx_curr_label]
        images_show = (images_show/255.0-0.5)/0.5
        images_show = torch.from_numpy(images_show)
        save_image(images_show.data, filename_real_images, nrow=n_col, normalize=True)

    ### output fake images from a trained GAN
    filename_fake_images = save_images_folder + '/{}_fake_images_grid_{}x{}.png'.format(args.GAN, n_row, n_col)
    images_show = np.zeros((n_row*n_col, images.shape[1], images.shape[2], images.shape[3]))
    for i_row in range(n_row):
        for j_col in range(n_col):
            curr_label = displayed_cellcounts[j_col]/args.end_count
            curr_image, _ = fn_sampleGAN_given_label(1, curr_label, 1)
            images_show[i_row*n_col+j_col,:,:,:] = curr_image
    images_show = torch.from_numpy(images_show)
    save_image(images_show.data, filename_fake_images, nrow=n_col, normalize=True)

    # #----------------------------------------------------------------
    # # dump 1000 images for each count
    # num_eval_labels = len(displayed_cellcounts)
    # for i in tqdm(range(num_eval_labels)):
    #     curr_label = displayed_cellcounts[i]/args.end_count
    #     curr_fake_images, curr_fake_labels = fn_sampleGAN_given_label(100, curr_label, args.samp_batch_size)
    #     if i == 0:
    #         fake_images = curr_fake_images
    #         fake_labels_assigned = curr_fake_labels.reshape(-1)
    #     else:
    #         fake_images = np.concatenate((fake_images, curr_fake_images), axis=0)
    #         fake_labels_assigned = np.concatenate((fake_labels_assigned, curr_fake_labels.reshape(-1)))
    #     # dump fake images
    #     curr_path = save_images_folder + '/{}/{}'.format(args.GAN, int(curr_label*args.end_count))
    #     os.makedirs(curr_path, exist_ok=True)
    #     for j in range(len(curr_fake_images)):
    #         curr_filename = curr_path + '/{}.jpg'.format(j)
    #         img_j = curr_fake_images[j]
    #         img_j = ((img_j*0.5+0.5)*255.0).astype(np.uint8)
    #         img_j_pil = Image.fromarray(img_j[0])
    #         img_j_pil.save(curr_filename)
    # #end for i

