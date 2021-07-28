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
# sampling parameters
assert args.eval_mode in [1,2,3,4] #evaluation mode must be in 1,2,3,4
if args.data_split == "all":
    args.eval_mode != 1

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
data_filename = args.data_path + '/RC-49_{}x{}.h5'.format(args.img_size, args.img_size)
hf = h5py.File(data_filename, 'r')
labels_all = hf['labels'][:]
labels_all = labels_all.astype(float)
images_all = hf['images'][:]
indx_train = hf['indx_train'][:]
hf.close()
print("\n RC-49 dataset shape: {}x{}x{}x{}".format(images_all.shape[0], images_all.shape[1], images_all.shape[2], images_all.shape[3]))

# data split
if args.data_split == "train":
    images_train = images_all[indx_train]
    labels_train_raw = labels_all[indx_train]
else:
    images_train = copy.deepcopy(images_all)
    labels_train_raw = copy.deepcopy(labels_all)

# only take images with label in (q1, q2)
q1 = args.min_label
q2 = args.max_label
indx = np.where((labels_train_raw>q1)*(labels_train_raw<q2)==True)[0]
labels_train_raw = labels_train_raw[indx]
images_train = images_train[indx]
assert len(labels_train_raw)==len(images_train)

if args.visualize_fake_images or args.comp_FID:
    indx = np.where((labels_all>q1)*(labels_all<q2)==True)[0]
    labels_all = labels_all[indx]
    images_all = images_all[indx]
    assert len(labels_all)==len(images_all)


### show some real  images
if args.show_real_imgs:
    unique_labels_show = np.array(sorted(list(set(labels_all))))
    indx_show = np.arange(0, len(unique_labels_show), len(unique_labels_show)//9)
    unique_labels_show = unique_labels_show[indx_show]
    nrow = len(unique_labels_show); ncol = 1
    sel_labels_indx = []
    for i in range(nrow):
        curr_label = unique_labels_show[i]
        indx_curr_label = np.where(labels_all==curr_label)[0]
        np.random.shuffle(indx_curr_label)
        indx_curr_label = indx_curr_label[0:ncol]
        sel_labels_indx.extend(list(indx_curr_label))
    sel_labels_indx = np.array(sel_labels_indx)
    images_show = images_all[sel_labels_indx]
    print(images_show.mean())
    images_show = (images_show/255.0-0.5)/0.5
    images_show = torch.from_numpy(images_show)
    save_image(images_show.data, save_images_folder +'/real_images_grid_{}x{}.png'.format(nrow, ncol), nrow=ncol, normalize=True)

# for each angle, take no more than args.max_num_img_per_label images
image_num_threshold = args.max_num_img_per_label
print("\n Original set has {} images; For each angle, take no more than {} images>>>".format(len(images_train), image_num_threshold))
unique_labels_tmp = np.sort(np.array(list(set(labels_train_raw))))
for i in tqdm(range(len(unique_labels_tmp))):
    indx_i = np.where(labels_train_raw == unique_labels_tmp[i])[0]
    if len(indx_i)>image_num_threshold:
        np.random.shuffle(indx_i)
        indx_i = indx_i[0:image_num_threshold]
    if i == 0:
        sel_indx = indx_i
    else:
        sel_indx = np.concatenate((sel_indx, indx_i))
images_train = images_train[sel_indx]
labels_train_raw = labels_train_raw[sel_indx]
print("{} images left and there are {} unique labels".format(len(images_train), len(set(labels_train_raw))))


# normalize labels_train_raw
print("\n Range of unnormalized labels: ({},{})".format(np.min(labels_train_raw), np.max(labels_train_raw)))

labels_train = labels_train_raw / args.max_label




#######################################################################################
'''                                    GAN training                                 '''
#######################################################################################

start = timeit.default_timer()
print("\n Begin Training cGAN-concat with %s:" % args.GAN_arch)

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
        netG, netD = train_cgan(images_train, labels_train, netG, netD, save_images_folder=save_images_in_train_folder, save_models_folder = save_models_folder)

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
    print("\n Evaluation in Mode {}...".format(args.eval_mode))

    PreNetFID = encoder(dim_bottleneck=512).cuda()
    PreNetFID = nn.DataParallel(PreNetFID)
    Filename_PreCNNForEvalGANs = os.path.join(args.eval_ckpt_path, 'ckpt_AE_epoch_200_seed_2020_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs)
    PreNetFID.load_state_dict(checkpoint_PreNet['net_encoder_state_dict'])

    # Diversity: entropy of predicted races within each eval center
    PreNetDiversity = ResNet34_class(num_classes=49, ngpu = torch.cuda.device_count()).cuda() #49 chair types
    Filename_PreCNNForEvalGANs_Diversity = os.path.join(args.eval_ckpt_path, 'ckpt_PreCNNForEvalGANs_ResNet34_class_epoch_200_seed_2020_classify_49_chair_types_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_Diversity)
    PreNetDiversity.load_state_dict(checkpoint_PreNet['net_state_dict'])

    # for LS
    PreNetLS = ResNet34_regre(ngpu = torch.cuda.device_count()).cuda()
    Filename_PreCNNForEvalGANs_LS = os.path.join(args.eval_ckpt_path, 'ckpt_PreCNNForEvalGANs_ResNet34_regre_epoch_200_seed_2020_CVMode_False.pth')
    checkpoint_PreNet = torch.load(Filename_PreCNNForEvalGANs_LS)
    PreNetLS.load_state_dict(checkpoint_PreNet['net_state_dict'])

    #####################
    # generate nfake images
    print("\n Start sampling {} fake images per label from GAN >>>".format(args.nfake_per_label))

    if args.eval_mode == 1: #Mode 1: eval on unique labels used for GAN training
        eval_labels = np.sort(np.array(list(set(labels_train_raw)))) #not normalized
    elif args.eval_mode in [2, 3]: #Mode 2 and 3: eval on all unique labels in the dataset
        eval_labels = np.sort(np.array(list(set(labels_all)))) #not normalized
    else: #Mode 4: eval on a interval [min_label, max_label] with num_eval_labels labels
        eval_labels = np.linspace(np.min(labels_all), np.max(labels_all), args.num_eval_labels) #not normalized

    unique_eval_labels = list(set(eval_labels))
    print("\n There are {} unique eval labels.".format(len(unique_eval_labels)))

    eval_labels_norm = eval_labels/args.max_label #normalized

    for i in range(len(eval_labels)):
        curr_label = eval_labels_norm[i]
        if i == 0:
            fake_labels_assigned = np.ones(args.nfake_per_label)*curr_label
        else:
            fake_labels_assigned = np.concatenate((fake_labels_assigned, np.ones(args.nfake_per_label)*curr_label))
    fake_images, _ = fn_sampleGAN_given_labels(fake_labels_assigned, args.samp_batch_size)
    assert len(fake_images) == args.nfake_per_label*len(eval_labels)
    assert len(fake_labels_assigned) == args.nfake_per_label*len(eval_labels)
    assert fake_images.min()>=0 and fake_images.max()<=255.0

    ## dump fake images for computing NIQE
    if args.dump_fake_for_NIQE:
        print("\n Dumping fake images for NIQE...")
        dump_fake_images_folder = save_images_folder + '/fake_images_for_NIQE_nfake_{}'.format(len(fake_images))
        os.makedirs(dump_fake_images_folder, exist_ok=True)
        for i in tqdm(range(len(fake_images))):
            label_i = fake_labels_assigned[i]*args.max_label
            filename_i = dump_fake_images_folder + "/{}_{}.png".format(i, label_i)
            os.makedirs(os.path.dirname(filename_i), exist_ok=True)
            image_i = fake_images[i].astype(np.uint8)
            # image_i = ((image_i*0.5+0.5)*255.0).astype(np.uint8)
            image_i_pil = Image.fromarray(image_i.transpose(1,2,0))
            image_i_pil.save(filename_i)
        #end for i

    print("End sampling! We got {} fake images.".format(len(fake_images)))


    #####################
    # normalize images
    if args.eval_mode in [1, 3]:
        real_images = (images_train/255.0-0.5)/0.5
        real_labels = labels_train_raw #not normalized
    else: #for both mode 2 and 4
        real_images = (images_all/255.0-0.5)/0.5
        real_labels = labels_all #not normalized
    fake_images = (fake_images/255.0-0.5)/0.5
    

    #######################
    # For each label take nreal_per_label images
    unique_labels_real = np.sort(np.array(list(set(real_labels))))
    indx_subset = []
    for i in range(len(unique_labels_real)):
        label_i = unique_labels_real[i]
        indx_i = np.where(real_labels==label_i)[0]
        np.random.shuffle(indx_i)
        if args.nreal_per_label>1:
            indx_i = indx_i[0:args.nreal_per_label]
        indx_subset.append(indx_i)
    indx_subset = np.concatenate(indx_subset)
    real_images = real_images[indx_subset]
    real_labels = real_labels[indx_subset]


    nfake_all = len(fake_images)
    nreal_all = len(real_images)


    ######################
    ## Evaluate FID within a sliding window with a radius R on the label's range (not normalized range, i.e., [min_label,max_label]). The center of the sliding window locate on [min_label+R,...,max_label-R].
    if args.eval_mode == 1:
        center_start = np.min(labels_train_raw)+args.FID_radius ##bug???
        center_stop = np.max(labels_train_raw)-args.FID_radius
    else:
        center_start = np.min(labels_all)+args.FID_radius
        center_stop = np.max(labels_all)-args.FID_radius

    if args.FID_num_centers<=0 and args.FID_radius==0: #completely overlap
        centers_loc = eval_labels #not normalized
    elif args.FID_num_centers>0:
        centers_loc = np.linspace(center_start, center_stop, args.FID_num_centers) #not normalized
    else:
        print("\n Error.")
    FID_over_centers = np.zeros(len(centers_loc))
    entropies_over_centers = np.zeros(len(centers_loc)) # entropy at each center
    labelscores_over_centers = np.zeros(len(centers_loc)) #label score at each center
    num_realimgs_over_centers = np.zeros(len(centers_loc))
    for i in range(len(centers_loc)):
        center = centers_loc[i]
        interval_start = (center - args.FID_radius)#/args.max_label
        interval_stop = (center + args.FID_radius)#/args.max_label
        indx_real = np.where((real_labels>=interval_start)*(real_labels<=interval_stop)==True)[0]
        np.random.shuffle(indx_real)
        real_images_curr = real_images[indx_real]
        num_realimgs_over_centers[i] = len(real_images_curr)
        indx_fake = np.where((fake_labels_assigned>=(interval_start/args.max_label))*(fake_labels_assigned<=(interval_stop/args.max_label))==True)[0]
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

        print("\n [{}/{}] Center:{}; Real:{}; Fake:{}; FID:{}; LS:{}; ET:{}.".format(i+1, len(centers_loc), center, len(real_images_curr), len(fake_images_curr), FID_over_centers[i], labelscores_over_centers[i], entropies_over_centers[i]))
    # end for i
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
    IS, IS_std = inception_score(imgs=fake_images[indx_shuffle_fake], num_classes=49, net=PreNetDiversity, cuda=True, batch_size = 200, splits=10, normalize_img=False)
    print("\n {}: IS of {} fake images: {}({}).".format(args.GAN_arch, nfake_all, IS, IS_std))


    #####################
    # Dump evaluation results
    eval_results_logging_fullpath = os.path.join(path_to_output, 'eval_results_{}.txt'.format(args.GAN_arch))
    if not os.path.isfile(eval_results_logging_fullpath):
        eval_results_logging_file = open(eval_results_logging_fullpath, "w")
        eval_results_logging_file.close()
    with open(eval_results_logging_fullpath, 'a') as eval_results_logging_file:
        eval_results_logging_file.write("\n===================================================================================================")
        eval_results_logging_file.write("\n Eval Mode: {}; Radius: {}; # Centers: {}.  \n".format(args.eval_mode, args.FID_radius, args.FID_num_centers))
        print(args, file=eval_results_logging_file)
        eval_results_logging_file.write("\n SFID: {}({}).".format(np.mean(FID_over_centers), np.std(FID_over_centers)))
        eval_results_logging_file.write("\n LS: {}({}).".format(np.mean(labelscores_over_centers), np.std(labelscores_over_centers)))
        eval_results_logging_file.write("\n Diversity: {}({}).".format(np.mean(entropies_over_centers), np.std(entropies_over_centers)))



#######################################################################################
'''               Visualize fake images of the trained GAN                          '''
#######################################################################################
if args.visualize_fake_images:

    # First, visualize conditional generation # vertical grid
    ## 10 rows; 3 columns (3 samples for each age)
    n_row = 10
    n_col = 10

    displayed_unique_labels = np.sort(np.array(list(set(labels_all))))
    displayed_labels_indx = (np.linspace(0.05, 0.95, n_row)*len(displayed_unique_labels)).astype(int)
    displayed_labels = displayed_unique_labels[displayed_labels_indx] #not normalized
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

    ### output some real images as baseline
    filename_real_images = save_images_folder + '/real_images_grid_{}x{}.png'.format(n_row, n_col)
    if not os.path.isfile(filename_real_images):
        images_show = np.zeros((n_row*n_col, args.num_channels, args.img_size, args.img_size))
        for i_row in range(n_row):
            # generate 3 real images from each interval
            curr_label = displayed_labels[i_row]
            for j_col in range(n_col):
                indx_curr_label = np.where(labels_all==curr_label)[0]
                np.random.shuffle(indx_curr_label)
                indx_curr_label = indx_curr_label[0]
                images_show[i_row*n_col+j_col] = images_all[indx_curr_label]
        #end for i_row
        images_show = (images_show/255.0-0.5)/0.5
        images_show = torch.from_numpy(images_show)
        save_image(images_show.data, filename_real_images, nrow=n_col, normalize=True)


print("\n===================================================================================================")