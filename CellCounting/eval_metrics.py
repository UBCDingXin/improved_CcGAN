"""
Compute
Inception Score (IS),
Frechet Inception Discrepency (FID), ref "https://github.com/mseitzer/pytorch-fid/blob/master/fid_score.py"
Maximum Mean Discrepancy (MMD)
for a set of fake images

use numpy array
Xr: high-level features for real images; nr by d array
Yr: labels for real images
Xg: high-level features for fake images; ng by d array
Yg: labels for fake images
IMGSr: real images
IMGSg: fake images

"""


import gc
import numpy as np
# from numpy import linalg as LA
from scipy import linalg
import torch
import torch.nn as nn
from scipy.stats import entropy
from torch.nn import functional as F

from utils import SimpleProgressBar




##############################################################################
# FID scores
##############################################################################
# compute FID based on extracted features
def FID(Xr, Xg, eps=1e-10):
    '''
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    '''
    #sample mean
    MUr = np.mean(Xr, axis = 0)
    MUg = np.mean(Xg, axis = 0)
    mean_diff = MUr - MUg
    #sample covariance
    SIGMAr = np.cov(Xr.transpose())
    SIGMAg = np.cov(Xg.transpose())

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(SIGMAr.dot(SIGMAg), disp=False)#square root of a matrix
    covmean = covmean.real
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(SIGMAr.shape[0]) * eps
        covmean = linalg.sqrtm((SIGMAr + offset).dot(SIGMAg + offset))

    #fid score
    fid_score = mean_diff.dot(mean_diff) + np.trace(SIGMAr + SIGMAg - 2*covmean)

    return fid_score

##test
#Xr = np.random.rand(10000,1000)
#Xg = np.random.rand(10000,1000)
#print(FID(Xr, Xg))

# compute FID from raw images
def cal_FID(PreNetFID, IMGSr, IMGSg, batch_size = 500, resize = None):
    #resize: if None, do not resize; if resize = (H,W), resize images to 3 x H x W

    PreNetFID.eval()

    nr = IMGSr.shape[0]
    ng = IMGSg.shape[0]

    nc = IMGSr.shape[1] #IMGSr is nrxNCxIMG_SIExIMG_SIZE
    img_size = IMGSr.shape[2]

    if batch_size > min(nr, ng):
        batch_size = min(nr, ng)
        # print("FID: recude batch size to {}".format(batch_size))

    #compute the length of extracted features
    with torch.no_grad():
        test_img = torch.from_numpy(IMGSr[0].reshape((1,nc,img_size,img_size))).type(torch.float).cuda()
        if resize is not None:
            test_img = nn.functional.interpolate(test_img, size = resize, scale_factor=None, mode='bilinear', align_corners=False)
        _, test_features = PreNetFID(test_img)
        d = test_features.shape[1] #length of extracted features

    Xr = np.zeros((nr, d))
    Xg = np.zeros((ng, d))

    #batch_size = 500
    with torch.no_grad():
        tmp = 0
        pb1 = SimpleProgressBar()
        for i in range(nr//batch_size):
            pb1.update(float(i)*100/(nr//batch_size))
            imgr_tensor = torch.from_numpy(IMGSr[tmp:(tmp+batch_size)]).type(torch.float).cuda()
            if resize is not None:
                imgr_tensor = nn.functional.interpolate(imgr_tensor, size = resize, scale_factor=None, mode='bilinear', align_corners=False)
            _, Xr_tmp = PreNetFID(imgr_tensor)
            Xr[tmp:(tmp+batch_size)] = Xr_tmp.detach().cpu().numpy()
            tmp+=batch_size
        del Xr_tmp,imgr_tensor; gc.collect()
        torch.cuda.empty_cache()

        tmp = 0
        pb2 = SimpleProgressBar()
        for j in range(ng//batch_size):
            pb2.update(float(j)*100/(ng//batch_size))
            imgg_tensor = torch.from_numpy(IMGSg[tmp:(tmp+batch_size)]).type(torch.float).cuda()
            if resize is not None:
                imgg_tensor = nn.functional.interpolate(imgg_tensor, size = resize, scale_factor=None, mode='bilinear', align_corners=False)
            _, Xg_tmp = PreNetFID(imgg_tensor)
            Xg[tmp:(tmp+batch_size)] = Xg_tmp.detach().cpu().numpy()
            tmp+=batch_size
        del Xg_tmp,imgg_tensor; gc.collect()
        torch.cuda.empty_cache()


    fid_score = FID(Xr, Xg, eps=1e-6)

    return fid_score






##############################################################################
# label_score
# difference between assigned label and predicted label
##############################################################################
def cal_labelscore(PreNet, images, labels_assi, batch_size = 500, resize = None):
    '''
    PreNet: pre-trained CNN
    images: fake images
    labels_assi: assigned labels
    resize: if None, do not resize; if resize = (H,W), resize images to 3 x H x W
    '''
    PreNet.eval()

    # assume images are nxncximg_sizeximg_size
    n = images.shape[0]
    nc = images.shape[1] #number of channels
    img_size = images.shape[2]
    labels_assi = labels_assi.reshape(-1)

    # predict labels
    labels_pred = np.zeros(n)
    with torch.no_grad():
        tmp = 0
        pb = SimpleProgressBar()
        for i in range(n//batch_size):
            pb.update(float(i)*100/(n//batch_size))
            image_tensor = torch.from_numpy(images[tmp:(tmp+batch_size)]).type(torch.float).cuda()
            if resize is not None:
                image_tensor = nn.functional.interpolate(image_tensor, size = resize, scale_factor=None, mode='bilinear', align_corners=False)
            labels_batch, _ = PreNet(image_tensor)
            labels_pred[tmp:(tmp+batch_size)] = labels_batch.detach().cpu().numpy().reshape(-1)
            tmp+=batch_size
        del image_tensor; gc.collect()
        torch.cuda.empty_cache()

    ls_mean = np.mean(np.abs(labels_pred-labels_assi))
    ls_std = np.std(np.abs(labels_pred-labels_assi))

    return ls_mean, ls_std
