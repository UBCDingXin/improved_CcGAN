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
from numpy import linalg as LA
from scipy import linalg
import torch
import torch.nn as nn
from scipy.stats import entropy
from torch.nn import functional as F

from utils import SimpleProgressBar

IMG_SIZE=28
NC=1


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
    mean_diff = LA.norm( MUr - MUg )
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
    fid_score = mean_diff + np.trace(SIGMAr + SIGMAg - 2*covmean)

    return fid_score

##test
#Xr = np.random.rand(10000,1000)
#Xg = np.random.rand(10000,1000)
#print(FID(Xr, Xg))

# compute FID from raw images
def FID_RAW(PreNetFID, IMGSr, IMGSg, batch_size = 500, NGPU=1, resize = None):
    #resize: if None, do not resize; if resize = (H,W), resize images to 3 x H x W

#    if NGPU>1:
#        PreNetFID = nn.DataParallel(PreNetFID).cuda()
#    else:
#        PreNetFID = PreNetFID.cuda()
#
#    PreNetFID.load_state_dict(checkpoint_PreNet['net_state_dict'])
    PreNetFID.eval()

    nr = IMGSr.shape[0]
    ng = IMGSg.shape[0]

    #compute the length of extracted features
    with torch.no_grad():
        test_img = torch.from_numpy(IMGSr[0].reshape((1,NC,IMG_SIZE,IMG_SIZE))).type(torch.float).cuda()
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
# Inception Scores
##############################################################################
def IS_RAW(PreNetIS, IMGSg, batch_size = 500, splits=1, NGPU=2, resize = None):
    #resize: if None, do not resize; if resize = (H,W), resize images to 3 x H x W

#    if NGPU>1:
#        PreNetIS = nn.DataParallel(PreNetIS).cuda()
#    else:
#        PreNetIS = PreNetIS.cuda()
#
#    PreNetIS.load_state_dict(checkpoint_PreNet['net_state_dict'])
    PreNetIS.eval()

    N = IMGSg.shape[0]

    #compute the number of classes
    with torch.no_grad():
        test_img = torch.from_numpy(IMGSg[0].reshape((1,NC,IMG_SIZE,IMG_SIZE))).type(torch.float).cuda()
        if resize is not None:
            test_img = nn.functional.interpolate(test_img, size = resize, scale_factor=None, mode='bilinear', align_corners=False)
        test_output, _ = PreNetIS(test_img)
        nc = test_output.shape[1] #number of classes

    # Get predictions
    def get_pred(x):
        x, _ = PreNetIS(x)
        return F.softmax(x,dim=1).data.cpu().numpy()

    preds = np.zeros((N, nc))

    with torch.no_grad():
        tmp = 0
        pb = SimpleProgressBar()
        for j in range(N//batch_size):
            pb.update(float(j)*100/(N//batch_size))
            imgg_tensor = torch.from_numpy(IMGSg[tmp:(tmp+batch_size)]).type(torch.float).cuda()
            if resize is not None:
                imgg_tensor = nn.functional.interpolate(imgg_tensor, size = resize, scale_factor=None, mode='bilinear', align_corners=False)
            preds[tmp:(tmp+batch_size)] = get_pred(imgg_tensor)
            tmp+=batch_size

    # Now compute the mean kl-div
    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
