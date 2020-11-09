"""
Some helper functions

"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.nn import functional as F
import sys
from scipy.stats import multivariate_normal
import os
from numpy import linalg as LA
from scipy import linalg


################################################################################
# Progress Bar
class SimpleProgressBar():
    def __init__(self, width=50):
        self.last_x = -1
        self.width = width

    def update(self, x):
        assert 0 <= x <= 100 # `x`: progress in percent ( between 0 and 100)
        if self.last_x == int(x): return
        self.last_x = int(x)
        pointer = int(self.width * (x / 100.0))
        sys.stdout.write( '\r%d%% [%s]' % (int(x), '#' * pointer + '.' * (self.width - pointer)))
        sys.stdout.flush()
        if x == 100:
            print('')


################################################################################
# torch dataset from numpy array
class custom_dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels=None):
        super(custom_dataset, self).__init__()

        self.data = data
        if labels is not None:
            self.labels = labels
        else:
            self.labels = None
        self.n_samples = self.data.shape[0]

    def __getitem__(self, index):

        x = self.data[index]
        if self.labels is not None:
            y = self.labels[index]
        else:
            y = -1
        return x, y

    def __len__(self):
        return self.n_samples

################################################################################
# Plot training loss
def PlotLoss(loss, filename):
    x_axis = np.arange(start = 1, stop = len(loss)+1)
    plt.switch_backend('agg')
    mpl.style.use('seaborn')
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_axis, np.array(loss))
    plt.legend()
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),  shadow=True, ncol=3)
    plt.title('Loss')
    plt.savefig(filename)

################################################################################
# Creat sampler for a mixture of Gaussian distributions
def sampler_CircleGaussian(n_samp_per_gaussian, angle_grid, radius, sigma = 0.05, dim = 2):
    '''

    n_samp_per_gaussian: how many samples will be draw from each Gaussian
    angle_grid: raw angles
    sigma: a fixed standard deviation
    dim: dimension of a component

    '''

    cov = np.diag(np.repeat(sigma**2, dim)) #covariance matrix; firxed for each component
    n_gaussians = len(angle_grid)
    means = np.zeros((n_gaussians, dim))
    for i in range(n_gaussians):
        angle = angle_grid[i]
        mean_curr = np.array([radius*np.sin(angle), radius*np.cos(angle)])
        means[i] = mean_curr

        if i == 0:
            samples = np.random.multivariate_normal(mean_curr, cov, size=n_samp_per_gaussian)
            angles = np.ones(n_samp_per_gaussian) * angle
        else:
            samples = np.concatenate((samples, np.random.multivariate_normal(mean_curr, cov, size=n_samp_per_gaussian)), axis=0)
            angles = np.concatenate((angles, np.ones(n_samp_per_gaussian) * angle), axis=0)

    assert len(samples) == n_samp_per_gaussian*n_gaussians
    assert len(angles) == n_samp_per_gaussian*n_gaussians
    assert samples.shape[1] == dim

    return samples, angles, means



################################################################################
# Plot samples in a 2-D coordinate
def ScatterPoints(tar_samples, prop_samples, filename, plot_real_samples = False, fig_size=5, point_size=None):
    # tar_samples and prop_samples are 2-D array: n_samples by num_features
    plt.switch_backend('agg')
    mpl.style.use('seaborn')
    plt.figure(figsize=(fig_size, fig_size), facecolor='w')
    plt.grid(b=True)
    plt.scatter(tar_samples[:, 0], tar_samples[:, 1], c='blue', edgecolor='none', alpha=0.5, s=point_size)
    if not os.path.isfile(filename[0:-4]+'_realsamples.pdf') and plot_real_samples:
        plt.savefig(filename[0:-4]+'_realsamples.png')
    plt.scatter(prop_samples[:, 0], prop_samples[:, 1], c='g', edgecolor='none', s=point_size)
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()



################################################################################
# Compute 2-Wasserstein distance
def two_wasserstein(mu1, mu2, cov1, cov2, eps=1e-10):

    mean_diff = mu1 - mu2
    # mean_diff = LA.norm( mu1 - mu2 )
    covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)#square root of a matrix
    covmean = covmean.real

    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(cov1.shape[0]) * eps
        covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

    #2-Wasserstein distance
    output = mean_diff.dot(mean_diff) + np.trace(cov1 + cov2 - 2*covmean)

    return output
