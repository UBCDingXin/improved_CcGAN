"""
Some helpful functions

"""
import numpy as np
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import matplotlib as mpl
from torch.nn import functional as F
import sys
import PIL
from PIL import Image



# ################################################################################
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
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    # elif classname.find('BatchNorm') != -1:
    #     m.weight.data.normal_(1.0, 0.02)
    #     m.bias.data.fill_(0)

################################################################################
# torch dataset from numpy array
class IMGs_dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, normalize=False):
        super(IMGs_dataset, self).__init__()

        self.images = images
        self.n_images = len(self.images)
        self.labels = labels
        if labels is not None:
            if len(self.images) != len(self.labels):
                raise Exception('images (' +  str(len(self.images)) +') and labels ('+str(len(self.labels))+') do not have the same length!!!')
        self.normalize = normalize


    def __getitem__(self, index):

        image = self.images[index]

        if self.normalize:
            image = image/255.0
            image = (image-0.5)/0.5

        if self.labels is not None:
            label = self.labels[index]
            return (image, label)
        else:
            return image

    def __len__(self):
        return self.n_images


def PlotLoss(loss, filename):
    x_axis = np.arange(start = 1, stop = len(loss)+1)
    plt.switch_backend('agg')
    mpl.style.use('seaborn')
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_axis, np.array(loss))
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.legend()
    #ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),  shadow=True, ncol=3)
    #plt.title('Training Loss')
    plt.savefig(filename)



#---------------------------------------
def SampPrecGAN(netG, GAN_Latent_Length = 100, NFAKE = 10000, BATCH_SIZE = 500, NC = 1, IMG_SIZE = 256):
    raw_fake_images = np.zeros((NFAKE+BATCH_SIZE, NC, IMG_SIZE, IMG_SIZE))
    raw_fake_counts = np.zeros(NFAKE+BATCH_SIZE)
    netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < NFAKE:
            z = torch.randn(BATCH_SIZE, GAN_Latent_Length, dtype=torch.float).cuda()
            y = np.random.randint(MIN_COUNT, MAX_COUNT, n_row**2)
            y = torch.from_numpy(y).type(torch.float).view(-1,1).to(device)
            batch_fake_images = netG(z, y)
            raw_fake_images[tmp:(tmp+BATCH_SIZE)] = batch_fake_images.cpu().detach().numpy()
            raw_fake_counts[tmp:(tmp+BATCH_SIZE)] = y.cpu().detach().numpy()
            tmp += BATCH_SIZE
    #remove unused entry and extra samples
    raw_fake_images = raw_fake_images[0:NFAKE]
    raw_fake_counts = raw_fake_counts[0:NFAKE]
    return raw_fake_images, raw_fake_counts
