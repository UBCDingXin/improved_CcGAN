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
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

################################################################################
# torch dataset from numpy array
class IMGs_dataset(torch.utils.data.Dataset):
    def __init__(self, images, labels=None, normalize=False, rotate=False, degrees = 15, crop=False, crop_size=28, crop_pad=4):
        super(IMGs_dataset, self).__init__()

        self.images = images
        self.n_images = len(self.images)
        self.labels = labels
        if labels is not None:
            if len(self.images) != len(self.labels):
                raise Exception('images (' +  str(len(self.images)) +') and labels ('+str(len(self.labels))+') do not have the same length!!!')
        self.normalize = normalize
        self.rotate = rotate
        self.TransHRotate = torchvision.transforms.RandomRotation(degrees=degrees)
        self.crop = crop
        self.TransCrop = torchvision.transforms.RandomCrop(size=crop_size, padding=4)

    def __getitem__(self, index):

        image = self.images[index]

        if self.rotate or self.crop:
            assert np.max(image)>1
            image = image[0]#C * W * H ----> W * H
            PIL_im = Image.fromarray(np.uint8(image), mode = 'L') #W * H
            if self.crop:
                PIL_Im = self.TransCrop(PIL_im)
            if self.rotate:
                PIL_im = self.TransHRotate(PIL_im)
            image = np.array(PIL_im)
            image = image[np.newaxis,:,:] #now C * W * H

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
def SampPreGAN(netG, GAN_Latent_Length = 100, Conditional = False, NFAKE = 10000, BATCH_SIZE = 500, N_CLASS = 10, NC = 3, IMG_SIZE = 32):
    raw_fake_images = np.zeros((NFAKE+BATCH_SIZE, NC, IMG_SIZE, IMG_SIZE))
    netG.eval()
    with torch.no_grad():
        tmp = 0
        while tmp < NFAKE:
            if not Conditional:
                z = torch.randn(BATCH_SIZE, GAN_Latent_Length, 1, 1, dtype=torch.float).cuda()
                batch_fake_images = netG(z)
                batch_fake_images = batch_fake_images.detach()
            else:
                z = torch.randn(BATCH_SIZE, GAN_Latent_Length, dtype=torch.float).cuda()
                gen_labels = torch.from_numpy(np.random.randint(0,N_CLASS,BATCH_SIZE)).type(torch.long).cuda()
                gen_labels_onehot = torch.FloatTensor(BATCH_SIZE, N_CLASS).cuda()
                gen_labels_onehot.zero_()
                gen_labels_onehot.scatter_(1,gen_labels.reshape(BATCH_SIZE,1),1)
                batch_fake_images = netG(z, gen_labels_onehot)
                batch_fake_images = batch_fake_images.detach()
            raw_fake_images[tmp:(tmp+BATCH_SIZE)] = batch_fake_images.cpu().detach().numpy()
            tmp += BATCH_SIZE
    #remove unused entry and extra samples
    raw_fake_images = raw_fake_images[0:NFAKE]
    return raw_fake_images

#---------------------------------------
def PredictLabel(IMGs, PreTrained_CNN, N_CLASS = 10, BATCH_SIZE = 500, resize = (299, 299)):
    N_IMGs = IMGs.shape[0]
    probs_pred = np.zeros((N_IMGs+BATCH_SIZE, N_CLASS))
    def get_pred(x):
        x, _ = PreTrained_CNN(x)
        return F.softmax(x,dim=1).data.cpu().numpy()
    assert N_IMGs > BATCH_SIZE;
    assert N_IMGs % BATCH_SIZE == 0; #check whether N_IMGs is divisible by BATCH_SIZE

    PreTrained_CNN.eval()
    with torch.no_grad():
        num_imgs = 0
        pb = SimpleProgressBar()
        while num_imgs < N_IMGs:
            imgg_tensor = torch.from_numpy(IMGs[num_imgs:(num_imgs+BATCH_SIZE)]).type(torch.float).cuda()
            if resize is not None:
                imgg_tensor = nn.functional.interpolate(imgg_tensor, size = resize, scale_factor=None, mode='bilinear', align_corners=False)
            probs_pred[num_imgs:(num_imgs+BATCH_SIZE)] = get_pred(imgg_tensor)
            num_imgs += BATCH_SIZE
            pb.update(float(num_imgs)*100/(N_IMGs))
    probs_pred = probs_pred[0:N_IMGs]
    labels_pred = np.argmax(probs_pred, axis=1)
    return labels_pred
