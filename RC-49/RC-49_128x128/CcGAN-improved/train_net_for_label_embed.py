
import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import os
import timeit
from PIL import Image




#-------------------------------------------------------------
def train_net_embed(net, net_name, trainloader, testloader, epochs=200, resume_epoch = 0, lr_base=0.01, lr_decay_factor=0.1, lr_decay_epochs=[80, 140], weight_decay=1e-4, path_to_ckpt = None):

    ''' learning rate decay '''
    def adjust_learning_rate_1(optimizer, epoch):
        """decrease the learning rate """
        lr = lr_base

        num_decays = len(lr_decay_epochs)
        for decay_i in range(num_decays):
            if epoch >= lr_decay_epochs[decay_i]:
                lr = lr * lr_decay_factor
            #end if epoch
        #end for decay_i
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    net = net.cuda()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = lr_base, momentum= 0.9, weight_decay=weight_decay)

    # resume training; load checkpoint
    if path_to_ckpt is not None and resume_epoch>0:
        save_file = path_to_ckpt + "/embed_x2y_ckpt_in_train/embed_x2y_checkpoint_epoch_{}.pth".format(resume_epoch)
        checkpoint = torch.load(save_file)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if

    start_tmp = timeit.default_timer()
    for epoch in range(resume_epoch, epochs):
        net.train()
        train_loss = 0
        adjust_learning_rate_1(optimizer, epoch)
        for _, (batch_train_images, batch_train_labels) in enumerate(trainloader):

            # batch_train_images = nn.functional.interpolate(batch_train_images, size = (299,299), scale_factor=None, mode='bilinear', align_corners=False)

            batch_train_images = batch_train_images.type(torch.float).cuda()
            batch_train_labels = batch_train_labels.type(torch.float).view(-1,1).cuda()

            #Forward pass
            outputs, _ = net(batch_train_images)
            loss = criterion(outputs, batch_train_labels)

            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
        #end for batch_idx
        train_loss = train_loss / len(trainloader)

        if testloader is None:
            print('Train net_x2y for embedding: [epoch %d/%d] train_loss:%f Time:%.4f' % (epoch+1, epochs, train_loss, timeit.default_timer()-start_tmp))
        else:
            net.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
            with torch.no_grad():
                test_loss = 0
                for batch_test_images, batch_test_labels in testloader:
                    batch_test_images = batch_test_images.type(torch.float).cuda()
                    batch_test_labels = batch_test_labels.type(torch.float).view(-1,1).cuda()
                    outputs,_ = net(batch_test_images)
                    loss = criterion(outputs, batch_test_labels)
                    test_loss += loss.cpu().item()
                test_loss = test_loss/len(testloader)

                print('Train net_x2y for label embedding: [epoch %d/%d] train_loss:%f test_loss:%f Time:%.4f' % (epoch+1, epochs, train_loss, test_loss, timeit.default_timer()-start_tmp))

        #save checkpoint
        if path_to_ckpt is not None and (((epoch+1) % 50 == 0) or (epoch+1==epochs)):
            save_file = path_to_ckpt + "/embed_x2y_ckpt_in_train/embed_x2y_checkpoint_epoch_{}.pth".format(epoch+1)
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            torch.save({
                    'epoch': epoch,
                    'net_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'rng_state': torch.get_rng_state()
            }, save_file)
    #end for epoch

    return net




###################################################################################
class label_dataset(torch.utils.data.Dataset):
    def __init__(self, labels):
        super(label_dataset, self).__init__()

        self.labels = labels
        self.n_samples = len(self.labels)

    def __getitem__(self, index):

        y = self.labels[index]
        return y

    def __len__(self):
        return self.n_samples


def train_net_y2h(unique_labels_norm, net_y2h, net_embed, epochs=500, lr_base=0.01, lr_decay_factor=0.1, lr_decay_epochs=[150, 250, 350], weight_decay=1e-4, batch_size=128):
    '''
    unique_labels_norm: an array of normalized unique labels
    '''

    ''' learning rate decay '''
    def adjust_learning_rate_2(optimizer, epoch):
        """decrease the learning rate """
        lr = lr_base

        num_decays = len(lr_decay_epochs)
        for decay_i in range(num_decays):
            if epoch >= lr_decay_epochs[decay_i]:
                lr = lr * lr_decay_factor
            #end if epoch
        #end for decay_i
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    assert np.max(unique_labels_norm)<=1 and np.min(unique_labels_norm)>=0
    trainset = label_dataset(unique_labels_norm)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    net_embed.eval()
    net_h2y=net_embed.module.h2y #convert embedding labels to original labels
    optimizer_y2h = torch.optim.SGD(net_y2h.parameters(), lr = lr_base, momentum= 0.9, weight_decay=weight_decay)

    start_tmp = timeit.default_timer()
    for epoch in range(epochs):
        net_y2h.train()
        train_loss = 0
        adjust_learning_rate_2(optimizer_y2h, epoch)
        for _, batch_labels in enumerate(trainloader):

            batch_labels = batch_labels.type(torch.float).view(-1,1).cuda()

            # generate noises which will be added to labels
            batch_size_curr = len(batch_labels)
            batch_gamma = np.random.normal(0, 0.2, batch_size_curr)
            batch_gamma = torch.from_numpy(batch_gamma).view(-1,1).type(torch.float).cuda()

            # add noise to labels
            batch_labels_noise = torch.clamp(batch_labels+batch_gamma, 0.0, 1.0)

            #Forward pass
            batch_hiddens_noise = net_y2h(batch_labels_noise)
            batch_rec_labels_noise = net_h2y(batch_hiddens_noise)

            loss = nn.MSELoss()(batch_rec_labels_noise, batch_labels_noise)

            #backward pass
            optimizer_y2h.zero_grad()
            loss.backward()
            optimizer_y2h.step()

            train_loss += loss.cpu().item()
        #end for batch_idx
        train_loss = train_loss / len(trainloader)

        print('\n Train net_y2h: [epoch %d/%d] train_loss:%f Time:%.4f' % (epoch+1, epochs, train_loss, timeit.default_timer()-start_tmp))
    #end for epoch

    return net_y2h
