
import torch
import torch.nn as nn
from torchvision.utils import save_image
import numpy as np
import os
import timeit
from PIL import Image



###################################################################################
def adjust_learning_rate(optimizer, epoch, base_lr):
    lr = base_lr
    if epoch >= 60:
        lr /= 5
    if epoch >= 120:
        lr /= 5
    if epoch >= 160:
        lr /= 5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


#-------------------------------------------------------------
def train_net_embed(trainloader, testloader, net, optimizer, epochs=200, base_lr=0.1, save_models_folder = None, resumeepoch = 0, device="cuda"):

    criterion = nn.MSELoss()
    net=net.to(device)

    # resume training; load checkpoint
    if save_models_folder is not None and resumeepoch>0:
        save_file = save_models_folder + "/embed_cnn_checkpoint_intrain/embed_cnn_checkpoint_epoch" + str(resumeepoch) + ".pth"
        checkpoint = torch.load(save_file)
        net.load_state_dict(checkpoint['net_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        torch.set_rng_state(checkpoint['rng_state'])
    #end if

    start_tmp = timeit.default_timer()
    for epoch in range(resumeepoch, epochs):
        net.train()
        train_loss = 0
        adjust_learning_rate(optimizer, epoch, base_lr)
        for batch_idx, (batch_train_images, batch_train_labels) in enumerate(trainloader):

            # batch_train_images = nn.functional.interpolate(batch_train_images, size = (299,299), scale_factor=None, mode='bilinear', align_corners=False)

            batch_train_images = batch_train_images.type(torch.float).to(device)
            batch_train_labels = batch_train_labels.type(torch.float).view(-1,1).to(device)

            #Forward pass
            outputs, batch_train_features = net(batch_train_images)
            loss = criterion(outputs, batch_train_labels)

            # batch_train_feature_cov = cov(batch_train_features)


            #backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
        #end for batch_idx
        train_loss = train_loss / len(trainloader)

        # print('Train net_x2y for embedding: [epoch %d/%d] train_loss:%f Time:%.4f' % (epoch+1, epochs, train_loss, timeit.default_timer()-start_tmp))


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
        if save_models_folder is not None and (((epoch+1) % 50 == 0) or (epoch+1==epochs)):
            save_file = save_models_folder + "/embed_cnn_checkpoint_intrain"
            os.makedirs(save_file, exist_ok=True)
            save_file = save_file + "/embed_cnn_checkpoint_epoch" + str(epoch+1) + ".pth"
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


def adjust_learning_rate2(optimizer, epoch, base_lr):
    lr = base_lr
    if epoch >= 100:
        lr /= 5
    if epoch >= 200:
        lr /= 5
    if epoch >= 400:
        lr /= 5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_net_y2h(unique_labels_norm, net_y2h, net_embed, optimizer_y2h, epochs=400, base_lr=0.1, batch_size=128, device="cuda"):
    '''
    unique_labels_norm: an array of normalized unique labels
    '''
    assert np.max(unique_labels_norm)<=1 and np.min(unique_labels_norm)>=0
    trainset = label_dataset(unique_labels_norm)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    net_embed.eval()
    net_h2y=net_embed.h2y #convert embedding labels to original labels

    start_tmp = timeit.default_timer()
    for epoch in range(epochs):
        net_y2h.train()
        train_loss = 0; train_loss_1 = 0; train_loss_2=0
        adjust_learning_rate2(optimizer_y2h, epoch, base_lr)
        for batch_idx, batch_labels in enumerate(trainloader):

            batch_labels = batch_labels.type(torch.float).view(-1,1).to(device)

            # generate noises which will be added to labels
            batch_size_curr = len(batch_labels)
            batch_gamma = np.random.normal(0, 0.2, batch_size_curr)
            batch_gamma = torch.from_numpy(batch_gamma).view(-1,1).type(torch.float).to(device)

            # add noise to labels
            batch_labels_noise = torch.clamp(batch_labels+batch_gamma, 0.0, 1.0)

            #Forward pass
            batch_hiddens = net_y2h(batch_labels)
            batch_hiddens_noise = net_y2h(batch_labels_noise)
            batch_rec_labels_noise = net_h2y(batch_hiddens_noise)

            loss1 = nn.MSELoss()(batch_rec_labels_noise, batch_labels_noise)
            # loss2 = - nn.MSELoss()(batch_hiddens, batch_hiddens_noise)
            batch_label_diff = (batch_labels_noise-batch_labels)**2
            batch_hidden_diff = torch.mean((batch_hiddens-batch_hiddens_noise)**2, dim=1, keepdim=True)
            loss2 = torch.mean((batch_hidden_diff - 10*batch_label_diff)**2)
            loss = loss1 #+ loss2

            # loss = nn.MSELoss()(batch_rec_labels_noise, batch_labels_noise)

            #backward pass
            optimizer_y2h.zero_grad()
            loss.backward()
            optimizer_y2h.step()

            train_loss += loss.cpu().item()
            train_loss_1 += loss1.cpu().item()
            train_loss_2 += loss2.cpu().item()
        #end for batch_idx
        train_loss = train_loss / len(trainloader)

        # print('Train net_y2h: [epoch %d/%d] train_loss:%f Time:%.4f' % (epoch+1, epochs, train_loss, timeit.default_timer()-start_tmp))
        print('Train net_y2h: [epoch %d/%d] train_loss:%f loss1:%f loss2:%f Time:%.4f' % (epoch+1, epochs, train_loss, train_loss_1, train_loss_2, timeit.default_timer()-start_tmp))
    #end for epoch

    return net_y2h
