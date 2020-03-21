import torch
import torch.nn as nn

def one_hot_encode(labels, num_classes):
    ones = torch.sparse.torch.eye(num_classes)
    ones = ones.type(torch.float)
    if labels.is_cuda:
        ones = ones.cuda()
    return ones.index_select(0,labels)


#########################################################
# genearator
class cond_generator(nn.Module):
    def __init__(self, ngpu=1, nz=2, out_dim=2, num_classes=200):
        super(cond_generator, self).__init__()
        self.nz = nz
        self.ngpu = ngpu
        self.out_dim = out_dim
        self.num_classes = num_classes

        inner_dim = 100
        self.linear = nn.Sequential(
                nn.Linear(nz+num_classes, inner_dim), #layer 1
                nn.BatchNorm1d(inner_dim),
                nn.ReLU(True),

                nn.Linear(inner_dim, inner_dim), #layer 2
                nn.BatchNorm1d(inner_dim),
                nn.ReLU(True),

                nn.Linear(inner_dim, inner_dim), #layer 3
                nn.BatchNorm1d(inner_dim),
                nn.ReLU(True),

                nn.Linear(inner_dim, inner_dim), #layer 4
                nn.BatchNorm1d(inner_dim),
                nn.ReLU(True),

                nn.Linear(inner_dim, inner_dim), #layer 5
                nn.BatchNorm1d(inner_dim),
                nn.ReLU(True),

                nn.Linear(inner_dim, inner_dim), #layer 6
                nn.BatchNorm1d(inner_dim),
                nn.ReLU(True),

                nn.Linear(inner_dim, self.out_dim), #layer 7
        )

    def forward(self, input, labels):
        input = input.view(-1, self.nz)
        input = torch.cat((one_hot_encode(labels, self.num_classes), input), 1)

        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.linear, input, range(self.ngpu))
        else:
            output = self.linear(input)
        return output

#########################################################
# discriminator
class cond_discriminator(nn.Module):
    def __init__(self, ngpu=1, input_dim = 2, num_classes=200):
        super(cond_discriminator, self).__init__()
        self.ngpu = ngpu
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.inner_dim = 100
        self.main = nn.Sequential(
            nn.Linear(self.input_dim+num_classes, self.inner_dim), #Layer 1
            nn.ReLU(True),

            nn.Linear(self.inner_dim, self.inner_dim), #layer 2
            nn.ReLU(True),

            nn.Linear(self.inner_dim, self.inner_dim), #layer 3
            nn.ReLU(True),

            nn.Linear(self.inner_dim, self.inner_dim), #layer 4
            nn.ReLU(True),

            nn.Linear(self.inner_dim, self.inner_dim), #layer 5
            nn.ReLU(True),

            nn.Linear(self.inner_dim, 1), #layer 6
            nn.Sigmoid()

        )

    def forward(self, input, labels):
        input = torch.cat((one_hot_encode(labels, self.num_classes), input), 1)
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1)


if __name__=="__main__":
    import numpy as np
    #test
    ngpu=1

    netG = cond_generator(ngpu=ngpu, nz=2, out_dim=2, num_classes=200).cuda()
    netD = cond_discriminator(ngpu=ngpu, input_dim = 2, num_classes=200).cuda()

    z = torch.randn(5, 2).cuda()
    labels = np.random.choice(np.arange(200),size=5,replace=True)
    labels = torch.from_numpy(labels).type(torch.long).cuda()
    x = netG(z,labels)
    o = netD(x,labels)
    print(x.size())
    print(o.size())
