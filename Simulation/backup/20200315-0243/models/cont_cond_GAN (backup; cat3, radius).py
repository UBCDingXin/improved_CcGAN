import torch
import torch.nn as nn
import numpy as np

set_bias = False
#########################################################
# genearator
class cont_cond_generator(nn.Module):
    def __init__(self, ngpu=1, nz=2, out_dim=2, raidus=5):
        super(cont_cond_generator, self).__init__()
        self.nz = nz
        self.ngpu = ngpu
        self.out_dim = out_dim
        self.radius = radius

        self.inner_dim = 100

        self.linear = nn.Sequential(
                nn.Linear(nz+2, self.inner_dim, bias=set_bias),
                nn.BatchNorm1d(self.inner_dim),
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.inner_dim, bias=set_bias),
                nn.BatchNorm1d(self.inner_dim),
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.inner_dim, bias=set_bias),
                nn.BatchNorm1d(self.inner_dim),
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.inner_dim, bias=set_bias),
                nn.BatchNorm1d(self.inner_dim),
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.inner_dim, bias=set_bias),
                nn.BatchNorm1d(self.inner_dim),
                nn.ReLU(True),

                nn.Linear(self.inner_dim, self.out_dim, bias=set_bias),
            )

    def forward(self, input, labels, radius=5):
        input = input.view(-1, self.nz)
        labels = labels.view(-1, 1)*2*np.pi
        input = torch.cat((input, self.radius*torch.sin(labels), self.radius*torch.cos(labels)), 1)

        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.linear, input, range(self.ngpu))
        else:
            output = self.linear(input)
        return output

#########################################################
# discriminator
class cont_cond_discriminator(nn.Module):
    def __init__(self, ngpu=1, input_dim = 2, radius=5):
        super(cont_cond_discriminator, self).__init__()
        self.ngpu = ngpu
        self.input_dim = input_dim
        self.radius = radius

        self.inner_dim = 100
        self.main = nn.Sequential(
            nn.Linear(input_dim+2, self.inner_dim, bias=set_bias),
            nn.BatchNorm1d(self.inner_dim),
            nn.ReLU(True),

            nn.Linear(self.inner_dim, self.inner_dim, bias=set_bias),
            nn.BatchNorm1d(self.inner_dim),
            nn.ReLU(True),

            nn.Linear(self.inner_dim, self.inner_dim, bias=set_bias),
            nn.BatchNorm1d(self.inner_dim),
            nn.ReLU(True),

            nn.Linear(self.inner_dim, self.inner_dim, bias=set_bias),
            nn.BatchNorm1d(self.inner_dim),
            nn.ReLU(True),

            nn.Linear(self.inner_dim, self.inner_dim, bias=set_bias),
            nn.BatchNorm1d(self.inner_dim),
            nn.ReLU(True),

            nn.Linear(self.inner_dim, 1, bias=set_bias),
            nn.Sigmoid()
        )


    def forward(self, input, labels):
        input = input.view(-1, self.input_dim)
        labels = labels.view(-1, 1)*2*np.pi
        input = torch.cat((input, self.radius*torch.sin(labels), self.radius*torch.cos(labels)), 1)

        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output.view(-1, 1)


if __name__=="__main__":
    import numpy as np
    #test
    ngpu=1

    netG = cont_cond_generator(ngpu=ngpu, nz=2, out_dim=2).cuda()
    netD = cont_cond_discriminator(ngpu=ngpu, input_dim = 2).cuda()

    z = torch.randn(32, 2).cuda()
    y = np.random.randint(100, 300, 32)
    y = torch.from_numpy(y).type(torch.float).view(-1,1).cuda()
    x = netG(z,y)
    o = netD(x,y)
    print(y.size())
    print(x.size())
    print(o.size())
