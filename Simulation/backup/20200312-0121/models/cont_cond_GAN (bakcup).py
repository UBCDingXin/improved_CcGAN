import torch
import torch.nn as nn

set_bias = False
#########################################################
# genearator
class cont_cond_generator(nn.Module):
    def __init__(self, ngpu=1, nz=2, out_dim=2):
        super(cont_cond_generator, self).__init__()
        self.nz = nz
        self.ngpu = ngpu
        self.out_dim = out_dim

        self.inner_dim = 100

        self.linear1 = nn.Sequential(
                nn.Linear(nz, self.inner_dim, bias=set_bias), #layer 1
                nn.ReLU(True),
                nn.Linear(self.inner_dim, self.inner_dim, bias=set_bias), #layer 2
                nn.ReLU(True),
            )

        self.linear2 = nn.Sequential(
                nn.Linear(self.inner_dim, self.inner_dim, bias=set_bias), #layer 3
                nn.ReLU(True),
                nn.Linear(self.inner_dim, self.out_dim, bias=set_bias), #layer 4
            )

    def forward(self, input, labels):
        input = input.view(-1, self.nz)
        labels_rep = labels.view(-1, 1).repeat(1,self.inner_dim)

        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.linear1, input, range(self.ngpu))
            output = output + labels_rep
            output = nn.parallel.data_parallel(self.linear2, output, range(self.ngpu))
        else:
            output = self.linear1(input)
            output = output + labels_rep
            output = self.linear2(output)
        return output

#########################################################
# discriminator
class cont_cond_discriminator(nn.Module):
    def __init__(self, ngpu=1, input_dim = 2):
        super(cont_cond_discriminator, self).__init__()
        self.ngpu = ngpu
        self.input_dim = input_dim

        self.inner_dim = 100
        self.main = nn.Sequential(
            nn.Linear(input_dim, self.inner_dim, bias=set_bias), #Layer 1
            nn.ReLU(True),
            nn.Linear(self.inner_dim, self.inner_dim, bias=set_bias), #layer 2
            nn.ReLU(True),
            nn.Linear(self.inner_dim, self.inner_dim, bias=set_bias), #layer 3
            nn.ReLU(True),
        )

        self.output = nn.Sequential(
            nn.Linear(self.inner_dim, 1, bias=set_bias), #layer 4
            nn.Sigmoid()
        )


    def forward(self, input, labels):
        labels_rep = labels.view(-1, 1).repeat(1,self.inner_dim)
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
            output = output + labels_rep
            output = nn.parallel.data_parallel(self.output, output, range(self.ngpu))
        else:
            output = self.main(input)
            output = output + labels_rep
            output = self.output(output)
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
