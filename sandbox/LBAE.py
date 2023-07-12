__author__ = 'Jiri Fajtl'
__email__ = 'ok1zjf@gmail.com'
__version__= '1.8'
__status__ = "Research"
__date__ = "2/1/2020"
__license__= "MIT License"

import torch
import torch.nn as nn

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()

def weight_init(model, mean=0, std=0.02):
    for m in model._modules:
        normal_init(model._modules[m], mean, std)
    return

#===========================================================================================
class QuantizerFunc(torch.autograd.Function):
    @staticmethod
    def forward(self, input, npoints=4, dropout=0):
        # self.save_for_backward(input)
        # self.constant = npoints
        if npoints < 0:
            x = torch.sign(input)
            x[x==0] = 1
            return x

        scale = 10**npoints
        input = input * scale
        input = torch.round(input)
        input = input / scale
        return input

    @staticmethod
    def backward(self, grad_output):
        # input, = self.saved_tensors
        grad_input = grad_output.clone()
        # grad_input[input < 0] = 0
        # grad_input[:] = 1
        return grad_input, None


class Quantizer(nn.Module):
    def __init__(self, npoints=3):
        super().__init__()
        self.npoints = npoints
        self.quant = QuantizerFunc.apply

    def forward(self,x):
        x = self.quant(x, self.npoints)
        return x

#===========================================================================================
class BinDropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p
        self.mask = None

    def forward(self, x):
        if self.training:
            if self.mask is None:
                po = int(self.p * x.size(1))
                self.mask = 1
                self.bdist = torch.distributions.Bernoulli(self.p)

            m = self.bdist.sample(x.size())*-2+1 
            x = x*m.cuda()
            return x
        return x

#===========================================================================================
class EncConvResBlock32(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size

        # encoder layers; will do ~3 toy FC layers
        self.ff_0 = nn.Linear(input_size, 512)
        self.ff_1 = nn.Linear(512, 256)
        self.ff_2 = nn.Linear(256, 128)

        self.quant = QuantizerFunc.apply

        self.act = nn.LeakyReLU(0.02)
        self.drop = None

    def forward(self, x):

        x = self.ff_0(x)
        x = self.act(x)

        x = self.ff_1(x)
        x = self.act(x)

        x = self.ff_2(x)
        x = self.act(x)


        # tahn for the bit quant
        x = torch.tahn(x)
        xq = self.quant(x)
        err_quant = torch.abs(x - xq)

        x = xq

        xe = x if xe is None else xe
        diff = ((x+xe) == 0).sum(1) 
        return x, xe, diff, err_quant.sum()/(x.size(0))


#===========================================================================================
class GenConvResBlock32(nn.Module):
    def __init__(self, hps):
        super().__init__()


        self.act = nn.LeakyReLU(0.02)
        self.quant = QuantizerFunc.apply

        if self.hps.img_size == 64:
            self.in_channels = self.dc1.in_channels
        else:
            self.in_channels = self.dc2.in_channels

        self.fmres = 4 
        out_size = self.in_channels*self.fmres*self.fmres
        bias = True
        self.l1l=nn.Linear(self.hps.zsize, out_size, bias=bias)
        

    def forward(self, x, sw=None):
        x = x.view(x.size(0), -1)

        x = self.l1l(x)
        x = x.view(x.size(0), self.in_channels,self.fmres,self.fmres)


        x = torch.sigmoid(x)
        return x

#=================================================================================
if __name__ == "__main__":
    print("NOT AN EXECUTABLE!")