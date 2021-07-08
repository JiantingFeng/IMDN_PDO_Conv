import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from math import sin, cos, pi

# preparation
PARTIAL_DICT = torch.tensor([[[0,0,0,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,1/2,0,0],[0,0,0,0,0],[0,0,-1/2,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,0,0,0],[0,1,-2,1,0],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,-1/4,0,1/4,0],[0,0,0,0,0],[0,1/4,0,-1/4,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,1,0,0],[0,0,-2,0,0],[0,0,1,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,0,0,0,0],[-1/2,1,0,-1,1/2],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,1/2,-1,1/2,0],[0,0,0,0,0],[0,-1/2,1,-1/2,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,1,0,-1,0],[0,-1/2,0,1/2,0],[0,0,0,0,0]],
                    [[0,0,1/2,0,0],[0,0,-1,0,0],[0,0,0,0,0],[0,0,1,0,0],[0,0,-1/2,0,0]],
                    [[0,0,0,0,0],[0,0,0,0,0],[1,-4,6,-4,1],[0,0,0,0,0],[0,0,0,0,0]],
                    [[0,0,0,0,0],[-1/4,1/2,0,-1/2,1/4],[0,0,0,0,0],[1/4,-1/2,0,1/2,-1/4],[0,0,0,0,0]],
                    [[0,0,0,0,0],[0,1,-2,1,0],[0,-2,4,-2,0],[0,1,-2,1,0],[0,0,0,0,0]],
                    [[0,-1/4,0,1/4,0],[0,1/2,0,-1/2,0],[0,0,0,0,0],[0,-1/2,0,1/2,0],[0,1/4,0,-1/4,0]],
                    [[0,0,1,0,0],[0,0,-4,0,0],[0,0,6,0,0],[0,0,-4,0,0],[0,0,1,0,0]]])
PARTIAL_DICT = PARTIAL_DICT.cuda()

P_NUM = 8
GROUP_ANGLE = [2*k*pi/P_NUM + pi/8 for k in range(P_NUM)]
TRAN_TO_PARTIAL_COEF = [torch.tensor([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                     [0,cos(x),sin(x),0,0,0,0,0,0,0,0,0,0,0,0],
                                     [0,-sin(x),cos(x),0,0,0,0,0,0,0,0,0,0,0,0],
                                     [0,0,0,pow(cos(x),2),2*cos(x)*sin(x),pow(sin(x),2),0,0,0,0,0,0,0,0,0],
                                     [0,0,0,-cos(x)*sin(x),pow(cos(x),2)-pow(sin(x),2),sin(x)*cos(x),0,0,0,0,0,0,0,0,0],
                                     [0,0,0,pow(sin(x),2),-2*cos(x)*sin(x),pow(cos(x),2),0,0,0,0,0,0,0,0,0],
                                     [0,0,0,0,0,0,-pow(cos(x),2)*sin(x),pow(cos(x),3)-2*cos(x)*pow(sin(x),2),-pow(sin(x),3)+2*pow(cos(x),2)*sin(x), pow(sin(x),2)*cos(x),0,0,0,0,0],
                                     [0,0,0,0,0,0,cos(x)*pow(sin(x),2),-2*pow(cos(x),2)*sin(x)+pow(sin(x),3),pow(cos(x),3)-2*cos(x)*pow(sin(x),2),sin(x)*pow(cos(x),2),0,0,0,0,0],
                                     [0,0,0,0,0,0,0,0,0,0,pow(sin(x),2)*pow(cos(x),2),-2*pow(cos(x),3)*sin(x)+2*cos(x)*pow(sin(x),3),pow(cos(x),4)-4*pow(cos(x),2)*pow(sin(x),2)+pow(sin(x),4),-2*cos(x)*pow(sin(x),3)+2*pow(cos(x),3)*sin(x),pow(sin(x),2)*pow(cos(x),2)]]) for x in GROUP_ANGLE]
TRAN_TO_PARTIAL_COEF = [a.cuda() for a in TRAN_TO_PARTIAL_COEF]

TRANSFORMATION = PARTIAL_DICT[[0, 1, 2, 3, 4, 5, 7, 8, 12], 1:4, 1:4]
TRANSFORMATION = TRANSFORMATION.view([9, 9])
INV_TRANSFORMATION = TRANSFORMATION.inverse()
INV_TRANSFORMATION = INV_TRANSFORMATION.cuda()

def get_coef(weight):
    in_channels, out_channels, k1, k2 = weight.size()
    assert(k1 == 3 and k2 == 3)

    betas = weight.view(-1, 9)
    betas = torch.mm(betas, INV_TRANSFORMATION)
    betas = betas.view(in_channels, out_channels, 9)
    return betas


# Convolutions
# in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True
class OpenConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, padding, stride=1, dilation=1, groups=1, bias=True):
        super(OpenConv2d, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.weight = nn.Parameter(torch.Tensor(self.in_channels, self.out_channels, 3, 3))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))

    def forward(self, x):
        betas = get_coef(self.weight)
        og_coef = betas.view(-1, 9) # (I*O, 9)
        partial_coef = [torch.mm(og_coef, a) for a in TRAN_TO_PARTIAL_COEF] # P, (I*O, 15)
        partial = PARTIAL_DICT.view(15, 25)
        kernel = [torch.mm(a, partial) for a in partial_coef] # P, (I*O, 25)
        kernel = torch.stack(kernel, dim=1) # (I*O, P*25)
        kernel = kernel.view(self.in_channels, self.out_channels * P_NUM, 5, 5) # (I, O*P, 5, 5)
        kernel = kernel.transpose(0, 1) # (O*P, I, 5, 5)
        input_size = x.size()
        x = x.view(input_size[0], self.in_channels, input_size[-2], input_size[-1])
        out =  F.conv2d(x, weight=kernel, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        batch_size, _, ny_out, nx_out = out.size()
        out = out.view(batch_size, self.out_channels, P_NUM, ny_out, nx_out)
        return out


# in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True
class GConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, padding, kernel_size=3, stride=1, dilation=1, groups=1, bias=True):
        super(GConv2d, self).__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.padding = padding
        self.kernel_size, self.stride = kernel_size, stride
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        assert(kernel_size in [1, 3])
        self.weight = nn.Parameter(torch.Tensor(self.in_channels * P_NUM, self.out_channels, kernel_size, kernel_size))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5.0))

    def forward(self, x):
        if self.kernel_size == 1:
            kernel = self.weight.view(self.in_channels, P_NUM, self.out_channels, 1, 1)
            kernel = [torch.cat([kernel[:, -k:], kernel[:, :-k]], dim=1) for k in range(P_NUM)] # P, (I, P, O, 1, 1)
            kernel = torch.stack(kernel, dim=3) # (I, P, O, P, 1, 1)
            kernel = kernel.view(self.in_channels * P_NUM, self.out_channels * P_NUM, 1, 1)
        else:
            betas = get_coef(self.weight)
            og_coef = betas.view(-1, 9) # (I*P*O, 9)
            partial_coef = [torch.mm(og_coef, a) for a in TRAN_TO_PARTIAL_COEF] # P, (I*P*O, 15)
            partial = PARTIAL_DICT.view(15, 25)
            kernel = [torch.mm(a, partial) for a in partial_coef] # P, (I*P*O, 25)
            kernel = [k.view(self.in_channels, P_NUM, self.out_channels, 25) for k in kernel] # P, (I, P, O, 25)
            kernel = [torch.cat([kernel[k][:, -k:, :], kernel[k][:, :-k, :]], dim=1) for k in range(P_NUM)] # P, (I, P, O, 25)
            kernel = torch.stack(kernel, dim=3) # (I, P, O, P, 25)
            kernel = kernel.view(self.in_channels * P_NUM, self.out_channels * P_NUM, 5, 5)
        kernel = kernel.transpose(0, 1) # (O*P, I*P, k, k)

        input_size = x.size()
        x = x.view(input_size[0], self.in_channels*P_NUM, input_size[-2], input_size[-1])
        out = F.conv2d(x, weight=kernel, bias=self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)

        batch_size, _, ny_out, nx_out = out.size()
        out = out.view(batch_size, self.out_channels, P_NUM, ny_out, nx_out)
        return out


class GUpsample2D(nn.Module):
    def __init__(self, scale_factor=2):
        super(GUpsample2D, self).__init__()
        self.scale_factor = scale_factor
        self.Upsample = nn.Upsample(scale_factor=self.scale_factor)

    def forward(self, x):
        [batch, channels, groups, height, width] = x.size()
        y = torch.zeros([batch, channels, groups, self.scale_factor * height, self.scale_factor * width]).cuda()
        for group in range(groups):
            y[:, :, group, :, :] = self.Upsample(x[:, :, group, :, :])
        return y


class GMaxPool2D(nn.Module):
    def __init__(self, kernel_size=2):
        super(GMaxPool2D, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size == 2
        self.MaxPool2D = nn.MaxPool2d(kernel_size=self.kernel_size)

    def forward(self, x):
        [batch, channels, groups, height, width] = x.size()
        y = torch.zeros([batch, channels, groups, int(height/self.kernel_size), int(width/self.kernel_size)]).cuda()
        for group in range(groups):
            y[:, :, group, :, :] = self.MaxPool2D(x[:, :, group, :, :])
        return y


class GAvgPool2D(nn.Module):
    def __init__(self, kernel_size=2):
        super(GAvgPool2D, self).__init__()
        self.kernel_size = kernel_size
        assert self.kernel_size == 2
        self.AvgPool2d = nn.AvgPool2d(kernel_size=self.kernel_size)

    def forward(self, x):
        [batch, channels, groups, height, width] = x.size()
        y = torch.zeros([batch, channels, groups, int(height/self.kernel_size), int(width/self.kernel_size)]).cuda()
        for group in range(groups):
            y[:, :, group, :, :] = self.AvgPool2d(x[:, :, group, :, :])
        return y
