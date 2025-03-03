import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import entropy_model
from conditional_entropy_model import ConditionalEntropyBottleneck
from ops import GDN
import math
from channel import Channel


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

# handle multiple input
class MySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs



class Enc(nn.Module):
    def __init__(self, num_features, M1, M, N2):
        super(Enc, self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.M = int(M)
        self.N2 = int(N2)

        # main encoder
        self.conv1 = nn.Conv2d(self.n_features, self.M1, kernel_size=5, stride=2, padding=2)
        #self.gdn1 = gdn.GDN(self.M1)
        self.norm1 = LayerNorm2d(self.M1)
        self.gdn1 = nn.ReLU()
        self.CA1 =  self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.M1, out_channels=self.M1, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.conv2 = nn.Conv2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2)
        # self.gdn2 = gdn.GDN(self.M1)
        self.norm2 = LayerNorm2d(self.M1)
        self.gdn2 = nn.ReLU()
        self.CA2 = self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.M1, out_channels=self.M1, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv3 = nn.Conv2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2)
        # self.gdn3 = gdn.GDN(self.M1)
        self.norm3 = LayerNorm2d(self.M1)
        self.gdn3 = nn.ReLU()
        self.CA3 = self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.M1, out_channels=self.M1, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv4 = nn.Conv2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2)
        # self.gdn4 = gdn.GDN(self.M1)
        self.norm4 = LayerNorm2d(self.M1)
        self.gdn4 = nn.ReLU()
        self.CA4 = self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.M1, out_channels=self.M1, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv5 = nn.Conv2d(self.M1, self.N2, kernel_size=3, stride=1, padding=1)
        # hyper encoder
        # self.conv1_hy = nn.Conv2d(self.N, self.M1, kernel_size=5, stride=2, padding=2)
        # self.conv2_hy = nn.Conv2d(self.M1, self.M1*2, kernel_size=5, stride=2, padding=2)

    def main_enc(self, x):
        x1 = self.conv1(x)
        x1 = self.norm1(x1)
        x1 = self.gdn1(x1)
        x1c = self.CA1(x1)
        x1 = x1 * x1c
        x2 = self.conv2(x1)
        x2 = self.norm2(x2)
        x2 = self.gdn2(x2)
        x2c = self.CA2(x2)
        x2 = x2 * x2c
        x3 = self.conv3(x2)
        x3 = self.norm3(x3)
        x3 = self.gdn3(x3)
        x3c = self.CA3(x3)
        x3 = x3 * x3c
        # x4 = self.conv4(x3)
        # x4 = self.norm4(x4)
        # x4 = self.gdn4(x4)
        # x4c = self.CA4(x4)
        # x4 = x4 * x4c

        x5 = self.conv5(x3)

        return x5

    def forward(self, x):
        enc = self.main_enc(x)
        return enc

class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''

    def __init__(self, c, N):
        super().__init__()
        self.scale = c ** -0.5
        self.N = N
        self.conv1r = nn.Conv2d(self.N, c, kernel_size=3, stride=1, padding=1)
        self.conv1l = nn.Conv2d(self.N, c, kernel_size=3, stride=1, padding=1)

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r, if_first):
        if if_first == 1:
            x_l = self.conv1l(x_l)
            x_r = self.conv1r(x_r)
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(0, 2, 1, 3)  # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l)  # B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r

class Hyper_Enc(nn.Module):
    def __init__(self, N2, M):
        super(Hyper_Enc, self).__init__()
        self.M = int(M)
        self.N2 = int(N2)
        # hyper encoder
        self.conv1 = nn.Conv2d(self.N2, self.M, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(self.M, self.M, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(self.M, self.N2, kernel_size=5, stride=2, padding=2)
        self.relu = nn.ReLU()

    def forward(self, xq):
        xq = torch.abs(xq)
        x1 = self.conv1(xq)
        x1 = self.relu(x1)
        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        x3 = self.conv3(x2)
        return x3

class Dyna_Enc(nn.Module):
    def __init__(self, M, C, rate_choice=5120):
        super(Dyna_Enc, self).__init__()
        self.M = M
        self.C = C
        self.weight = nn.Parameter(torch.zeros(self.C, rate_choice))
        self.bias = nn.Parameter(torch.zeros(rate_choice))
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.C)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, px):
        B, C, H, W = x.size()
        x_BLC = x.flatten(2).permute(0, 2, 1)
        w = self.weight.reshape(H * W, C, -1)
        b = self.bias.reshape(H * W, -1)
        x_BLC_masked = (torch.matmul(x_BLC.unsqueeze(2), w).squeeze() + b)
        x4 = torch.reshape(x_BLC_masked, shape=(B, H, W, -1)).permute(0, 3, 1, 2)

        return x4

class Dyna_Dec(nn.Module):
    def __init__(self, M, C, rate_choice=5120):
        super(Dyna_Dec, self).__init__()
        self.M = M
        self.C = C
        self.weight = nn.Parameter(torch.zeros(rate_choice, self.C))
        self.bias = nn.Parameter(torch.zeros(rate_choice))
        torch.nn.init.kaiming_normal_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.C)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, px):
        B, C, H, W = x.size()
        x_BLC = x.flatten(2).permute(0, 2, 1)
        w = self.weight.reshape(H * W, -1, C)
        b = self.bias.reshape(H * W, C)
        x_BLC = torch.matmul(x_BLC.unsqueeze(2), w).squeeze() + b
        x4 = x_BLC.reshape(B, H, W, -1).permute(0, 3, 1, 2)

        return x4


class Hyper_Dec(nn.Module):
    def __init__(self, N2, M):
        super(Hyper_Dec, self).__init__()
        self.M = int(M)
        self.N2 = int(N2)
        # hyper decoder
        self.conv1 = nn.ConvTranspose2d(self.N2, self.M, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(self.M, self.M, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(self.M, self.N2, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, xq2):
        x1 = self.conv1(xq2)
        x1 = self.relu(x1)
        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        x3 = self.conv3(x2)
        x3 = self.relu(x3)
        return x3


class Dec(nn.Module):
    def __init__(self, num_features, M1, N):
        super(Dec, self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.N = int(N)

        # main decoder
        self.conv1 = nn.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2,
                                        padding=2, output_padding=1)
        self.norm1 = LayerNorm2d(self.M1)
        #self.gdn1 = gdn.GDN(self.M1, inverse=True)
        self.gdn1 = nn.ReLU()
        self.CA1 = self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.M1, out_channels=self.M1, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv2 = nn.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.norm2 = LayerNorm2d(self.M1)
        #self.gdn2 = gdn.GDN(self.M1, inverse=True)
        self.gdn2 = nn.ReLU()
        self.CA2 = self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.M1, out_channels=self.M1, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv3 = nn.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.norm3 = LayerNorm2d(self.M1)
        #self.gdn3 = gdn.GDN(self.M1, inverse=True)
        self.gdn3 = nn.ReLU()
        self.CA3 = self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.M1, out_channels=self.M1, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv4 = nn.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.norm4 = LayerNorm2d(self.M1)
        #self.gdn4 = gdn.GDN(self.M1, inverse=True)
        self.gdn4 = nn.ReLU()
        self.CA4 = self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.M1, out_channels=self.M1, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv5 = nn.Conv2d(self.M1, 3, kernel_size=3, stride=1, padding=1)

        # decoder_r
        self.conv1r = nn.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2,
                                        padding=2, output_padding=1)
        self.norm1r = LayerNorm2d(self.M1)
        # self.gdn1 = gdn.GDN(self.M1, inverse=True)
        self.gdn1r = nn.ReLU()
        self.CA1r = self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.M1, out_channels=self.M1, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv2r = nn.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.norm2r = LayerNorm2d(self.M1)
        # self.gdn2r = gdn.GDN(self.M1, inverse=True)
        self.gdn2r = nn.ReLU()
        self.CA2r = self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.M1, out_channels=self.M1, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv3r = nn.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.norm3r = LayerNorm2d(self.M1)
        # self.gdn3r = gdn.GDN(self.M1, inverse=True)
        self.gdn3r = nn.ReLU()
        self.CA3r = self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.M1, out_channels=self.M1, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv4r = nn.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.norm4r = LayerNorm2d(self.M1)
        # self.gdn4r = gdn.GDN(self.M1, inverse=True)
        self.gdn4r = nn.ReLU()
        self.CA4r = self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.M1, out_channels=self.M1, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv5r = nn.Conv2d(self.M1, 3, kernel_size=3, stride=1, padding=1)

        self.scam1 = SCAM(self.M1, self.N)
        self.scam2 = SCAM(self.M1, self.N)
        self.scam3 = SCAM(self.M1, self.N)

    def forward(self, x_l, x_r):
        x1 = self.conv1(x_l)
        x1 = self.norm1(x1)
        x1 = self.gdn1(x1)
        x1c = self.CA1(x1)
        x1 = x1 * x1c
        x1r = self.conv1r(x_r)
        x1r = self.norm1r(x1r)
        x1r = self.gdn1r(x1r)
        x1cr = self.CA1r(x1r)
        x1r = x1r * x1cr

        x1, x1r = self.scam1(x1, x1r, 0)

        x2 = self.conv2(x1)
        x2 = self.norm2(x2)
        x2 = self.gdn2(x2)
        x2c = self.CA2(x2)
        x2 = x2 * x2c
        x2r = self.conv2r(x1r)
        x2r = self.norm2r(x2r)
        x2r = self.gdn2r(x2r)
        x2cr = self.CA2r(x2r)
        x2r = x2r * x2cr

        x2, x2r = self.scam2(x2, x2r, 0)

        x3 = self.conv3(x2)
        x3 = self.norm3(x3)
        x3 = self.gdn3(x3)
        x3c = self.CA3(x3)
        x3 = x3 * x3c
        x3r = self.conv3r(x2r)
        x3r = self.norm3r(x3r)
        x3r = self.gdn3r(x3r)
        x3cr = self.CA3r(x3r)
        x3r = x3r * x3cr

        x3, x3r = self.scam3(x3, x3r, 0)

        # x4 = self.conv4(x3)
        # x4 = self.norm4(x4)
        # x4 = self.gdn4(x4)
        # x4c = self.CA4(x4)
        # x4 = x4 * x4c
        # x4r = self.conv4r(x3r)
        # x4r = self.norm4r(x4r)
        # x4r = self.gdn4r(x4r)
        # x4cr = self.CA4r(x4r)
        # x4r = x4r * x4cr

        x5_l = self.conv5(x3)
        x5_r = self.conv5r(x3r)

        return x5_l, x5_r

class Hyper_Fusion(nn.Module):
    def __init__(self, M):
        super(Hyper_Fusion, self).__init__()
        self.M = int(M)
        self.Wq = nn.Parameter(torch.zeros(self.M, self.M))
        self.Wv = nn.Parameter(torch.zeros(self.M, self.M))
        self.Wk = nn.Parameter(torch.zeros(self.M, self.M))
        self.Wo = nn.Parameter(torch.zeros(self.M, self.M))
        torch.nn.init.kaiming_normal_(self.Wq, a=math.sqrt(5))
        torch.nn.init.kaiming_normal_(self.Wv, a=math.sqrt(5))
        torch.nn.init.kaiming_normal_(self.Wk, a=math.sqrt(5))
        torch.nn.init.kaiming_normal_(self.Wo, a=math.sqrt(5))
        self.ln1 = nn.LayerNorm(self.M)
        self.ln2 = nn.LayerNorm(self.M)
        self.softmax = nn.Softmax()

    def forward(self, x_l, x_r):
        B, C, H, W = x_l.size()
        x_l = torch.reshape(x_l, shape=(B, C, H * W))
        x_r = torch.reshape(x_r, shape=(B, C, H * W))
        x_l = x_l.permute(0, 2, 1)  # B, H*W, C
        x_r = x_r.permute(0, 2, 1)  # B, H*W, C
        x_ln_l = self.ln1(x_l)
        x_ln_r = self.ln2(x_r)
        #x_ln_l = x_l
        #x_ln_r = x_r
        Q_l = torch.matmul(x_ln_l, self.Wq)
        Q_r = torch.matmul(x_ln_r, self.Wq)
        K_l = torch.matmul(x_ln_l, self.Wk)
        K_r = torch.matmul(x_ln_r, self.Wk)
        V_l = torch.matmul(x_ln_l, self.Wv)
        V_r = torch.matmul(x_ln_r, self.Wv)

        A_l = torch.matmul(Q_l, K_r.transpose(1, 2))
        A_r = torch.matmul(Q_r, K_l.transpose(1, 2))

        out_l = torch.matmul(self.softmax(A_l/math.sqrt(C)), V_l)
        out_l = torch.matmul(out_l, self.Wo)
        out_l = x_l + out_l

        out_r = torch.matmul(self.softmax(A_r / math.sqrt(C)), V_r)
        out_r = torch.matmul(out_r, self.Wo)
        out_r = x_r + out_r

        out_l = out_l.permute(0, 2, 1)
        out_r = out_r.permute(0, 2, 1)
        out_l = torch.reshape(out_l, shape=(B, C, H, W))
        out_r = torch.reshape(out_r, shape=(B, C, H, W))

        return out_l, out_r
class Fusion(nn.Module):
    def __init__(self, N):
        super(Fusion, self).__init__()
        self.N = N
        self.M = int(N)
        self.Wq = nn.Parameter(torch.zeros(self.M, self.M))
        self.Wv = nn.Parameter(torch.zeros(self.M, self.M))
        self.Wk = nn.Parameter(torch.zeros(self.M, self.M))
        self.Wo = nn.Parameter(torch.zeros(self.M, self.M))
        torch.nn.init.kaiming_normal_(self.Wq, a=math.sqrt(5))
        torch.nn.init.kaiming_normal_(self.Wv, a=math.sqrt(5))
        torch.nn.init.kaiming_normal_(self.Wk, a=math.sqrt(5))
        torch.nn.init.kaiming_normal_(self.Wo, a=math.sqrt(5))
        self.ln1 = nn.LayerNorm(self.M)
        self.ln2 = nn.LayerNorm(self.M)
        self.softmax = nn.Softmax()
        self.conv1r = nn.Conv1d(self.N, 256, kernel_size=3, stride=1, padding=1)
        self.conv1l = nn.Conv1d(self.N, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x_l, x_r):
        B, C, H, W = x_l.size()
        x_l = torch.reshape(x_l, shape=(B, C, H * W))
        x_r = torch.reshape(x_r, shape=(B, C, H * W))
        #x_l = self.conv1l(x_l)
        #x_r = self.conv1r(x_r)
        B, C2, H2 = x_l.size()
        x_l = x_l.permute(0, 2, 1)  # B, H*W, C
        x_r = x_r.permute(0, 2, 1)  # B, H*W, C
        x_ln_l = self.ln1(x_l)
        x_ln_r = self.ln2(x_r)
        #x_ln_l = x_l
        #x_ln_r = x_r
        Q_l = torch.matmul(x_ln_l, self.Wq)
        Q_r = torch.matmul(x_ln_r, self.Wq)
        K_l = torch.matmul(x_ln_l, self.Wk)
        K_r = torch.matmul(x_ln_r, self.Wk)
        V_l = torch.matmul(x_ln_l, self.Wv)
        V_r = torch.matmul(x_ln_r, self.Wv)

        A_l = torch.matmul(Q_l, K_r.transpose(1, 2))
        A_r = torch.matmul(Q_r, K_l.transpose(1, 2))

        out_l = torch.matmul(self.softmax(A_l/math.sqrt(C)), V_l)
        out_l = torch.matmul(out_l, self.Wo)
        out_l = x_l + out_l

        out_r = torch.matmul(self.softmax(A_r / math.sqrt(C)), V_r)
        out_r = torch.matmul(out_r, self.Wo)
        out_r = x_r + out_r

        out_l = out_l.permute(0, 2, 1)
        out_r = out_r.permute(0, 2, 1)
        out_l = torch.reshape(out_l, shape=(B, C2, H, W))
        out_r = torch.reshape(out_r, shape=(B, C2, H, W))

        return out_l, out_r

class Scaler(nn.Module):
    def __init__(self, channels):
        super(Scaler, self).__init__()
        self.bias = nn.Parameter(torch.zeros([1, channels, 1, 1]))
        self.factor = nn.Parameter(torch.ones([1, channels, 1, 1]))

    def compress(self, x):
        return self.factor * (x - self.bias)

    def decompress(self, x):
        return self.bias + x / self.factor

lower_bound = entropy_model.lower_bound_fn.apply

class Image_encdec(nn.Module):
    def __init__(self, M, N2, num_features=3, M1=256):
        super(Image_encdec, self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.M = int(M)
        self.N2 = int(N2)
        self.enc = Enc(num_features, self.M1, self.M, self.N2)
        self.factorized_entropy_func = entropy_model.EntropyBottleneck(self.N2)
        self.hyper_enc = Hyper_Enc(N2, M)
        self.hyper_dec = Hyper_Dec(N2, M)
        self.conditional_entropy_bottleneck = ConditionalEntropyBottleneck()

        self.bound = 0.11
    def add_noise(self, x):
        noise = np.random.uniform(-0.5, 0.5, x.size())
        noise = torch.Tensor(noise).cuda()
        return x + noise

    def forward(self, x, if_training):
        y_main = self.enc(x)
        y_hyper = self.hyper_enc(y_main)

        y_hyper_q, p_hyper = self.factorized_entropy_func(y_hyper)

        gaussian_params = self.hyper_dec(y_hyper_q)
        sigma_lower_bounded = lower_bound(gaussian_params, self.bound)
        # if if_training:
        #     y_main_q = self.add_noise(y_main)
        # else:
        #     y_main_q = torch.round(y_main)
        y_main_q, p_main = self.conditional_entropy_bottleneck(y_main, sigma_lower_bounded)

        # output = self.decoder(y_main_q)
        # y_main = self.encoder(x)
        # output = self.decoder(y_main)
        return y_main, p_main, p_hyper, y_main_q, gaussian_params

class Image_coding(nn.Module):
    def __init__(self, M, N2, num_features=3, M1=256):
        super(Image_coding, self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.M = int(M)
        self.N2 = int(N2)
        self.enc_l = Image_encdec(M, N2, num_features, M1)
        self.enc_r = Image_encdec(M, N2, num_features, M1)
        self.JSCCencl = Dyna_Enc(M, N2)
        self.JSCCencr = Dyna_Enc(M, N2)
        self.JSCCdec = Dyna_Dec(M, N2)
        self.channel = Channel(5)
        self.decoder = Dec(num_features, self.M, self.N2)
        self.fusion = SCAM(self.M1, self.N2)

        self.hyper_fusion = Hyper_Fusion(self.N2*3)

        #self.mlp1 = nn.Linear(20*16*16*5, 20*16*16)
        #self.mlp2 = nn.Linear(20*16*16*5, 20*16*16)

    def add_noise(self, x):
        noise = np.random.uniform(-0.5, 0.5, x.size())
        noise = torch.Tensor(noise).cuda()
        return x + noise

    def forward(self, x_l, x_r, if_training):
        y_main_l, p_main_l, p_hyper_l, y_main_q_l, gaussian_params_l = self.enc_l(x_l, if_training)
        y_main_r, p_main_r, p_hyper_r, y_main_q_r, gaussian_params_r = self.enc_l(x_r, if_training)

        # 对每个元素进行熵掩码
        # m_l = torch.log(p_main_l) / (-math.log(2)) * 10
        # m_r = torch.log(p_main_r) / (-math.log(2)) * 10
        # mask_l = torch.ones_like(m_l)
        # mask_r = torch.ones_like(m_r)
        # mask_l[m_l < 0.0001] = 0
        # mask_r[m_r < 0.0001] = 0
        # num_l = np.sum((m_l < 0.0001).cpu().numpy())

        # y = torch.log(p_main_l) / (-math.log(2))
        #y = p_main_l
        # _,C,_,_ = y.size()
        # y1 = y[0,:,0,0].detach().cpu().numpy()
        # y2 = y[0,:,1,3].detach().cpu().numpy()
        # x = [i for i in range(C)]
        # x2 = [i+0.5 for i in range(C)]
        # plt.Figure()
        # plt.plot(x,y1,color='blue',marker='*')
        # plt.plot(x,y2,color='red',marker='^0')
        # plt.xlabel('i-th channel')
        # plt.ylabel('Entropy')
        # plt.legend(['pixel1', 'pixel2'])
        # plt.show()

        # 对通道元素进行熵掩码判断
        m_l = torch.sum(torch.log(p_main_l) / (-math.log(2)) * 10, dim=1)
        m_r = torch.sum(torch.log(p_main_r) / (-math.log(2)) * 10, dim=1)
        B, H, W = m_l.size()

        mask_l = torch.ones_like(m_l)
        mask_r = torch.ones_like(m_r)
        mask_l[m_l < 75] = 0
        mask_r[m_r < 75] = 0
        mask_l = mask_l.reshape((B, 1, H, W))
        mask_r = mask_r.reshape((B, 1, H, W))
        num_l = np.sum((m_l < 75).cpu().numpy())

        # y_main_l = y_main_l * mask_l
        # y_main_r = y_main_r * mask_r
        #s_l = self.JSCCencl(y_main_l, p_main_l)
        s_l,_ = self.channel(y_main_l)
        #s_l = self.JSCCdec(s_l, p_main_l)

        #s_r = self.JSCCencr(y_main_r, p_main_r)
        s_r, _ = self.channel(y_main_r)
        #s_r = self.JSCCdec(s_r, p_main_r


        s_mask_l = s_l * mask_l
        s_mask_r = s_r * mask_r
        # s_mask_l = s_l
        # s_mask_r = s_r


        # 双视角特征信息融合
        s_fusion_l, s_fusion_r = self.fusion(s_mask_l, s_mask_r, 1)
        output_l, output_r = self.decoder(s_fusion_l, s_fusion_r)

        #return output_l, output_r, channel_usage_l, channel_usage_r, p_main_l, p_hyper_l, p_main_r, p_hyper_r, symbol_l, symbol_r
        #return output_l, output_r
        return output_l, output_r, p_main_l, p_hyper_l, p_main_r, p_hyper_r, num_l
