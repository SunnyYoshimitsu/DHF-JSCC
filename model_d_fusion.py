import math

import torch
import torch.nn as nn
import numpy as np
from gaussian_entropy_model import Distribution_for_entropy2
from balle2017 import gdn
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


class Hyper_Enc(nn.Module):
    def __init__(self, N2, M):
        super(Hyper_Enc, self).__init__()
        self.M = int(M)
        self.N2 = int(N2)
        # hyper encoder
        self.conv1 = nn.Conv2d(self.M, self.N2, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(self.N2, self.N2, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(self.N2, self.N2, kernel_size=5, stride=2, padding=2)
        self.relu = nn.ReLU()

    def forward(self, xq):
        xq = torch.abs(xq)
        x1 = self.conv1(xq)
        x1 = self.relu(x1)
        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        x3 = self.conv3(x2)
        return x3


class Hyper_Dec(nn.Module):
    def __init__(self, N2, M):
        super(Hyper_Dec, self).__init__()
        self.M = int(M)
        self.N2 = int(N2)
        # hyper decoder
        self.conv1 = nn.ConvTranspose2d(self.N2, self.N2, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(self.N2, self.N2, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(self.N2, self.M * 2, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, xq2):
        x1 = self.conv1(xq2)
        x1 = self.relu(x1)
        x2 = self.conv2(x1)
        x2 = self.relu(x2)
        x3 = self.conv3(x2)
        x3 = self.relu(x3)
        return x3


class SCAM(nn.Module):
    '''
    Stereo Cross Attention Module (SCAM)
    '''

    def __init__(self, c):
        super().__init__()
        self.scale = c ** -0.5
        self.N = 15
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

        self.scam1 = SCAM(self.M1)
        self.scam2 = SCAM(self.M1)
        self.scam3 = SCAM(self.M1)

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

        #x1, x1r = self.scam1(x1, x1r, 0)

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

        #x2, x2r = self.scam2(x2, x2r, 0)

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

        #x3, x3r = self.scam3(x3, x3r, 0)

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

class Fusion(nn.Module):
    def __init__(self, N):
        super(Fusion, self).__init__()
        self.N = N
        self.M = int(256)
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
        x_l = self.conv1l(x_l)
        x_r = self.conv1r(x_r)
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


class Image_coding(nn.Module):
    def __init__(self, M, N2, num_features=3, M1=256):
        super(Image_coding, self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.M = int(M)
        self.N2 = int(N2)
        #self.M2 = int(16)
        self.encoder_l = Enc(num_features, self.M1, self.M, self.N2)
        self.encoder_r = Enc(num_features, self.M1, self.M, self.N2)
        self.decoder = Dec(num_features, self.M, self.N2)
        self.fusion = SCAM(self.M1)
        self.channel = Channel(5)

    def add_noise(self, x):
        noise = np.random.uniform(-0.5, 0.5, x.size())
        noise = torch.Tensor(noise).cuda()
        return x + noise

    def forward(self, x_l, x_r):
        y_main_l = self.encoder_l(x_l)
        y_main_r = self.encoder_r(x_r)
        #y_hyper = self.hyper_enc(y_main)
        #y_hyper_q, p_hyper = self.factorized_entropy_func(y_hyper, if_training)
        #gaussian_params = self.hyper_dec(y_hyper_q)
        # if if_training:
        #     y_main_q = self.add_noise(y_main)
        # else:
        #     y_main_q = torch.round(y_main)
        #p_main = self.gaussin_entropy_func(y_main_q, gaussian_params)

        #y_main_l = torch.reshape(y_main_l, shape=(-1, self.M * self.M2 * self.M2))
        #y_main_r = torch.reshape(y_main_r, shape=(-1, self.M * self.M2 * self.M2))
        #y_main = self.fc1(y_main)
        channel_input_l = y_main_l
        channel_input_r = y_main_r
        channel_output_l, channel_usage_l = self.channel.forward(channel_input_l)
        channel_output_r, channel_usage_r = self.channel.forward(channel_input_r)
        #y_rece = self.fc2(channel_output)
        #y_rece = torch.reshape(y_rece, shape=(-1, self.M, self.M2, self.M2))

        y_fusion_l, y_fusion_r = self.fusion(channel_output_l, channel_output_r, 1)
        output_l, output_r = self.decoder(y_fusion_l, y_fusion_r)

        #output_l, output_r = self.decoder(channel_output_l, channel_output_r) #seperated

        # y_main = self.encoder(x)
        # output = self.decoder(y_main)
        return output_l, output_r
