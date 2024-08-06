import math

import torch
import torch.nn as nn
import numpy as np
from factorized_entropy_model import Entropy_bottleneck
from gaussian_entropy_model import Distribution_for_entropy2
from basic_module import ResBlock, Non_local_Block
from fast_context_model import Context4
from balle2017 import gdn
from channel import Channel


class Enc_l(nn.Module):
    def __init__(self, num_features, M1, M, N2):
        super(Enc_l, self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.M = int(M)
        self.N2 = int(N2)

        # main encoder
        self.conv1 = nn.Conv2d(self.n_features, self.M1, kernel_size=5, stride=2, padding=2)
        #self.gdn1 = gdn.GDN(self.M1)
        self.gdn1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2)
        #self.gdn2 = gdn.GDN(self.M1)
        self.gdn2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2)
        #self.gdn3 = gdn.GDN(self.M1)
        self.gdn3 = nn.ReLU()
        self.conv3p = nn.Conv2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2)
        #self.gdn3p = gdn.GDN(self.M1)
        self.gdn3p = nn.ReLU()
        self.conv4 = nn.Conv1d(self.M1, self.N2, kernel_size=3, stride=1, padding=1)
        # hyper encoder
        # self.conv1_hy = nn.Conv2d(self.N, self.M1, kernel_size=5, stride=2, padding=2)
        # self.conv2_hy = nn.Conv2d(self.M1, self.M1*2, kernel_size=5, stride=2, padding=2)

    def main_enc(self, x):
        x1 = self.conv1(x)
        x1 = self.gdn1(x1)
        x2 = self.conv2(x1)
        x2 = self.gdn2(x2)
        x3 = self.conv3(x2)
        x3 = self.gdn3(x3)
        #x3 = self.conv3p(x3)
        #x3 = self.gdn3p(x3)

        B,C,H,W = x3.size()
        x3 = torch.reshape(x3, shape=(B, C, H*W))
        x4 = self.conv4(x3)
        B,C2,H2= x4.size()
        x4 = torch.reshape(x4, shape=(B, C2, H, W))
        return x4

    def forward(self, x):
        enc = self.main_enc(x)
        return enc

class Enc_r(nn.Module):
    def __init__(self, num_features, M1, M, N2):
        super(Enc_r, self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.M = int(M)
        self.N2 = int(N2)

        # main encoder
        self.conv1 = nn.Conv2d(self.n_features, self.M1, kernel_size=5, stride=2, padding=2)
        #self.gdn1 = gdn.GDN(self.M1)
        self.gdn1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2)
        # self.gdn2 = gdn.GDN(self.M1)
        self.gdn2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2)
        # self.gdn3 = gdn.GDN(self.M1)
        self.gdn3 = nn.ReLU()

        self.conv3p = nn.Conv2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2)
        # self.gdn3p = gdn.GDN(self.M1)
        self.gdn3p = nn.ReLU()

        self.conv4 = nn.Conv1d(self.M1, self.N2, kernel_size=3, stride=1, padding=1)

        # hyper encoder
        # self.conv1_hy = nn.Conv2d(self.N, self.M1, kernel_size=5, stride=2, padding=2)
        # self.conv2_hy = nn.Conv2d(self.M1, self.M1*2, kernel_size=5, stride=2, padding=2)

    def main_enc(self, x):
        x1 = self.conv1(x)
        x1 = self.gdn1(x1)
        x2 = self.conv2(x1)
        x2 = self.gdn2(x2)
        x3 = self.conv3(x2)
        x3 = self.gdn3(x3)
        #x3 = self.conv3p(x3)
        #x3 = self.gdn3p(x3)

        B, C, H, W = x3.size()
        x3 = torch.reshape(x3, shape=(B, C, H * W))
        x4 = self.conv4(x3)
        B, C2, H2 = x4.size()
        x4 = torch.reshape(x4, shape=(B, C2, H, W))

        return x4

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


class Dec(nn.Module):
    def __init__(self, num_features, M1, N):
        super(Dec, self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.N = int(N)

        # main decoder
        self.conv1 = nn.ConvTranspose2d(self.N, self.M1, kernel_size=5, stride=2,
                                        padding=2, output_padding=1)
        #self.gdn1 = gdn.GDN(self.M1, inverse=True)
        self.gdn1 = nn.ReLU()
        self.conv2 = nn.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        #self.gdn2 = gdn.GDN(self.M1, inverse=True)
        self.gdn2 = nn.ReLU()
        self.conv3 = nn.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        #self.gdn3 = gdn.GDN(self.M1, inverse=True)
        self.gdn3 = nn.ReLU()
        self.conv3p = nn.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        # self.gdn3p = gdn.GDN(self.M1, inverse=True)
        self.gdn3p = nn.ReLU()
        self.conv4 = nn.Conv1d(self.M1, 3, kernel_size=3, stride=1, padding=1)

        self.conv1r = nn.ConvTranspose2d(self.N, self.M1, kernel_size=5, stride=2,
                                        padding=2, output_padding=1)
        #self.gdn1r = gdn.GDN(self.M1, inverse=True)
        self.gdn1r = nn.ReLU()
        self.conv2r = nn.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        #self.gdn2r = gdn.GDN(self.M1, inverse=True)
        self.gdn2r = nn.ReLU()
        self.conv3r = nn.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        #self.gdn3r = gdn.GDN(self.M1, inverse=True)
        self.gdn3r = nn.ReLU()
        self.conv3pr = nn.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        # self.gdn3pr = gdn.GDN(self.M1, inverse=True)
        self.gdn3pr = nn.ReLU()
        self.conv4r = nn.Conv1d(self.M1, 3, kernel_size=3, stride=1, padding=1)

        self.fusion1 = Fusion(self.M1)
        self.fusion2 = Fusion(self.M1)

    def forward(self, x_l, x_r):
        x1 = self.conv1(x_l)
        x1 = self.gdn1(x1)

        x1r = self.conv1r(x_r)
        x1r = self.gdn1r(x1r)

        #x1, x1r = self.fusion1(x1, x1r)

        x2 = self.conv2(x1)
        x2 = self.gdn2(x2)
        x2r = self.conv2r(x1r)
        x2r = self.gdn2r(x2r)

        #x2, x2r = self.fusion2(x2, x2r)

        x3 = self.conv3(x2)
        x3 = self.gdn3(x3)
        # x3 = self.conv3p(x3)
        # x3 = self.gdn3p(x3)
        x3r = self.conv3r(x2r)
        x3r = self.gdn3r(x3r)
        # x3r = self.conv3pr(x3r)
        # x3r = self.gdn3pr(x3r)

        B,C,H,W = x3.size()
        x3 = torch.reshape(x3, shape=(B, C, H*W))
        x3r = torch.reshape(x3r, shape=(B, C, H*W))

        x4 = self.conv4(x3)
        x4r = self.conv4r(x3r)
        B, C2, H2 = x4.size()
        x4 = torch.reshape(x4, shape=(B, C2, H,  W))
        x4r = torch.reshape(x4r, shape=(B, C2, H, W))

        return x4, x4r

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
        self.encoder_l = Enc_l(num_features, self.M1, self.M, self.N2)
        self.encoder_r = Enc_r(num_features, self.M1, self.M, self.N2)
        self.decoder = Dec(num_features, self.M, self.N2)
        #self.fusion = Fusion(self.N2)
        #self.factorized_entropy_func = Entropy_bottleneck(N2)
        #self.hyper_enc = Hyper_Enc(N2, M)
        #self.hyper_dec = Hyper_Dec(N2, M)
        #self.gaussin_entropy_func = Distribution_for_entropy2()
        #self.fc1 = nn.Linear(self.M * self.M2 * self.M2, 3000)
        #self.fc2 = nn.Linear(3000, self.M * self.M2 * self.M2)
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

        #y_fusion_l, y_fusion_r = self.fusion(channel_output_l, channel_output_r)
        #output_l, output_r = self.decoder(y_fusion_l, y_fusion_r)

        output_l, output_r = self.decoder(channel_output_l, channel_output_r) #seperated

        # y_main = self.encoder(x)
        # output = self.decoder(y_main)
        return output_l, output_r

