import torch
import torch.nn as nn
import numpy as np
from factorized_entropy_model import Entropy_bottleneck
from gaussian_entropy_model import Distribution_for_entropy2
import math
from channel import Channel


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
        self.relu1 = nn.ReLU()
        self.CA1 =  self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.M1, out_channels=self.M1, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.conv2 = nn.Conv2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2)
        # self.gdn2 = gdn.GDN(self.M1)
        self.relu2 = nn.ReLU()
        self.CA2 = self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.M1, out_channels=self.M1, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv3 = nn.Conv2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2)
        # self.gdn3 = gdn.GDN(self.M1)
        self.relu3 = nn.ReLU()
        self.CA3 = self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.M1, out_channels=self.M1, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        self.conv4 = nn.Conv2d(self.M1, self.N2, kernel_size=3, stride=1, padding=1)
        # hyper encoder
        # self.conv1_hy = nn.Conv2d(self.N, self.M1, kernel_size=5, stride=2, padding=2)
        # self.conv2_hy = nn.Conv2d(self.M1, self.M1*2, kernel_size=5, stride=2, padding=2)

    def main_enc(self, x):
        x1 = self.conv1(x)
        x1 = self.re1lu(x1)
        x1c = self.CA1(x1)
        x1 = x1 * x1c
        x2 = self.conv2(x1)
        x2 = self.relu2(x2)
        x2c = self.CA2(x2)
        x2 = x2 * x2c
        x3 = self.conv3(x2)
        x3 = self.relu3(x3)
        x3c = self.CA3(x3)
        x3 = x3 * x3c

        x4 = self.conv4(x3)

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
    def __init__(self, M, C):
        super(Dyna_Enc, self).__init__()
        self.M = M
        self.C = C

    def forward(self, x, px):
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1)
        px = px.permute(0, 2, 3, 1)
        x = torch.reshape(x, shape=(B, H * W, C))
        px = torch.reshape(px, shape=(B, H * W, C))
        hx = torch.clamp_min(-torch.log(px)/ math.log(2), 0)
        #symbol = hx * 0.2
        eta = torch.max(hx)
        print(px)
        mask = torch.arange(0, self.C).repeat(H*W, 1).cuda()  # mask: [H*W, C]
        #self.register_buffer("mask", mask)
        mask = mask.repeat(B, 1, 1)  # mask: [B, H*W, C]
        mask_new = torch.zeros_like(mask)
        mask_new[hx < eta] = 0
        mask_new[hx >= eta] = 1

        x_masked = x * mask_new
        x_masked = torch.reshape(x_masked, shape=(B, H, W, C))
        x_masked = x_masked.permute(0, 3, 1, 2)
        return x_masked


class Hyper_Dec(nn.Module):
    def __init__(self, N2, M):
        super(Hyper_Dec, self).__init__()
        self.M = int(M)
        self.N2 = int(N2)
        # hyper decoder
        self.conv1 = nn.ConvTranspose2d(self.N2, self.M, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(self.M, self.M, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(self.M, self.N2 * 2, kernel_size=3, stride=1, padding=1)
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
        #self.gdn1 = GDN(self.M1, inverse=True)
        self.gdn1 = nn.LeakyReLU()
        self.conv2 = nn.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        #self.gdn2 = GDN(self.M1, inverse=True)
        self.gdn2 = nn.LeakyReLU()
        self.conv3 = nn.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        #self.gdn3 = GDN(self.M1, inverse=True)
        self.gdn3 = nn.LeakyReLU()
        self.conv4 = nn.ConvTranspose2d(self.M1, 3, kernel_size=5, stride=2, padding=2, output_padding=1)

        self.conv1r = nn.ConvTranspose2d(self.N, self.M1, kernel_size=5, stride=2,
                                        padding=2, output_padding=1)
        #self.gdn1r = GDN(self.M1, inverse=True)
        self.gdn1r = nn.LeakyReLU()
        self.conv2r = nn.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        #self.gdn2r = GDN(self.M1, inverse=True)
        self.gdn2r = nn.LeakyReLU()
        self.conv3r = nn.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        #self.gdn3r = GDN(self.M1, inverse=True)
        self.gdn3r = nn.LeakyReLU()
        self.conv4r = nn.ConvTranspose2d(self.M1, 3, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, x_l, x_r):
        x1 = self.conv1(x_l)
        x1 = self.gdn1(x1)
        x1r = self.conv1r(x_r)
        x1r = self.gdn1r(x1r)

        x2 = self.conv2(x1)
        x2 = self.gdn2(x2)
        x2r = self.conv2r(x1r)
        x2r = self.gdn2r(x2r)

        x3 = self.conv3(x2)
        x3 = self.gdn3(x3)
        x3r = self.conv3r(x2r)
        x3r = self.gdn3r(x3r)

        x4 = self.conv4(x3)
        x4r = self.conv4r(x3r)
        return x4, x4r

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

class Image_encdec(nn.Module):
    def __init__(self, M, N2, num_features=3, M1=64):
        super(Image_encdec, self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.M = int(M)
        self.N2 = int(N2)
        self.enc = Enc(num_features, self.M1, self.M, self.N2)
        self.factorized_entropy_func = Entropy_bottleneck(N2)
        self.hyper_enc = Hyper_Enc(N2, M)
        self.hyper_dec = Hyper_Dec(N2, M)
        self.gaussin_entropy_func = Distribution_for_entropy2()

    def add_noise(self, x):
        noise = np.random.uniform(-0.5, 0.5, x.size())
        noise = torch.Tensor(noise).cuda()
        return x + noise

    def forward(self, x, if_training):
        y_main = self.enc(x)
        y_hyper = self.hyper_enc(y_main)

        y_hyper_q, p_hyper = self.factorized_entropy_func(y_hyper, if_training)
        gaussian_params = self.hyper_dec(y_hyper_q)
        if if_training:
            y_main_q = self.add_noise(y_main)
        else:
            y_main_q = torch.round(y_main)
        p_main = self.gaussin_entropy_func(y_main_q, gaussian_params)

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
        self.JSCCenc_l = Dyna_Enc(M, N2)
        self.JSCCenc_r = Dyna_Enc(M, N2)
        self.channel = Channel(5)
        self.decoder = Dec(num_features, self.M, self.N2)
        self.fusion = Fusion(self.N2)


    def add_noise(self, x):
        noise = np.random.uniform(-0.5, 0.5, x.size())
        noise = torch.Tensor(noise).cuda()
        return x + noise

    def forward(self, x_l, x_r, if_training):
        y_main_l, p_main_l, p_hyper_l, y_main_q_l, gaussian_params_l = self.enc_l(x_l, if_training)
        y_main_r, p_main_r, p_hyper_r, y_main_q_r, gaussian_params_r = self.enc_r(x_r, if_training)

        y_l = self.JSCCenc_l(y_main_l, p_main_l)
        y_r = self.JSCCenc_r(y_main_r, p_main_r)

        s_l, _ = self.channel.forward(y_l)
        s_r, _ = self.channel.forward(y_r)

        # 双视角特征信息融合
        s_fusion_l, s_fusion_r = self.fusion(s_l, s_r)
        output_l, output_r = self.decoder(s_fusion_l, s_fusion_r)

        return output_l, output_r, p_main_l, p_hyper_l, p_main_r, p_hyper_r
