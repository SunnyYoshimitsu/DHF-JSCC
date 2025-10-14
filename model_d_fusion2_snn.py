import math

import torch
import torch.nn as nn
import numpy as np
from gaussian_entropy_model import Distribution_for_entropy2
from channel import Channel

# Import SpikingJelly components
from spikingjelly.activation_based import neuron, functional, layer


class Enc_l_SNN(nn.Module):
    def __init__(self, num_features, M1, M, N2, T=4):
        super(Enc_l_SNN, self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.M = int(M)
        self.N2 = int(N2)
        self.T = T  # Number of time steps for SNN
        
        # main encoder with SNN layers
        self.conv1 = layer.Conv2d(self.n_features, self.M1, kernel_size=5, stride=2, padding=2)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv2 = layer.Conv2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2)
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv3 = layer.Conv2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2)
        self.lif3 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv3p = layer.Conv2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2)
        self.lif3p = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv4 = layer.Conv1d(self.M1, self.N2, kernel_size=3, stride=1, padding=1)
        self.lif4 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())

    def main_enc(self, x):
        # Process input through time steps manually
        # x is [B, C, H, W]
        outputs = []
        
        for t in range(self.T):
            x1 = self.conv1(x)
            x1 = self.lif1(x1)
            
            x2 = self.conv2(x1)
            x2 = self.lif2(x2)
            
            x3 = self.conv3(x2)
            x3 = self.lif3(x3)
            
            B, C, H, W = x3.size()
            x3 = torch.reshape(x3, shape=(B, C, H*W))
            
            x4 = self.conv4(x3)
            x4 = self.lif4(x4)
            
            B, C2, H2 = x4.size()
            x4 = torch.reshape(x4, shape=(B, C2, H, W))
            
            outputs.append(x4)
        
        # Average outputs over time steps
        x_out = torch.stack(outputs).mean(dim=0)  # [B, C2, H, W]
        return x_out

    def forward(self, x):
        functional.reset_net(self)  # Reset membrane potential
        enc = self.main_enc(x)
        return enc


class Enc_r_SNN(nn.Module):
    def __init__(self, num_features, M1, M, N2, T=4):
        super(Enc_r_SNN, self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.M = int(M)
        self.N2 = int(N2)
        self.T = T  # Number of time steps for SNN

        # main encoder with SNN layers
        self.conv1 = layer.Conv2d(self.n_features, self.M1, kernel_size=5, stride=2, padding=2)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv2 = layer.Conv2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2)
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv3 = layer.Conv2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2)
        self.lif3 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv3p = layer.Conv2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2)
        self.lif3p = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv4 = layer.Conv1d(self.M1, self.N2, kernel_size=3, stride=1, padding=1)
        self.lif4 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())

    def main_enc(self, x):
        # Process input through time steps manually
        # x is [B, C, H, W]
        outputs = []
        
        for t in range(self.T):
            x1 = self.conv1(x)
            x1 = self.lif1(x1)
            
            x2 = self.conv2(x1)
            x2 = self.lif2(x2)
            
            x3 = self.conv3(x2)
            x3 = self.lif3(x3)
            
            B, C, H, W = x3.size()
            x3 = torch.reshape(x3, shape=(B, C, H*W))
            
            x4 = self.conv4(x3)
            x4 = self.lif4(x4)
            
            B, C2, H2 = x4.size()
            x4 = torch.reshape(x4, shape=(B, C2, H, W))
            
            outputs.append(x4)
        
        # Average outputs over time steps
        x_out = torch.stack(outputs).mean(dim=0)  # [B, C2, H, W]
        return x_out

    def forward(self, x):
        functional.reset_net(self)  # Reset membrane potential
        enc = self.main_enc(x)
        return enc


class Hyper_Enc_SNN(nn.Module):
    def __init__(self, N2, M, T=4):
        super(Hyper_Enc_SNN, self).__init__()
        self.M = int(M)
        self.N2 = int(N2)
        self.T = T
        
        # hyper encoder with SNN layers
        self.conv1 = layer.Conv2d(self.M, self.N2, kernel_size=3, stride=1)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv2 = layer.Conv2d(self.N2, self.N2, kernel_size=5, stride=2, padding=2)
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv3 = layer.Conv2d(self.N2, self.N2, kernel_size=5, stride=2, padding=2)
        self.lif3 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())

    def forward(self, xq):
        functional.reset_net(self)
        
        xq = torch.abs(xq)
        xq = xq.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, B, C, H, W]
        
        x1 = self.conv1(xq)
        x1 = self.lif1(x1)
        
        x2 = self.conv2(x1)
        x2 = self.lif2(x2)
        
        x3 = self.conv3(x2)
        x3 = self.lif3(x3)
        
        # Average over time steps
        x3 = x3.mean(dim=0)
        return x3


class Hyper_Dec_SNN(nn.Module):
    def __init__(self, N2, M, T=4):
        super(Hyper_Dec_SNN, self).__init__()
        self.M = int(M)
        self.N2 = int(N2)
        self.T = T
        
        # hyper decoder with SNN layers
        self.conv1 = layer.ConvTranspose2d(self.N2, self.N2, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv2 = layer.ConvTranspose2d(self.N2, self.N2, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv3 = layer.ConvTranspose2d(self.N2, self.M * 2, kernel_size=3, stride=1, padding=1)
        self.lif3 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())

    def forward(self, xq2):
        functional.reset_net(self)
        
        xq2 = xq2.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [T, B, C, H, W]
        
        x1 = self.conv1(xq2)
        x1 = self.lif1(x1)
        
        x2 = self.conv2(x1)
        x2 = self.lif2(x2)
        
        x3 = self.conv3(x2)
        x3 = self.lif3(x3)
        
        # Average over time steps
        x3 = x3.mean(dim=0)
        return x3


class Dec_SNN(nn.Module):
    def __init__(self, num_features, M1, N, T=4):
        super(Dec_SNN, self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.N = int(N)
        self.T = T

        # main decoder with SNN layers
        self.conv1 = layer.ConvTranspose2d(self.N, self.M1, kernel_size=5, stride=2,
                                        padding=2, output_padding=1)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv2 = layer.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv3 = layer.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.lif3 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv3p = layer.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.lif3p = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv4 = layer.Conv1d(self.M1, 3, kernel_size=3, stride=1, padding=1)
        self.sigmoid = nn.Sigmoid()  # Add sigmoid to constrain output to [0, 1]

        # Right image decoder
        self.conv1r = layer.ConvTranspose2d(self.N, self.M1, kernel_size=5, stride=2,
                                        padding=2, output_padding=1)
        self.lif1r = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv2r = layer.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.lif2r = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv3r = layer.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.lif3r = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv3pr = layer.ConvTranspose2d(self.M1, self.M1, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.lif3pr = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv4r = layer.Conv1d(self.M1, 3, kernel_size=3, stride=1, padding=1)
        # Sigmoid is shared from above

        self.fusion1 = Fusion_SNN(self.M1, T=T)
        self.fusion2 = Fusion_SNN(self.M1, T=T)

    def forward(self, x_l, x_r):
        functional.reset_net(self)
        
        # Process through time steps manually
        outputs_l = []
        outputs_r = []
        
        for t in range(self.T):
            # Left path
            x1 = self.conv1(x_l)
            x1 = self.lif1(x1)
            
            # Right path  
            x1r = self.conv1r(x_r)
            x1r = self.lif1r(x1r)
            
            # Continue left
            x2 = self.conv2(x1)
            x2 = self.lif2(x2)
            
            # Continue right
            x2r = self.conv2r(x1r)
            x2r = self.lif2r(x2r)
            
            # Continue left
            x3 = self.conv3(x2)
            x3 = self.lif3(x3)
            
            # Continue right
            x3r = self.conv3r(x2r)
            x3r = self.lif3r(x3r)
            
            B, C, H, W = x3.size()
            x3 = torch.reshape(x3, shape=(B, C, H*W))
            x3r = torch.reshape(x3r, shape=(B, C, H*W))
            
            x4 = self.conv4(x3)
            x4r = self.conv4r(x3r)
            
            B, C2, H2 = x4.size()
            x4 = torch.reshape(x4, shape=(B, C2, H, W))
            x4r = torch.reshape(x4r, shape=(B, C2, H, W))
            
            outputs_l.append(x4)
            outputs_r.append(x4r)
        
        # Average over time steps
        x4 = torch.stack(outputs_l).mean(dim=0)
        x4r = torch.stack(outputs_r).mean(dim=0)
        
        # Apply sigmoid to constrain output to [0, 1]
        x4 = self.sigmoid(x4)
        x4r = self.sigmoid(x4r)

        return x4, x4r


class Fusion_SNN(nn.Module):
    def __init__(self, N, T=4):
        super(Fusion_SNN, self).__init__()
        self.N = N
        self.M = int(256)
        self.T = T
        
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
        
        self.conv1r = layer.Conv1d(self.N, 256, kernel_size=3, stride=1, padding=1)
        self.lif1r = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        self.conv1l = layer.Conv1d(self.N, 256, kernel_size=3, stride=1, padding=1)
        self.lif1l = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())

    def forward(self, x_l, x_r):
        functional.reset_net(self)
        
        B, C, H, W = x_l.size()
        x_l = torch.reshape(x_l, shape=(B, C, H * W))
        x_r = torch.reshape(x_r, shape=(B, C, H * W))
        
        # Process through time steps manually
        outputs_l = []
        outputs_r = []
        
        for t in range(self.T):
            x_l_conv = self.conv1l(x_l)
            x_l_lif = self.lif1l(x_l_conv)
            
            x_r_conv = self.conv1r(x_r)
            x_r_lif = self.lif1r(x_r_conv)
            
            outputs_l.append(x_l_lif)
            outputs_r.append(x_r_lif)
        
        # Average over time
        x_l = torch.stack(outputs_l).mean(dim=0)
        x_r = torch.stack(outputs_r).mean(dim=0)
        
        B, C2, H2 = x_l.size()
        x_l = x_l.permute(0, 2, 1)  # B, H*W, C
        x_r = x_r.permute(0, 2, 1)  # B, H*W, C
        
        x_ln_l = self.ln1(x_l)
        x_ln_r = self.ln2(x_r)
        
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


class Image_coding_SNN(nn.Module):
    def __init__(self, M, N2, num_features=3, M1=256, T=4):
        super(Image_coding_SNN, self).__init__()
        self.M1 = int(M1)
        self.n_features = int(num_features)
        self.M = int(M)
        self.N2 = int(N2)
        self.T = T
        
        self.encoder_l = Enc_l_SNN(num_features, self.M1, self.M, self.N2, T=T)
        self.encoder_r = Enc_r_SNN(num_features, self.M1, self.M, self.N2, T=T)
        self.decoder = Dec_SNN(num_features, self.M, self.N2, T=T)
        self.channel = Channel(5)

    def add_noise(self, x):
        noise = np.random.uniform(-0.5, 0.5, x.size())
        noise = torch.Tensor(noise).cuda()
        return x + noise

    def forward(self, x_l, x_r):
        y_main_l = self.encoder_l(x_l)
        y_main_r = self.encoder_r(x_r)
        
        channel_input_l = y_main_l
        channel_input_r = y_main_r
        channel_output_l, channel_usage_l = self.channel.forward(channel_input_l)
        channel_output_r, channel_usage_r = self.channel.forward(channel_input_r)

        output_l, output_r = self.decoder(channel_output_l, channel_output_r)

        return output_l, output_r
