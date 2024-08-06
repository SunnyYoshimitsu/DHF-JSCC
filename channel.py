import torch.nn as nn
import numpy as np
import torch


class Channel(nn.Module):
    def __init__(self, para):
        super(Channel, self).__init__()
        self.chan_type = 'awgn'
        self.chan_param = para
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def gaussian_noise_layer(self, input_layer, std):
        device = input_layer.get_device()
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(input_layer), device=device)
        noise = noise_real + 1j * noise_imag
        return input_layer + noise

    def forward(self, input, avg_pwr=None, power=1):
        if avg_pwr is None:
            zero_num = np.sum((input == 0).cpu().numpy())
            b,h,w,c = input.size()
            avg_pwr = torch.sum(input ** 2) / (b*h*w*c - zero_num)
            #avg_pwr = torch.mean(input ** 2)
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        else:
            channel_tx = np.sqrt(power) * input / torch.sqrt(avg_pwr * 2)
        input_shape = channel_tx.shape
        # 将信道输入符号展开
        channel_in = channel_tx.reshape(-1)
        channel_in = channel_in[::2] + channel_in[1::2] * 1j
        channel_usage = channel_in.numel()
        channel_output = self.channel_forward(channel_in)
        channel_rx = torch.zeros_like(channel_tx.reshape(-1))
        channel_rx[::2] = torch.real(channel_output)
        channel_rx[1::2] = torch.imag(channel_output)
        # 将维度还原
        channel_rx = channel_rx.reshape(input_shape)
        return channel_rx * torch.sqrt(avg_pwr * 2), channel_usage

    def channel_forward(self, channel_in):
        if self.chan_type == 0 or self.chan_type == 'noiseless':
            return channel_in

        elif self.chan_type == 1 or self.chan_type == 'awgn':
            channel_tx = channel_in
            # 信道输入符号为复数形式，因此信道噪声标准差需乘根号0.5
            sigma = np.sqrt(1.0 / (2 * 10 ** ((self.chan_param) / 10)))
            chan_output = self.gaussian_noise_layer(channel_tx,
                                                    std=sigma)
            return chan_output
