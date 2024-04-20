import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import torch_msssim
import model_d_fusion2
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset.PairKitti import PairKitti

# from utils import dali
# from nvidia.dali.plugin.pytorch import DALIGenericIterator

class SimpleDataset(Dataset):
    def __init__(self, input_path, img_size=256):
        super(SimpleDataset, self).__init__()
        self.input_list = []
        self.label_list = []
        self.num = 0
        self.img_size = img_size

        for _ in range(30):
            for i in os.listdir(input_path):
                input_img = input_path + i
                self.input_list.append(input_img)
                self.num = self.num + 1

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img = np.array(Image.open(self.input_list[idx]))
        input_np = img.astype(np.float32).transpose(2, 0, 1) / 255.0
        input_tensor = torch.from_numpy(input_np)
        return input_tensor


class MyDataset(Dataset):
    def __init__(self, input_path, img_size=128):
        super(MyDataset, self).__init__()
        self.input_list = []
        self.label_list = []
        self.num = 0
        self.img_size = img_size

        for i in os.listdir(input_path):
            input_img = input_path + i
            self.input_list.append(input_img)
            self.num = self.num + 1

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img = np.array(Image.open(self.input_list[idx]))
        # x = np.random.randint(0, img.shape[0] - self.img_size)
        # y = np.random.randint(0, img.shape[1] - self.img_size)
        x = 128
        y = 600
        input_np = img[x:x + self.img_size, y:y + self.img_size, :].astype(np.float32).transpose(2, 0, 1) / 255.0
        input_tensor = torch.from_numpy(input_np)
        return input_tensor

class MyDataset2(Dataset):
    def __init__(self, input_path, img_size=128):
        super(MyDataset2, self).__init__()
        self.input_list = []
        self.label_list = []
        self.num = 0
        self.img_size = img_size

        for i in os.listdir(input_path):
            input_img = input_path + i
            self.input_list.append(input_img)
            self.num = self.num + 1

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        img = np.array(Image.open(self.input_list[idx]))
        # x = np.random.randint(0, img.shape[0] - self.img_size)
        # y = np.random.randint(0, img.shape[1] - self.img_size)
        x = 128
        y = 600
        input_np = img[x:x + self.img_size, y:y + self.img_size, :].astype(np.float32).transpose(2, 0, 1) / 255.0
        input_tensor = torch.from_numpy(input_np)
        return input_tensor

def eval():
    pass

path='./dataset'
resize = (128, 128)
# train_data = SimpleDataset(input_path='/datasets/img256x256/')
train_dataset = PairKitti(path=path, set_type='train', resize=resize)
val_dataset = PairKitti(path=path, set_type='val', resize=resize)
test_dataset = PairKitti(path=path, set_type='test', resize=resize)
batch_size = 16
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
# pipe = dali.SimplePipeline('../datasets', batch_size=12, num_threads = 2, device_id = 0)
# pipe.build()
# train_loader = DALIGenericIterator(pipe, ['data'], size=90306)

TRAINING = True
M = 256
N2 = 40
image_comp = model_d_fusion2.Image_coding(M=M, N2=N2).cuda()
#image_comp = torch.load('ae_4999_57.42413_0.00000000.pkl')

METRIC = "PSNR"
print("====> using PSNR", METRIC)
lamb = 5.
lr = 0.0001

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# params = list(image_comp.parameters()) + list(context.parameters())

optimizer = torch.optim.Adam(image_comp.parameters(), lr=lr)

if METRIC == "MSSSIM":
    loss_func = torch_msssim.MS_SSIM(max_val=1).cuda()
elif METRIC == "PSNR":
    loss_func = nn.MSELoss()
pix = torch.tensor(255. * 255.)

hx = 0
cc = 0
for epoch in range(5000):
    rec_loss, bpp = 0., 0.
    channel_sum_l = 0
    channel_sum_r = 0
    cnt = 0
    for step, data in enumerate(iter(train_loader)):
        # batch_x = batch_x[0]['data']
        # batch_x = batch_x.type(dtype=torch.float32)
        # batch_x = torch.cast(batch_x,"float")/255.0
        # batch_x = batch_x/255.0
        batch_x_l, batch_x_r, _, _ = data
        batch_x_r = batch_x_r.cuda()
        batch_x_l = batch_x_l.cuda()
        num_pixels = batch_x_r.size()[0] * batch_x_r.size()[2] * batch_x_r.size()[3]
        rec_l, rec_r = \
            image_comp(batch_x_l,batch_x_r, TRAINING)
        # y_main, rec = image_comp(batch_x, TRAINING)

        if METRIC == "MSSSIM":
            dloss = (1.-loss_func(rec_l, batch_x_l) + 1.-loss_func(rec_r, batch_x_r)) * 0.5
        elif METRIC == "PSNR":
            dloss = (loss_func(rec_l * 255., batch_x_l * 255.)
                     + loss_func(rec_r * 255., batch_x_r * 255.)) * 0.5

        # train_bpp_hyper_l = torch.sum(torch.log(p_hyper_l)) / (-math.log(2) * num_pixels)
        # train_bpp_main_l = torch.sum(torch.log(p_main_l)) / (-math.log(2) * num_pixels)
        # train_bpp_hyper_r = torch.sum(torch.log(p_hyper_r)) / (-math.log(2) * num_pixels)
        # train_bpp_main_r = torch.sum(torch.log(p_main_r)) / (-math.log(2) * num_pixels)
        #
        # train_bpp_main = train_bpp_main_l + train_bpp_main_r
        # train_bpp_hyper = train_bpp_hyper_l + train_bpp_hyper_r
        # loss = lamb * dloss + (train_bpp_main + train_bpp_hyper)
        loss = dloss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if METRIC == "MSSSIM":
            rec_loss = rec_loss + (1. - dloss.item())
            d = 1. - dloss.item()
        elif METRIC == "PSNR":
            rec_loss = rec_loss + dloss.item()
            d = dloss.item()
        # channel_sum_l = channel_sum_l + channel_usage_l
        # channel_sum_r = channel_sum_r + channel_usage_r
        cnt = cnt + 1

        #bpp = bpp + train_bpp_main.item() + train_bpp_hyper.item()
        pix = torch.tensor(255 * 255)
        # print('epoch',epoch,'step:', step, '%s:'%(METRIC), 10*torch.log10(pix/d), 'main_bpp:',train_bpp_main.item(),
        #       'hyper_bpp:',train_bpp_hyper.item(), 'channel_usage:', channel_usage)
        if step % 50 == 0:
            cc = 1
            #print(indexs)
            hx = hx + 1
            #test(hx)
            # print('epoch',epoch,'step:', step, '%s:'%(METRIC), 10*torch.log10(pix/d), 'main_bpp_l:',train_bpp_main_l.item(),
            #       'main_bpp_r:', train_bpp_main_r.item(),
            #     'hyper_bpp:',train_bpp_hyper.item(), 'channel_usage:', channel_usage_l, channel_usage_r)
            # print(symbol_l)
            # print(symbol_r)
            if METRIC == "MSSSIM":
                print(hx, '%s:' % (METRIC), d)
            elif METRIC == "PSNR":
                print(hx, '%s:' % (METRIC), 10 * torch.log10(pix / d))

        if (epoch + 1) % 50 == 0 and step == 1:
            print('channel_sum:', channel_sum_l / cnt, channel_sum_r / cnt)
            torch.save(image_comp, 'ae_%d_%.5f_%.8f.pkl' % (epoch, rec_loss / cnt, bpp / cnt))
            rec_loss, bpp = 0., 0.
            channel_sum_l = 0
            channel_sum_r = 0
            cnt = 0
