import argparse
import os
import numpy as np
import pandas as pd
import torch
import yaml
import PIL.Image as Image
from collections import OrderedDict
from torch.utils.data import DataLoader
from dataset.PairKitti import PairKitti
from dataset.InStereo2K import InStereo2K
import model_d_fusion2_snn
from pytorch_msssim import ms_ssim
import math
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml', help="configuration")


def get_bpp(model_out, config):  # Returns calculated bpp for train and test
    alpha = config['alpha']
    beta = config['beta']
    if config['baseline_model'] == 'bmshj18':
        if config['use_side_info']:  # If the side information (correlated image) has to be used
            ''' 
            The loss function consists of:
            Rate terms for input image (likelihoods), correlated image (y_likelihoods),
            and the common information (w_likelihoods), hyperpriors for input image (z_likelihoods)
            , hyperpriors for correlated image (z_likelihoods_cor).
            Sum of these rate terms is returned as bpp, along with the actual bpp transmitted over the channel,
            which consists only of likelihoods + z_likelihoods.
            '''
            x_recon, y_recon, likelihoods, y_likelihoods, z_likelihoods, z_likelihoods_cor, w_likelihoods = model_out
            size_est = (-np.log(2) * x_recon.numel() / 3)
            bpp = (torch.sum(torch.log(likelihoods)) + torch.sum(torch.log(z_likelihoods))) / size_est
            transmitted_bpp = bpp.clone().detach()  # the real bpp value which is transmitted (for test)
            bpp += alpha * (torch.sum(torch.log(y_likelihoods)) + torch.sum(torch.log(z_likelihoods_cor))) / size_est
            bpp += beta * torch.sum(torch.log(w_likelihoods)) / size_est
            return bpp, transmitted_bpp
        else:  # The baseline implementation (Balle2018) without the side information
            x_recon, likelihoods, z_likelihoods = model_out
            size_est = (-np.log(2) * x_recon.numel() / 3)
            bpp = (torch.sum(torch.log(likelihoods)) + torch.sum(torch.log(z_likelihoods))) / size_est
            return bpp, bpp
    elif config['baseline_model'] == 'bls17':
        if config['use_side_info']:
            x_recon, y_recon, likelihoods, y_likelihoods, w_likelihoods = model_out
            size_est = (-np.log(2) * x_recon.numel() / 3)
            bpp = torch.sum(torch.log(likelihoods)) / size_est
            transmitted_bpp = bpp.clone().detach()  # the real bpp value which is transmitted (for test)
            bpp += alpha * torch.sum(torch.log(y_likelihoods)) / size_est
            bpp += beta * torch.sum(torch.log(w_likelihoods)) / size_est
            return bpp, transmitted_bpp
        else:
            x_recon, likelihoods = model_out
            size_est = (-np.log(2) * x_recon.numel() / 3)
            bpp = torch.sum(torch.log(likelihoods)) / size_est
            return bpp, bpp
    return None


def get_distortion(config, out_l, out_r, img, cor_img, mse):
    distortion = None
    alpha = config['alpha']
    if config['use_side_info']:
        ''' 
        The loss function consists of:
        Distortion terms for input image (x_recon), and correlated image (x_cor_recon).
        '''
        x_recon, y_recon = out_l, out_r
        if config['distortion_loss'] == 'MS-SSIM':
            distortion = (1 - ms_ssim(img.cpu(), x_recon.cpu(), data_range=1.0, size_average=True,
                                      win_size=7))
            distortion += alpha * (1 - ms_ssim(cor_img.cpu(), y_recon.cpu(), data_range=1.0, size_average=True,
                                               win_size=7))
        elif config['distortion_loss'] == 'MSE':
            distortion = mse(img, x_recon)
            distortion += alpha * mse(cor_img, y_recon)
    else:
        x_recon = out_l
        if config['distortion_loss'] == 'MS-SSIM':
            distortion = (1 - ms_ssim(img.cpu(), x_recon.cpu(), data_range=1.0, size_average=True,
                                      win_size=7))
        elif config['distortion_loss'] == 'MSE':
            distortion = mse(img, x_recon)

    return distortion


def map_layers(weight):
    """ Since the pre-trained weights provided for bls17 by us were trained with
        different layer names, we map the layer names in the state dictionaries
        to the new names using the following function map_layers().
    """
    return OrderedDict([(k.replace('z', 'w'), v) if 'z' in k else (k, v) for k, v in weight.items()])


def save_image(x_recon, x, path, name):
    img_recon = np.clip((x_recon * 255).squeeze().cpu().numpy(), 0, 255)
    img = np.clip((x * 255).squeeze().cpu().numpy(), 0, 255)
    img_recon = np.transpose(img_recon, (1, 2, 0)).astype('uint8')
    img = np.transpose(img, (1, 2, 0)).astype('uint8')
    img_final = Image.fromarray(np.concatenate((img, img_recon), axis=1), 'RGB')
    if not os.path.exists(path):
        os.makedirs(path)
    img_final.save(os.path.join(path, name + '.png'))


def main(config):
    # Dataset initialization
    path='./dataset'
    resize = tuple(config['resize'])
    if config['dataset_name'] == 'KITTI':
        train_dataset = PairKitti(path=path, set_type='train', resize=resize)
        val_dataset = PairKitti(path=path, set_type='val', resize=resize)
        test_dataset = PairKitti(path=path, set_type='test', resize=resize)
    elif config['dataset_name'] == 'InStereo2K':
        train_dataset = InStereo2K(path=path, set_type='train', resize=resize)
        val_dataset = InStereo2K(path=path, set_type='val', resize=resize)
        test_dataset = InStereo2K(path=path, set_type='test', resize=resize)
    else:
        raise Exception("Dataset not found")

    batch_size = config['train_batch_size']
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True, num_workers=3)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=3)

    # Model initialization
    with_side_info = config['use_side_info']
    model_class = None
    
        
    # Initialize SNN model
    print("=" * 60)
    print("⚡ Using SPIKING NEURAL NETWORK (SNN) Architecture")
    print("=" * 60)
    print("  • Framework: SpikingJelly")
    print("  • Neuron Type: LIF (Leaky Integrate-and-Fire)")
    print("  • Time Steps: 8")
    print("  • Membrane tau: 2.0")
    print("  • Surrogate Gradient: ATan")
    print("=" * 60)
    model = model_d_fusion2_snn.Image_coding_SNN(M=256, N2=25, T=8)
    
    print("Training SNN from scratch")
    
    model = model.cuda() if config['cuda'] else model
    
    # Enable Multi-GPU training with DataParallel
    if config['cuda']:
        num_gpus = torch.cuda.device_count()
        if num_gpus >= 2:
            model = torch.nn.DataParallel(model, device_ids=[0, 1])
            print(f"Using 2 GPUs (DataParallel)")
        else:
            print(f"Using 1 GPU")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], amsgrad=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-7)
    experiment_name = str(train_dataset) + '_' + config['distortion_loss'] + '_lambda:' + \
                      str(config['lambda'])

    print('Experiment: ', experiment_name)

    weight_folder = None
    if config['save_weights']:
        weight_folder = os.path.join(config['save_output_path'], 'weight')
        if not os.path.exists(weight_folder):
            os.makedirs(weight_folder)

    # Training initialization
    mse = torch.nn.MSELoss(reduction='mean')
    mse = mse.cuda() if config['cuda'] else mse
    lmbda = config['lambda']
    if config['train']:
        min_val_loss = None
        for epoch in range(config['epochs']):
            model.train()
            train_loss = []
            channel_numl = []
            num_pixels = 3*128*128
            for i, data in enumerate(iter(train_loader)):
                img, cor_img, _, _ = data
                img = img.cuda().float() if config['cuda'] else img.float()
                cor_img = cor_img.cuda().float() if config['cuda'] else cor_img.float()

                optimizer.zero_grad()

                # if with_side_info:
                #     out_l, out_r = model(img, cor_img)
                # else:
                #     out_l = model(img)

                #out_l, out_r, p_main_l, p_hyper_l, p_main_r, p_hyper_r, num_l = model(img, cor_img, True)
                out_l, out_r = model(img, cor_img)
                # mm = torch.log(p_main_r) / (-math.log(2)) * 10
                # print(np.sum((mm<12).cpu().numpy()))
                # print(np.sum((mm<0.0001).cpu().numpy()))
                # print(torch.max(mm))
                # print('min:', torch.min(mm))

                #bpp, _ = get_bpp(out, config)

                distortion = get_distortion(config, out_l, out_r, img, cor_img, mse)
                # train_bpp_hyper_l = torch.sum(torch.log(p_hyper_l)) / (-math.log(2) * num_pixels)
                # train_bpp_main_l = torch.sum(torch.log(p_main_l)) / (-math.log(2) * num_pixels)
                # train_bpp_hyper_r = torch.sum(torch.log(p_hyper_r)) / (-math.log(2) * num_pixels)
                # train_bpp_main_r = torch.sum(torch.log(p_main_r)) / (-math.log(2) * num_pixels)
                # train_bpp_main = train_bpp_main_l + train_bpp_main_r
                # train_bpp_hyper = train_bpp_hyper_l + train_bpp_hyper_r

                #loss = 10 * distortion * (255 ** 2) + (train_bpp_main + 0.1*train_bpp_hyper) # multiplied by (255 ** 2) for distortion scaling
                loss = distortion * (255 ** 2)
                loss.backward()
                optimizer.step()
                train_loss.append(distortion.item())
                #channel_numl.append(num_l)
                #channel_usager.append(channel_usage_r)


            #print('channel_numl:', (sum(channel_numl)/len(channel_numl)))
            dis = (sum(train_loss) / (len(train_loss)))

            # Validation
            model.eval()
            val_loss = []
            val_mse = []
            val_msssim = []
            val_bpp = []
            val_transmitted_bpp = []
            val_num = []
            val_distortion = []
            with torch.no_grad():
                for i, data in enumerate(iter(val_loader)):
                    # img = input image, cor_img = side information/correlated image (designated y in the paper)
                    img, cor_img, _, _ = data
                    img = img.cuda().float() if config['cuda'] else img.float()
                    cor_img = cor_img.cuda().float() if config['cuda'] else cor_img.float()

                    # if with_side_info:
                    #     out_l, out_r = model(img, cor_img)
                    # else:
                    #     out = model(img)

                    #out_l, out_r, p_main_l, p_hyper_l, p_main_r, p_hyper_r, num_l = model(img, cor_img, True)
                    out_l, out_r = model(img, cor_img)

                    # mm = torch.log(p_main_r) / (-math.log(2)) * 10
                    # print(np.sum((mm < 0.0001).cpu().numpy()))
                    # print(torch.max(mm))
                    # print('min:', torch.min(mm))

                    #bpp, transmitted_bpp = get_bpp(out, config)

                    x_recon = out_l
                    mse_dist = mse(img, x_recon)
                    msssim = 1 - ms_ssim(img.clone().cpu(), x_recon.clone().cpu(), data_range=1.0, size_average=True,
                                         win_size=7)
                    msssim_db = -10 * np.log10(msssim)

                    distortion = get_distortion(config, out_l, out_r, img, cor_img, mse)

                    loss = lmbda * distortion * (255 ** 2)  # multiplied by (255 ** 2) for distortion scaling

                    val_mse.append(mse_dist.item())
                    #val_bpp.append(bpp.item())
                    #val_transmitted_bpp.append(transmitted_bpp.item())
                    val_loss.append(loss.item())
                    val_msssim.append(msssim_db.item())
                    val_distortion.append(distortion.item())
                    #val_num.append(num_l)



            val_loss_to_track = sum(val_loss) / len(val_loss)
            #scheduler.step(val_loss_to_track)
            #print('test_channel_num:', sum(val_num) / len(val_num))
            # Verbose
            if config['verbose_period'] > 0 and (epoch + 1) % config['verbose_period'] == 0:
                tracking = ['Epoch {}:'.format(epoch + 1),
                            'Loss = {:.4f},'.format(val_loss_to_track),
                            #'BPP = {:.4f},'.format(sum(val_bpp) / len(val_bpp)),
                            'Distortion = {:.4f},'.format(sum(val_distortion) / len(val_distortion)),
                            #'Transmitted BPP = {:.4f},'.format(sum(val_transmitted_bpp) / len(val_transmitted_bpp)),
                            'PSNR = {:.4f},'.format(10 * np.log10(1 / (sum(val_mse) / (len(val_mse))))),
                            'MS-SSIM = {:.4f}'.format(sum(val_msssim) / len(val_msssim))]
                print(" ".join(tracking))

            # Save weights
            if config['save_weights']:
                if min_val_loss is None or min_val_loss > val_loss_to_track:
                    min_val_loss = val_loss_to_track
                    
                    # Create organized directory structure with date
                    import datetime
                    date_folder = datetime.datetime.now().strftime("%m_%d")  # e.g., "10_07"
                    pkl_dir = os.path.join('.', 'checkpoints', date_folder, 'pkl')
                    pth_dir = os.path.join('.', 'checkpoints', date_folder, 'pth')
                    
                    # Create directories if they don't exist
                    os.makedirs(pkl_dir, exist_ok=True)
                    os.makedirs(pth_dir, exist_ok=True)
                    
                    # Calculate PSNR for filename
                    psnr = 10 * np.log10(1 / (sum(val_mse) / (len(val_mse))))
                    
                    # Get the actual model (unwrap DataParallel if needed)
                    model_to_save = model.module if isinstance(model, torch.nn.DataParallel) else model
                    
                    # Save full model as .pkl (for backup/compatibility)
                    pkl_path = os.path.join(pkl_dir, 'epoch_%04d_psnr_%.2fdB.pkl' % (epoch + 1, psnr))
                    torch.save(model_to_save, pkl_path)
                    
                    # Save state dict as .pth (recommended format)
                    pth_path = os.path.join(pth_dir, 'epoch_%04d_psnr_%.2fdB.pth' % (epoch + 1, psnr))
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'psnr': psnr,
                        'loss': min_val_loss
                    }, pth_path)
                    
                    print(f"✅ Saved checkpoint: {pkl_path}")
                    print(f"✅ Saved state dict: {pth_path}")

    if config['test']:
        results_path = os.path.join(config['save_output_path'], 'results')
        if not os.path.exists(results_path):
            os.makedirs(results_path)
        names = ["Image Number", "BPP", "PSNR", "MS-SSIM"]
        cols = dict()
        model.eval()
        mse_test = []
        with torch.no_grad():
            for i, data in enumerate(iter(test_loader)):
                img, cor_img, _, _ = data
                img = img.cuda().float() if config['cuda'] else img.float()
                cor_img = cor_img.cuda().float() if config['cuda'] else cor_img.float()

                #out_l, out_r, p_main_l, p_hyper_l, p_main_r, p_hyper_r, num_l = model(img, cor_img, True)
                out_l, out_r = model(img, cor_img)


                x_recon = out_l
                #x_recon = out_r
                mse_dist = mse(img, x_recon)
                mse_test.append(mse_dist.item())
                msssim = 1 - ms_ssim(img.clone().cpu(), x_recon.clone().cpu(), data_range=1.0, size_average=True,
                                     win_size=7)
                msssim_db = -10 * np.log10(msssim)

                vals = [str(i)] + ['{:.8f}'.format(x) for x in [
                                                                10 * np.log10(1 / mse_dist.item()),
                                                                msssim.item()]]
                print(vals)
                for (name, val) in zip(names, vals):
                    if name not in cols:
                        cols[name] = []
                    cols[name].append(val)

                if config['save_image']:
                    save_image(x_recon[0], img[0], os.path.join(results_path, '{}_images'.format(1)),
                               str(i))

            df = pd.DataFrame.from_dict(cols)
            df.to_csv(os.path.join(results_path, experiment_name + '.csv'))
            print(msssim)


if __name__ == '__main__':
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    main(config)
