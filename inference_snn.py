import argparse
import os
import numpy as np
import torch
import yaml
import PIL.Image as Image
from torch.utils.data import DataLoader
from dataset.PairKitti import PairKitti
import model_d_fusion2_snn
import math

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 20 * math.log10(1.0 / math.sqrt(mse.item()))
    return psnr

def save_images(original_left, original_right, recon_left, recon_right, save_path, idx):
    """Save original and reconstructed images side by side"""
    # Convert tensors to numpy arrays
    orig_l = np.clip((original_left.squeeze().cpu().numpy() * 255), 0, 255).astype('uint8')
    orig_r = np.clip((original_right.squeeze().cpu().numpy() * 255), 0, 255).astype('uint8')
    rec_l = np.clip((recon_left.squeeze().cpu().numpy() * 255), 0, 255).astype('uint8')
    rec_r = np.clip((recon_right.squeeze().cpu().numpy() * 255), 0, 255).astype('uint8')
    
    # Transpose from CHW to HWC
    orig_l = np.transpose(orig_l, (1, 2, 0))
    orig_r = np.transpose(orig_r, (1, 2, 0))
    rec_l = np.transpose(rec_l, (1, 2, 0))
    rec_r = np.transpose(rec_r, (1, 2, 0))
    
    # Create side-by-side comparisons
    # Left: Original Left | Reconstructed Left
    left_comparison = np.concatenate((orig_l, rec_l), axis=1)
    # Right: Original Right | Reconstructed Right
    right_comparison = np.concatenate((orig_r, rec_r), axis=1)
    # Final: Stack both comparisons vertically
    final_image = np.concatenate((left_comparison, right_comparison), axis=0)
    
    # Save image
    img_pil = Image.fromarray(final_image, 'RGB')
    img_pil.save(os.path.join(save_path, f'test_{idx:04d}.png'))
    
    # Also save individual images
    Image.fromarray(orig_l, 'RGB').save(os.path.join(save_path, f'test_{idx:04d}_orig_left.png'))
    Image.fromarray(orig_r, 'RGB').save(os.path.join(save_path, f'test_{idx:04d}_orig_right.png'))
    Image.fromarray(rec_l, 'RGB').save(os.path.join(save_path, f'test_{idx:04d}_recon_left.png'))
    Image.fromarray(rec_r, 'RGB').save(os.path.join(save_path, f'test_{idx:04d}_recon_right.png'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help="configuration file")
    parser.add_argument('--checkpoint', type=str, required=True, help="path to checkpoint")
    parser.add_argument('--output_dir', type=str, default='inference_results/results/1_images', 
                        help="output directory for results")
    parser.add_argument('--num_images', type=int, default=10, help="number of test images to process")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("⚡ SNN INFERENCE")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output dir: {args.output_dir}")
    print(f"Num images: {args.num_images}")
    print("=" * 60)
    
    # Load test dataset
    path = './dataset'
    resize = tuple(config['resize'])
    test_dataset = PairKitti(path=path, set_type='test', resize=resize)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    print(f"Test dataset loaded: {len(test_dataset)} images")
    
    # Load model
    print("Loading SNN model...")
    model = model_d_fusion2_snn.Image_coding_SNN(M=256, N2=25, T=8)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded checkpoint (direct state dict)")
    
    # Move to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    print(f"Model on device: {device}")
    
    # Run inference
    print("\nRunning inference...")
    psnr_left_list = []
    psnr_right_list = []
    
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            if idx >= args.num_images:
                break
            
            # Unpack data (dataset returns 4 values: left, right, name_left, name_right)
            img_left, img_right = data[0], data[1]
            img_left = img_left.to(device).float()
            img_right = img_right.to(device).float()
            
            # Forward pass
            recon_left, recon_right = model(img_left, img_right)
            
            # Calculate PSNR
            psnr_left = calculate_psnr(img_left, recon_left)
            psnr_right = calculate_psnr(img_right, recon_right)
            
            psnr_left_list.append(psnr_left)
            psnr_right_list.append(psnr_right)
            
            # Save images
            save_images(img_left, img_right, recon_left, recon_right, args.output_dir, idx)
            
            print(f"Image {idx+1}/{args.num_images}: PSNR_L={psnr_left:.2f}dB, PSNR_R={psnr_right:.2f}dB")
    
    # Print summary
    avg_psnr_left = np.mean(psnr_left_list)
    avg_psnr_right = np.mean(psnr_right_list)
    avg_psnr = (avg_psnr_left + avg_psnr_right) / 2
    
    print("\n" + "=" * 60)
    print("INFERENCE RESULTS SUMMARY")
    print("=" * 60)
    print(f"Average PSNR (Left):  {avg_psnr_left:.2f} dB")
    print(f"Average PSNR (Right): {avg_psnr_right:.2f} dB")
    print(f"Average PSNR (Both):  {avg_psnr:.2f} dB")
    print(f"Images saved to: {args.output_dir}")
    print("=" * 60)

if __name__ == '__main__':
    main()
