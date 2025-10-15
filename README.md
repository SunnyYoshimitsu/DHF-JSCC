# DHF-JSCC: Deep Hierarchical Fusion for Joint Source-Channel Coding

A deep learning-based distributed image compression framework that leverages side information for improved compression efficiency. This implementation supports multi-GPU training and includes both training and inference capabilities.

## Features

- **Distributed Compression**: Utilizes side information (stereo pairs) for enhanced compression
- **Multi-GPU Training**: DataParallel support for accelerated training on multiple GPUs
- **Rate-Distortion Optimization**: Balances compression rate and reconstruction quality
- **Multiple Datasets**: Support for KITTI and InStereo2K stereo datasets
- **Jupyter Notebook**: Interactive training and visualization notebook included

## Requirements

- Python 3.8+
- PyTorch 2.8.0+ with CUDA support
- torchvision
- numpy
- matplotlib
- tqdm
- pyyaml

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SunnyYoshimitsu/DHF-JSCC.git
cd DHF-JSCC
```

2. Install dependencies:
```bash
pip install torch torchvision numpy matplotlib tqdm pyyaml
```

## Datasets

### KITTI Dataset
Download from: https://www.cvlibs.net/datasets/kitti/

### InStereo2K Dataset
Download from Baidu Netdisk: https://pan.baidu.com/s/1-QGEbj4Qnw6YxPTolvoHbg (code: a2a2)

## Usage

### Training

Run the main training script:
```bash
python main22.py
```

For multi-GPU training (automatically detected):
```bash
python main22.py  # Uses DataParallel if multiple GPUs available
```

### Inference/Testing

Run inference on trained models:
```bash
python main22.py --test  # Set test: true in config.yaml
```

### Configuration

Modify `config.yaml` to adjust:
- Training parameters (epochs, batch size, learning rate)
- Dataset paths
- Lambda for rate-distortion trade-off
- GPU settings

### Jupyter Notebook

Use `DHF-JSCC.IPYNB` for interactive training, visualization, and experimentation.

## Project Structure

```
DHF-JSCC/
├── main22.py                 # Main training/inference script
├── config.yaml              # Configuration file
├── DHF-JSCC.IPYNB          # Jupyter notebook
├── model_d_fusion2.py       # DHF-JSCC model architecture
├── hyperfusion_2.py         # Hyperprior fusion components
├── entropy_model.py         # Entropy modeling
├── gaussian_entropy_model.py # Gaussian entropy model
├── conditional_entropy_model.py # Conditional entropy model
├── channel.py               # Channel modeling
├── balle2017/               # Baseline compression models
├── dataset/                 # Dataset loaders
│   ├── PairKitti.py
│   ├── InStereo2K.py
│   └── data_paths/
└── checkpoints/             # Saved model checkpoints
```

## Key Modifications

- Added DataParallel support for multi-GPU training
- Improved checkpoint saving/loading with model unwrapping
- Enhanced configuration management
- Added comprehensive logging and monitoring scripts

## Results

The model achieves competitive PSNR and MS-SSIM metrics on stereo image compression tasks, with significant improvements when leveraging side information.

## Citation

If you use this code, please cite the original DHF-JSCC paper and acknowledge the modifications for multi-GPU support.
