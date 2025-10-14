# DHF-JSCC: Spiking Neural Network Implementation

This repository contains a **Spiking Neural Network (SNN)** implementation of DHF-JSCC for stereo image compression using the SpikingJelly framework.

## 🧠 Architecture

- **Model Type:** Spiking Neural Network (SNN)
- **Framework:** SpikingJelly
- **Neuron Type:** LIF (Leaky Integrate-and-Fire)
- **Time Steps:** 4
- **Baseline:** Based on bls17 architecture (Ballé et al. 2017)

## 📦 Datasets

The KITTI dataset can be downloaded at https://www.cvlibs.net/datasets/kitti/
The InStereo2K dataset is shared on Baidu Netdisk：https://pan.baidu.com/s/1-QGEbj4Qnw6YxPTolvoHbg  （code：a2a2）

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
python main22.py --config config.yaml
```

### Inference

```bash
python inference_snn.py --checkpoint ./checkpoints/10_13/pth/epoch_XXXX_psnr_XX.XXdB.pth
```

## 📖 Documentation

- `README_SNN.md` - Detailed SNN implementation guide
- `SNN_CONVERSION_GUIDE.md` - Technical details on CNN to SNN conversion
- `CLEANUP_SUMMARY.md` - Repository cleanup history

## 🔧 Configuration

Edit `config.yaml` to adjust:
- Training epochs
- Learning rate
- Batch size
- SNN parameters (time steps, tau)
- Lambda (rate-distortion tradeoff)

