# DHF-JSCC-SNN: Spiking Neural Network Implementation

Deep Hybrid Fusion Joint Source-Channel Coding converted to Spiking Neural Networks using SpikingJelly.

## Overview

This project implements a Spiking Neural Network (SNN) version of the DHF-JSCC stereo image compression model. The original CNN architecture has been converted to use biologically-inspired LIF (Leaky Integrate-and-Fire) neurons with temporal processing.

## Key Features

- ✅ **Complete SNN Architecture**: All CNN layers replaced with SNN equivalents
- ✅ **SpikingJelly Framework**: Built on SpikingJelly v0.0.0.0.14
- ✅ **Temporal Processing**: Configurable time steps (T=4, T=8)
- ✅ **Multi-GPU Support**: DataParallel training on 2 GPUs
- ✅ **Stereo Image Compression**: Joint compression of left/right stereo pairs

## Architecture

### SNN Components
- **Neurons**: LIF (Leaky Integrate-and-Fire) with tau=2.0
- **Layers**: Convolutional layers via `layer.Conv2d` and `layer.ConvTranspose2d`
- **Activation**: Spike-based activation through `neuron.LIFNode`
- **Surrogate Gradient**: ATan surrogate function for backpropagation
- **Time Steps**: T=4 or T=8 temporal iterations

### Model Structure
```
Encoder (Left/Right) → Quantization → Channel → Decoder → Reconstruction
```

## Installation

### Requirements
```bash
# Python 3.12+
pip install torch==2.8.0+cu128
pip install spikingjelly==0.0.0.0.14
pip install torchvision
pip install PyYAML
pip install Pillow
pip install pytorch-msssim
pip install pandas
```

### Dataset
- **KITTI Stereo 2015** dataset
- Place in `dataset/` directory
- Update paths in `dataset/data_paths/KITTI_stereo_train.txt`

## Usage

### Training
```bash
# Train with default settings (T=8, tau=2.0, 2 GPUs)
python main22.py --config config.yaml

# Monitor GPU usage
./monitor_gpu.sh

# Check training progress
tail -f training_T8_tau2.log
```

### Inference
```bash
# Run inference on test set
python inference_snn.py \
  --checkpoint checkpoints/10_13/pth/epoch_0022_psnr_10.33dB.pth \
  --output_dir outputs/results/10_13_T8 \
  --num_images 10
```

## Configuration

### Main Parameters (config.yaml)
```yaml
# Model
model_type: 'SNN'
num_filters: 256
snn_time_steps: 8        # Temporal processing steps
snn_neuron_tau: 2.0      # LIF membrane time constant

# Training
epochs: 10000
train_batch_size: 8      # Per GPU
lr: 0.0001
lambda: 0.00003          # Rate-distortion trade-off

# Hardware
cuda: True               # Uses 2 GPUs with DataParallel
```

## Experimental Results

### Experiments Conducted

| Configuration | Epochs | Unique Pixels | Pixel Range | PSNR (dB) | Status |
|--------------|--------|---------------|-------------|-----------|---------|
| tau=2.0, T=4 | 237 | 29 | 109-204 (95) | 10.36 | ⚠️ Best |
| tau=8.0, T=4 | 5 | 2 | 126-127 (1) | 10.32 | ❌ Failed |
| tau=2.0, T=8 | 22 | 3 | 123-129 (6) | 9.93 | ❌ Poor |

### Key Findings

1. **tau=2.0, T=4** produced the best results with 29 unique pixel values
2. **Higher tau (8.0)** caused very slow learning and poor diversity
3. **Higher T (8)** did not improve results, likely due to over-smoothing
4. **Overall quality** still needs improvement (target PSNR: 25-30 dB)

### Known Issues

- Output images have low diversity (limited unique pixel values)
- PSNR around 10 dB (vs target 25-30 dB for CNNs)
- Grey/dark reconstruction outputs
- Likely issues with loss function (MSE) for SNNs

## Project Structure

```
DHF-JSCC-SNN/
├── model_d_fusion2_snn.py          # Main SNN model
├── inference_snn.py                # Inference script
├── main22.py                       # Training script
├── config.yaml                     # Configuration
├── gaussian_entropy_model.py       # Entropy coding
├── channel.py                      # Channel simulation
├── ops.py                          # Utility operations
├── dataset/                        # Dataset loaders
│   ├── PairKitti.py
│   └── InStereo2K.py
├── checkpoints/                    # Model checkpoints (excluded)
├── outputs/                        # Results (excluded)
└── docs/
    ├── SNN_IMPLEMENTATION_EXPLAINED.md
    ├── T8_TRAINING_LOG.md
    └── CHECKPOINT_AND_OUTPUT_LOCATIONS.md
```

## Documentation

- **[SNN_IMPLEMENTATION_EXPLAINED.md](SNN_IMPLEMENTATION_EXPLAINED.md)**: Detailed explanation of CNN to SNN conversion
- **[T8_TRAINING_LOG.md](T8_TRAINING_LOG.md)**: Experiments with T=8 time steps
- **[CHECKPOINT_AND_OUTPUT_LOCATIONS.md](CHECKPOINT_AND_OUTPUT_LOCATIONS.md)**: File organization

## Hardware Requirements

- **GPU**: 2x NVIDIA GPUs with 11GB+ memory (tested on RTX 2080 Ti)
- **Memory**: ~8GB per GPU for T=8
- **Storage**: ~300MB per checkpoint

## Future Work

### Potential Improvements
1. **Loss Function**: Try perceptual loss or SSIM loss instead of MSE
2. **Learning Rate**: Adjust or use adaptive scheduling
3. **Architecture**: Design SNN-specific layers
4. **Hybrid Approach**: Combine CNN and SNN layers
5. **Training Techniques**: STDP, local learning rules
6. **Time Steps**: Optimize T parameter for speed/quality trade-off

## Citation

Original DHF-JSCC paper:
```bibtex
@article{dhf-jscc,
  title={Deep Hybrid Fusion for Joint Source-Channel Coding},
  author={...},
  journal={...},
  year={...}
}
```

SpikingJelly framework:
```bibtex
@misc{spikingjelly,
  title={SpikingJelly},
  author={Fang, Wei and Chen, Yanqi and Ding, Jianhao and Yu, Zhaofei and Zhou, Yonghong and others},
  year={2020},
  howpublished={\url{https://github.com/fangwei123456/spikingjelly}}
}
```

## License

[Specify your license here]

## Acknowledgments

- Original DHF-JSCC implementation
- SpikingJelly framework developers
- KITTI dataset providers

## Contact

[Your contact information]

---

**Status**: Experimental - SNN implementation complete, but quality needs improvement for practical use.
