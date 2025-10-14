# Training with T=8 - October 13, 2025

## Configuration
- **Time Steps (T)**: 8 (increased from 4)
- **Tau**: 2.0 (reverted from 8.0)
- **GPUs**: 2 GPUs with DataParallel
- **Batch Size**: 8 per GPU (16 effective)
- **Epochs**: 10,000
- **Log File**: training_T8_tau2.log

## Rationale for T=8

### Why Increase T?
With tau=2.0, membrane potential decays 50% per time step:
- **T=4**: Information retention over time: 100% → 50% → 25% → 12.5% → 6.25%
  - Early time step information almost completely lost
  
- **T=8**: Information retention: 100% → 50% → 25% → 12.5% → 6.25% → 3.1% → 1.6% → 0.8%
  - More integration time (double the duration)
  - Better temporal accumulation
  - More opportunities for spikes to accumulate

### Expected Benefits
1. **Better temporal integration**: More time steps = better information accumulation
2. **Smoother gradients**: Averaged over 8 steps instead of 4
3. **More diverse outputs**: Hopefully >29 unique pixel values
4. **Standard practice**: T=8 to T=16 is common in SNN literature

## Resource Usage

### GPU Memory
- **GPU 0**: 7.9 GB / 11 GB (70%)
- **GPU 1**: 7.6 GB / 11 GB (67%)
- Previous T=4: ~4.5 GB per GPU
- Increase: ~75% more memory (as expected)

### Training Speed
- T=4: ~60-70 seconds per epoch
- T=8: ~120-150 seconds per epoch (2x slower)
- Still manageable with 2 GPUs

## Initial Results

### Epoch 1
- **PSNR**: 10.31 dB
- **Status**: Training from scratch with T=8

### Comparison with Previous Attempts
1. **tau=2.0, T=4** (epoch 237): 
   - PSNR: 10.36 dB
   - 29 unique pixel values, range 109-204
   
2. **tau=8.0, T=4** (epoch 5):
   - PSNR: 10.32 dB
   - 2 unique pixel values, range 126-127 (FAILED)
   
3. **tau=2.0, T=8** (epoch 1):
   - PSNR: 10.31 dB
   - To be tested...

## Monitoring Plan

1. **Every 10 epochs**: Check PSNR progression
2. **Epoch 50**: Run inference test to check pixel diversity
3. **Epoch 100**: Full evaluation if showing improvement
4. **Epoch 200**: Compare with T=4 results

## Success Criteria

To consider T=8 successful:
- **>35 unique pixel values** (vs 29 with T=4)
- **>120 pixel range** (vs 95 with T=4)
- **>11 dB PSNR** (vs 10.36 with T=4)
- **Visible improvement** in reconstructed images

## Backup Plan

If T=8 doesn't show improvement after 100 epochs:
- Try T=16 (more extreme)
- Change loss function (perceptual loss, SSIM loss)
- Adjust learning rate
- Consider hybrid CNN-SNN approach
