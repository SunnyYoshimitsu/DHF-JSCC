# Training and Testing Locations

## 📁 Directory Paths

### Checkpoints (Model Weights)
**Location:** `/home/past/sunny/DHF-JSCC-modified/checkpoints/10_13/`

**Structure:**
```
checkpoints/
├── 10_13/                          # Current training (with sigmoid)
│   ├── pkl/                        # Full checkpoint files
│   │   ├── epoch_0001_psnr_10.32dB.pkl
│   │   ├── epoch_0002_psnr_10.32dB.pkl
│   │   └── ... (up to epoch_0050)
│   └── pth/                        # State dict files (lighter)
│       ├── epoch_0001_psnr_10.32dB.pth  (237 MB each)
│       ├── epoch_0002_psnr_10.32dB.pth
│       └── ... (up to epoch_0050)
└── 10_13_no_sigmoid_backup/        # Old training (no sigmoid, grey images)
    └── ... (50 epochs of failed training)
```

**Current Size:** 237 MB per epoch × 50 epochs = ~11.8 GB total when complete

---

### Test Images (Inference Results)
**Location:** `/home/past/sunny/DHF-JSCC-modified/inference_results/results/1_images/`

**Structure:**
```
inference_results/
└── results/
    ├── 1_images/                   # Test reconstruction images
    │   ├── test_0000.png           # Side-by-side comparison (all 4 images)
    │   ├── test_0000_orig_left.png
    │   ├── test_0000_orig_right.png
    │   ├── test_0000_recon_left.png
    │   ├── test_0000_recon_right.png
    │   ├── test_0001.png
    │   ├── test_0001_orig_left.png
    │   ├── test_0001_orig_right.png
    │   ├── test_0001_recon_left.png
    │   ├── test_0001_recon_right.png
    │   └── ... (5 files per test image)
    └── KITTI_stereo_MSE_lambda:3e-05.csv  # PSNR results
```

**Image Layout:**
Each `test_XXXX.png` contains:
```
┌─────────────────┬─────────────────┐
│  Original Left  │ Reconstructed L │
├─────────────────┼─────────────────┤
│  Original Right │ Reconstructed R │
└─────────────────┴─────────────────┘
```

---

## 🎯 Current Status

### Training (Ongoing)
- **Log file:** `training_with_sigmoid.log`
- **Progress:** Epoch 2/50
- **Checkpoint dir:** `./checkpoints/10_13/pth/`
- **Latest:** `epoch_0002_psnr_10.32dB.pth`

### Testing (Not Started Yet)
- **Output dir:** `./inference_results/results/1_images/`
- **Currently contains:** Old test images from epoch 50 (no sigmoid, grey images)
- **Will be overwritten** when I run new inference

---

## 🔬 How to Run Testing

### Manual Testing Command:
```bash
cd /home/past/sunny/DHF-JSCC-modified
source .venv/bin/activate

# Test with a specific checkpoint (e.g., epoch 10)
python inference_snn.py \
  --checkpoint ./checkpoints/10_13/pth/epoch_0010_psnr_XX.XXdB.pth \
  --output_dir inference_results/results/1_images \
  --num_images 5
```

### What Gets Saved:
For `--num_images 5`, you'll get:
- 5 comparison images: `test_0000.png` to `test_0004.png`
- 20 individual images: 4 images × 5 test samples
  - Original left/right
  - Reconstructed left/right

### Output Example:
```
inference_results/results/1_images/
├── test_0000.png                   # Side-by-side comparison
├── test_0000_orig_left.png         # 128×128 RGB
├── test_0000_orig_right.png        # 128×128 RGB
├── test_0000_recon_left.png        # 128×128 RGB
├── test_0000_recon_right.png       # 128×128 RGB
├── test_0001.png
└── ... (25 files total for 5 test images)
```

---

## 📊 Monitoring

### Check Training Progress:
```bash
# Live monitoring
tail -f training_with_sigmoid.log

# Last 30 lines
tail -30 training_with_sigmoid.log

# List saved checkpoints
ls -lh checkpoints/10_13/pth/
```

### Check Disk Space:
```bash
du -sh checkpoints/10_13/
du -sh inference_results/
```

---

## 🎨 Expected Results

### Before Fix (No Sigmoid):
- **Checkpoints:** `checkpoints/10_13_no_sigmoid_backup/`
- **Images:** Grey, flat, 102-108 pixel values
- **PSNR:** 10.15 dB at epoch 50
- **Unique values:** 3

### After Fix (With Sigmoid):
- **Checkpoints:** `checkpoints/10_13/` (current training)
- **Images:** Full color, 0-255 range (expected)
- **PSNR:** 20-25+ dB at epoch 50 (expected)
- **Unique values:** 256+ (expected)

---

## ⏱️ Timing

- **Training:** ~2 minutes per epoch
- **50 epochs:** ~100 minutes (~1.7 hours)
- **Inference:** ~30 seconds for 5 test images
- **Started:** 04:26 AM
- **Expected completion:** ~06:00 AM

---

## 🗂️ File Sizes

### Checkpoints:
- **Per epoch (pkl):** ~474 MB (full checkpoint with optimizer state)
- **Per epoch (pth):** ~237 MB (model weights only)
- **50 epochs total:** ~11.8 GB (pth files)

### Test Images:
- **Per test image:** ~5 files = ~500 KB
- **10 test images:** ~5 MB
- **Current directory:** ~4 KB (old images cleared)

---

## 📝 Notes

1. **Old checkpoints preserved:** Moved to `10_13_no_sigmoid_backup/` for comparison
2. **Output directory reused:** Same `1_images/` folder, files will be overwritten
3. **CSV results:** PSNR scores saved to `KITTI_stereo_MSE_lambda:3e-05.csv`
4. **Batch size reduced:** 12 → 8 to avoid GPU memory issues with sigmoid
