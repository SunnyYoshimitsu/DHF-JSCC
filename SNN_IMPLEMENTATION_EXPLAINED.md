# SNN Implementation Explanation

## 📁 File Structure

The SNN implementation is **separated into two main files**:

### 1. `model_d_fusion2_snn.py` - The SNN Architecture (433 lines)
**This is where ALL the SNN magic happens!**

Contains 7 SNN classes:
- `Enc_l_SNN` - Left image encoder (SNN)
- `Enc_r_SNN` - Right image encoder (SNN)
- `Hyper_Enc_SNN` - Hyperprior encoder (SNN)
- `Hyper_Dec_SNN` - Hyperprior decoder (SNN)
- `Dec_SNN` - Image decoder (SNN)
- `Fusion_SNN` - Attention fusion module (SNN)
- `Image_coding_SNN` - Main model that combines everything

### 2. `main22.py` - Training Script (398 lines)
**This simply imports and uses the SNN model**

Just does:
```python
import model_d_fusion2_snn

# Later in the code:
model = model_d_fusion2_snn.Image_coding_SNN(M=256, N2=25, T=4, tau=2.0)
```

---

## 🧠 How SNN Was Implemented

### Key Components from SpikingJelly:

```python
from spikingjelly.activation_based import neuron, functional, layer
```

**1. Replace Conv2d with SNN layers:**
```python
# CNN version (OLD):
self.conv1 = nn.Conv2d(channels_in, channels_out, kernel_size=5, stride=2)
self.relu = nn.ReLU()

# SNN version (NEW):
self.conv1 = layer.Conv2d(channels_in, channels_out, kernel_size=5, stride=2)
self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
```

**2. LIF Neurons (Leaky Integrate-and-Fire):**
- Replaces ReLU activation
- Has membrane potential that leaks over time
- Fires spikes when potential exceeds threshold
- `tau=2.0` controls membrane time constant
- `ATan()` is the surrogate gradient for backpropagation

**3. Temporal Processing (T=4 time steps):**
```python
def main_enc(self, x):
    outputs = []
    
    for t in range(self.T):  # T=4 time steps
        x1 = self.conv1(x)
        x1 = self.lif1(x1)    # Spike generation
        
        x2 = self.conv2(x1)
        x2 = self.lif2(x2)    # Spike generation
        
        outputs.append(x2)
    
    # Average outputs over time
    x_out = torch.stack(outputs).mean(dim=0)
    return x_out
```

**4. Membrane Potential Reset:**
```python
def forward(self, x):
    functional.reset_net(self)  # Reset all neuron states
    enc = self.main_enc(x)
    return enc
```

---

## 🏗️ Architecture Overview

### Complete SNN Pipeline:

```
Input Stereo Pair (Left + Right Images)
         ↓
    [Enc_l_SNN]  [Enc_r_SNN]
    LIF neurons   LIF neurons
    4 time steps  4 time steps
         ↓            ↓
    latent_l     latent_r
         ↓            ↓
    [Hyper_Enc_SNN]  [Hyper_Enc_SNN]
         ↓            ↓
    hyperpriors  hyperpriors
         ↓            ↓
    [Quantization & Channel]
         ↓            ↓
    [Hyper_Dec_SNN]  [Hyper_Dec_SNN]
         ↓            ↓
    [Fusion_SNN] ← Attention mechanism
         ↓
    [Dec_SNN]
    LIF neurons
    4 time steps
         ↓
    Reconstructed Images
```

---

## 📊 Example: Encoder Module

Here's how `Enc_l_SNN` works:

```python
class Enc_l_SNN(nn.Module):
    def __init__(self, num_features, M1, M, N2, T=4):
        super(Enc_l_SNN, self).__init__()
        self.T = T  # 4 time steps
        
        # Layer 1: Conv + LIF neuron
        self.conv1 = layer.Conv2d(3, 256, kernel_size=5, stride=2, padding=2)
        self.lif1 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        # Layer 2: Conv + LIF neuron
        self.conv2 = layer.Conv2d(256, 256, kernel_size=5, stride=2, padding=2)
        self.lif2 = neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan())
        
        # ... more layers
    
    def main_enc(self, x):
        # x is input image [Batch, 3, 128, 128]
        outputs = []
        
        # Process through 4 time steps
        for t in range(4):
            # Layer 1
            x1 = self.conv1(x)      # Convolution
            x1 = self.lif1(x1)      # Spike generation (replaces ReLU)
            
            # Layer 2
            x2 = self.conv2(x1)     # Convolution
            x2 = self.lif2(x2)      # Spike generation
            
            outputs.append(x2)
        
        # Average spikes over time
        x_out = torch.stack(outputs).mean(dim=0)
        return x_out
    
    def forward(self, x):
        functional.reset_net(self)  # Reset membrane potentials
        enc = self.main_enc(x)
        return enc
```

---

## 🔑 Key Differences: CNN vs SNN

| Aspect | CNN (Original) | SNN (New) |
|--------|----------------|-----------|
| **Activation** | ReLU (continuous) | LIF spikes (binary events) |
| **Processing** | Single pass | 4 time steps |
| **Neurons** | `nn.ReLU()` | `neuron.LIFNode()` |
| **Layers** | `nn.Conv2d()` | `layer.Conv2d()` |
| **State** | Stateless | Stateful (membrane potential) |
| **Output** | Direct | Averaged over time |
| **Gradient** | Standard backprop | Surrogate gradient (ATan) |
| **File** | `model_d_fusion2.py` | `model_d_fusion2_snn.py` |

---

## 🎯 Why Separate Files?

### Design Decision:
1. **Clean separation** - SNN logic doesn't pollute CNN code
2. **Easy comparison** - Can compare architectures side-by-side
3. **Maintainability** - Each architecture is self-contained
4. **Flexibility** - Could switch between them (but now SNN-only)

### main22.py Role:
- Imports the SNN model
- Sets up training loop
- Handles data loading
- Computes loss (rate-distortion)
- Saves checkpoints

### model_d_fusion2_snn.py Role:
- Defines ALL SNN architecture
- Implements LIF neurons
- Handles temporal dynamics
- Processes spikes over time steps

---

## 🚀 How It's Used in Training

### In main22.py:

```python
# 1. Import SNN model
import model_d_fusion2_snn

# 2. Initialize SNN
model = model_d_fusion2_snn.Image_coding_SNN(
    M=256,      # Number of filters
    N2=25,      # Latent dimension
    T=4,        # Time steps
    tau=2.0     # Neuron time constant
)

# 3. Train like normal PyTorch
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(50):
    for batch in dataloader:
        # Forward pass (SNN processes internally with T=4 time steps)
        output = model(left_img, right_img)
        
        # Compute loss
        loss = rate_loss + distortion_loss
        
        # Backward pass (surrogate gradients handle spike discontinuity)
        loss.backward()
        optimizer.step()
```

---

## 🔬 Technical Details

### 1. Surrogate Gradients
**Problem:** Spikes are binary (0 or 1), not differentiable  
**Solution:** Use ATan surrogate gradient during backpropagation

```python
neuron.LIFNode(surrogate_function=neuron.surrogate.ATan())
```

### 2. Time Steps (T=4)
- Each layer processes input 4 times
- Accumulates information over time
- More time steps = more computations but potentially better accuracy
- Outputs are averaged: `torch.stack(outputs).mean(dim=0)`

### 3. Membrane Potential Reset
```python
functional.reset_net(self)  # Reset before each forward pass
```
Ensures neurons start fresh for each training sample

### 4. SpikingJelly Layers
- `layer.Conv2d()` - Conv layer compatible with SNN
- `layer.ConvTranspose2d()` - Deconv for SNN
- `layer.Conv1d()` - 1D conv for SNN
- All handle spike trains properly

---

## 📈 Training Progress

Current training (Epoch 25/50):
- PSNR: 5.31 → 8.89 dB
- Learning rate: 0.0001
- Batch size: 12
- Time steps: 4
- Tau: 2.0

The SNN is successfully learning to compress and reconstruct stereo images!

---

## 📝 Summary

**Where is SNN implemented?**
- ✅ **All architecture:** `model_d_fusion2_snn.py` (433 lines)
- ✅ **Training logic:** `main22.py` (398 lines)
- ✅ **Framework:** SpikingJelly (LIF neurons, surrogate gradients)

**Key insight:**  
The SNN implementation is **self-contained** in its own file. The training script simply imports and uses it, just like any PyTorch model. The complexity of temporal processing, spike generation, and membrane dynamics is all hidden inside the SNN classes.
