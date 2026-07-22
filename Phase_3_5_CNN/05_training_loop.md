# 3.5.6 — Training Loop: Run It Yourself

> **Objective:** Open the Colab notebook, run the CNN training, and record the results.
> Reference: Slide 3.5.6 and 3.5.8 of `phase3_5_cnn_mnist.html`

---

## Setup: Open in Colab

```
https://colab.research.google.com/github/ashokmulchandani/CUDA-GPU-Colab-Computer-Vision-Project-Ashok-1/blob/main/1_Ashok_CUDA.ipynb
```

Navigate to the **CNN section** (approximately cells 70-110).

---

## Step-by-Step Run Instructions

### Step 1: Verify GPU
```python
!nvidia-smi
```
Expected: Tesla T4 with ~15GB VRAM. If you get K80, go to Runtime → Change runtime type → T4.

### Step 2: Write the CUDA code
Run the cell containing `%%writefile cnn_mnist.cu`. This creates a ~300-line file with all 10 kernels.

### Step 3: Compile
```bash
!nvcc cnn_mnist.cu -o cnn_mnist -lm
```
Expected output: (no errors). If you see errors, check the cell above was fully executed.

### Step 4: Train!
```bash
!./cnn_mnist
```
This trains for 10 epochs. Expected output per epoch:
```
Epoch  1 | Test Accuracy: ~89.5%
Epoch  2 | Test Accuracy: ~90.8%
Epoch  3 | Test Accuracy: ~91.8%
...
Epoch 10 | Test Accuracy: 96.29%
Training time: ~12.9 seconds
```

---

## Your Turn: Record Your Results

| Epoch | Accuracy | Cumulative Time |
|-------|----------|----------------|
| 1 | % | s |
| 2 | % | s |
| 3 | % | s |
| 4 | % | s |
| 5 | % | s |
| 6 | % | s |
| 7 | % | s |
| 8 | % | s |
| 9 | % | s |
| 10 | % | s |

---

## Experiment: Change Hyperparameters

**Task:** Run the following experiments. Modify the `#define` values at the top of `cnn_mnist.cu` and recompile.

### Experiment 1: Learning Rate
| LR | Accuracy at Epoch 5 | Final Accuracy | Training Time |
|----|---------------------|---------------|---------------|
| 0.001 | | | |
| 0.01 (default) | | 96.29% | |
| 0.05 | | | |

### Experiment 2: Filter Count
| Filters | Conv Params | FC Params | Total Params | Final Accuracy |
|---------|------------|-----------|-------------|---------------|
| 4 | | | | |
| 8 (default) | 208 | 11,530 | 11,738 | 96.29% |
| 16 | | | | |

### Experiment 3: Filter Size
| Kernel | Conv Output | FC Input | FC Params | Final Accuracy |
|--------|------------|----------|-----------|---------------|
| 3×3 | 26×26 | | | |
| 5×5 (default) | 24×24 | 1152 | 11,530 | 96.29% |
| 7×7 | 22×22 | | | |

---

## The Training Loop Explained

Here's what the code does for each batch of 64 images:

```
┌─────────────────────────────────────────┐
│ FORWARD PASS (5 kernels, GPU)           │
│                                         │
│ conv2d_forward → [64×8×24×24]           │
│ relu_forward   → [64×8×24×24]           │
│ maxpool_forward → [64×8×12×12]          │
│ fc_forward     → [64×10]                │
│ softmax        → [64×10] probs          │
├─────────────────────────────────────────┤
│ BACKWARD PASS (7 kernels, GPU)          │
│                                         │
│ softmax_backward → dL/d(fc_out)         │
│ fc_backward → dW_fc, db_fc, dL/d(pool)  │
│ maxpool_backward → dL/d(conv) [atomic!] │
│ relu_backward → zero where input≤0      │
│ conv2d_backward → dW_conv, db_conv      │
├─────────────────────────────────────────┤
│ WEIGHT UPDATE (4× sgd_update, GPU)      │
│                                         │
│ conv filters -= lr × dW_conv            │
│ conv bias    -= lr × db_conv            │
│ fc weights   -= lr × dW_fc              │
│ fc bias      -= lr × db_fc              │
└─────────────────────────────────────────┘
```

**Your Turn:** Which kernel do you think takes the MOST time? (Hint: check the inner loops) Run with Nsight profiling to verify:

```bash
!ncu --kernel-name conv2d_forward --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed ./cnn_mnist
```
