# 3.5.2/3.5.3 — Conv2D CUDA Kernel Walkthrough

> **Objective:** Understand every line of the conv2d_forward kernel and how threads are mapped to output pixels.
> Reference: Slide 3.5.3 of `phase3_5_cnn_mnist.html`

---

## The Kernel (Full Code)

```cuda
__global__ void conv2d_forward(float *input, float *filters, float *bias, float *output,
                                int batch) {
    // ── Step 1: Which output pixel am I responsible for? ──
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 0..23
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 0..23
    int idx = blockIdx.z;
    int b = idx / NUM_FILTERS;     // which image in the batch?
    int f = idx % NUM_FILTERS;     // which filter?

    // ── Step 2: Guard — only compute if within bounds ──
    if (row < CONV_H && col < CONV_W && b < batch) {
        float sum = bias[f];

        // ── Step 3: 5×5 weighted sum ──
        for (int fh = 0; fh < K; fh++)
            for (int fw = 0; fw < K; fw++)
                sum += input[b*IN_H*IN_W + (row+fh)*IN_W + (col+fw)] *
                       filters[f*K*K + fh*K + fw];

        // ── Step 4: Write result ──
        output[b*NUM_FILTERS*CONV_H*CONV_W + f*CONV_H*CONV_W + row*CONV_W + col] = sum;
    }
}
```

---

## Thread Mapping — 3D Grid

```
dim3 threads(16, 16);     // 256 threads per block
dim3 blocks(2, 2, 512);   // CONV_W=24, CONV_H=24, BATCH=64, FILTERS=8

blockIdx.x → which 16-column chunk?   (0 or 1, covering 0-15 and 16-23)
blockIdx.y → which 16-row chunk?      (0 or 1, covering 0-15 and 16-23)
blockIdx.z → which (batch, filter)?   (0..511 for 64×8=512 combinations)

Each thread = ONE output pixel for ONE filter for ONE image.
```

---

## Your Turn: Trace One Thread

**Given:** Thread (threadIdx.x=3, threadIdx.y=5) in block (blockIdx.x=1, blockIdx.y=0, blockIdx.z=258)

**Task 1:** Which output pixel does this thread compute?
- col = ?
- row = ?
- batch b = ?
- filter f = ?

**Task 2:** What input pixels does this thread read? List the 5×5 = 25 (row, col) coordinates.

**Task 3:** What filter weights does it use? Which 25 values from the filters array?

**Task 4:** If bias[f] = 0.1 and the input patch is all zeros except input[row+2][col+2] = 1.0, and filters[f][2][2] = 0.5, what is the output value?

---

## Launch Configuration — Why These Numbers?

| Parameter | Value | Why? |
|-----------|-------|------|
| Threads per block | 16×16 = 256 | Warp size = 32. 256 = 8 warps. Good occupancy for T4. |
| Blocks in x | (24+15)/16 = 2 | Ceiling division covers all 24 columns |
| Blocks in y | (24+15)/16 = 2 | Covers all 24 rows |
| Blocks in z | 64 × 8 = 512 | One slice per (batch, filter) pair |
| Total threads | 2×2×512×256 = **524,288** | All launched simultaneously! |

---

## Key Takeaways

1. Each thread computes **one number** — the convolution output at one (row, col, filter, batch) position
2. The 3D grid separates: **x=column, y=row, z=(batch,filter)** so no if-else spaghetti
3. The inner double loop does 25 multiplications and 25 additions — this is the "heavy lifting"
4. The DRAM throughput is only 1.08% — **this kernel is compute-bound, not memory-bound** (Nsight confirms)
