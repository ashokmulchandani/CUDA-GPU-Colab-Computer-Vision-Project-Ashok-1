# 3.5.7 — Benchmarks, Profiling & Reflection

> **Objective:** Compare CUDA CNN vs PyTorch, profile the bottleneck kernel, and reflect on what you built.
> Reference: Slides 3.5.7 and 3.5.9 of `phase3_5_cnn_mnist.html`

---

## Benchmark Comparison

Run the PyTorch comparison cell in the notebook. Fill in:

| Metric | CUDA CNN (from scratch) | PyTorch CNN |
|--------|------------------------|-------------|
| Architecture | Conv(5×5,8)→ReLU→Pool→FC(1152→10) | Same |
| Parameters | 11,738 | 11,738 |
| Epoch 1 Accuracy | % | % |
| Epoch 5 Accuracy | % | % |
| Epoch 10 Accuracy | % | % |
| Training Time | s | s |
| Speedup | — | — |

**Question:** Why is the CUDA version 7.4× faster than PyTorch when both run the SAME architecture on the SAME GPU? Write your answer:

```
(Your explanation here)
```

---

## Nsight Compute Profiling

Run the profiling cell:

```bash
!ncu --kernel-name conv2d_forward --launch-skip 0 --launch-count 1 \
  --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed \
  ./cnn_mnist
```

Record your results:

| Metric | Value | What it means |
|--------|-------|---------------|
| SM Throughput | % | % of GPU compute being used |
| DRAM Throughput | % | % of GPU memory bandwidth being used |
| Launch Config | (?, ?, ?)×(?, ?, ?) | Grid and block dimensions |
| Device | | GPU model |

**Question:** The DRAM throughput is only ~1%. Why? Is this good or bad?

```
(Your answer here)

Hint: If the kernel does 25 multiplications for every memory read, it's
doing mostly COMPUTE, not memory transfers. Low DRAM throughput + decent
SM throughput = compute-bound. This is actually fine for a small kernel!
```

---

## CNN vs MLP: The Final Comparison

| Metric | Phase 3 MLP | Phase 3.5 CNN |
|--------|------------|---------------|
| Architecture | 784→128→64→10 | Conv→ReLU→Pool→FC→Softmax |
| Parameters | 109,386 | 11,738 |
| Accuracy | 95.55% | 96.29% |
| Training time | 6.8s | 12.9s |
| Speedup vs PyTorch | 30× | 7.4× |
| Number of kernels | 6 | 10 |

**Question:** The CNN has 9.3× FEWER parameters but HIGHER accuracy. Why?

```
(Your answer here)

Hint: Think about what the MLP sees vs what the CNN sees. The MLP gets
784 independent numbers. The CNN sees a 28×28 GRID — it knows that pixel
(3,4) is next to pixel (3,5). That spatial knowledge is built into the
convolution operation itself.
```

---

## Reflection: What You Built

**Check all that apply:**

- [ ] I can explain how a 2D convolution works (filter sliding, weighted sum)
- [ ] I understand the conv2d_forward CUDA kernel — especially the 3D grid and thread mapping
- [ ] I can trace an input pixel through Conv→ReLU→Pool→FC→Softmax
- [ ] I understand WHY max pooling saves indices (for backprop's atomicAdd)
- [ ] I ran the full training loop and got ~96% accuracy
- [ ] I ran at least one hyperparameter experiment (different LR, filters, or kernel size)
- [ ] I profiled the conv2d_forward kernel with Nsight Compute
- [ ] I can explain why weight sharing gives the CNN 9× fewer params than the MLP
- [ ] I understand why the CNN outperforms the MLP despite having fewer parameters

---

## Bonus Challenge: Tiled Convolution

The current `conv2d_forward` kernel reads directly from global memory — that's why DRAM throughput is 1.08%. A **tiled convolution** loads a block of input into `__shared__` memory and reuses it across threads.

```cuda
__global__ void conv2d_forward_tiled(...) {
    __shared__ float tile[TILE_H][TILE_W];  // shared memory

    // Step 1: Cooperatively load input tile into shared memory
    // Step 2: __syncthreads()
    // Step 3: Compute convolution using the tile (fast!)
    // Step 4: __syncthreads()
}
```

**Challenge:** Write a tiled version of `conv2d_forward`. Compare DRAM throughput with the naive version using Nsight Compute. Target: >50% DRAM throughput.

> Hint: A TILE_H × TILE_W tile of input can be shared across all threads in a block since neighboring output pixels read overlapping input regions.
