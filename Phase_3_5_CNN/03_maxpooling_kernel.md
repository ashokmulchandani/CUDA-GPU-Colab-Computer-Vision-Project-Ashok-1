# 3.5.4 — Max Pooling CUDA Kernel

> **Objective:** Understand how max pooling downsamples feature maps and why we save the index for backprop.
> Reference: Slide 3.5.4 of `phase3_5_cnn_mnist.html`

---

## The Concept

After Conv2D + ReLU, we have 64×8×24×24 = 294,912 values. That's too many to feed into the FC layer. MaxPool 2×2 with stride 2 reduces it by 4×:

```
Input:  24×24              Output: 12×12
┌──┬──┬──┬──┐             ┌────┬────┐
│3 │7 │2 │1 │             │ 7  │ 9  │   Each 2×2 window →
│1 │5 │9 │4 │    →        │    │    │   keep only the MAX
├──┼──┼──┼──┤             ├────┼────┤
│2 │0 │4 │8 │             │ 6  │ 12 │
│6 │3 │1 │12│             │    │    │
└──┴──┴──┴──┘             └────┴────┘

First window [3,7,1,5]: max = 7 (at position 0,1 inside the window)
Saved index = (0×24+1) = 1  ← CRITICAL for backprop!
```

---

## The Kernel

```cuda
__global__ void maxpool_forward(float *input, float *output, int *indices, int batch) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = blockIdx.z;
    int b = idx / NUM_FILTERS;
    int f = idx % NUM_FILTERS;

    if (row < POOL_H && col < POOL_W && b < batch) {
        float max_val = -1e9;
        int max_pos = 0;
        for (int ph = 0; ph < 2; ph++)
            for (int pw = 0; pw < 2; pw++) {
                int ih = row*2 + ph, iw = col*2 + pw;
                int pos = b*NUM_FILTERS*CONV_H*CONV_W + f*CONV_H*CONV_W
                         + ih*CONV_W + iw;
                if (input[pos] > max_val) {
                    max_val = input[pos];
                    max_pos = ih*CONV_W + iw;  // save the winner's position!
                }
            }
        int out_pos = b*NUM_FILTERS*POOL_H*POOL_W + f*POOL_H*POOL_W
                     + row*POOL_W + col;
        output[out_pos] = max_val;
        indices[out_pos] = max_pos;  // stored for backprop
    }
}
```

---

## Why Save the Index? — The Backward Pass

During backprop, only the **winning neuron** gets the gradient:

```cuda
__global__ void maxpool_backward(float *grad_out, float *grad_in, int *indices,
                                  int batch) {
    // grad_out[pool_pos] = how much this pooled value contributed to the loss
    // indices[pool_pos]  = which input position was the max

    int in_pos = b*NUM_FILTERS*CONV_H*CONV_W + f*CONV_H*CONV_W + indices[pool_pos];
    atomicAdd(&grad_in[in_pos], grad_out[pool_pos]);  // gradient → winner only!
}
```

The other 3 positions in the 2×2 window get ZERO gradient — they didn't contribute.

---

## Your Turn: MaxPool by Hand

Given this 4×4 input:

```
3  7  2  1
1  5  9  4
2  0  4  8
6  3  1  12
```

**Task 1:** Apply 2×2 MaxPool with stride 2. Fill in:

| Output position | 2×2 window values | Max value | Winner position (in original) |
|----------------|-------------------|-----------|-------------------------------|
| (0,0) | 3,7,1,5 | 7 | (0,1) |
| (0,1) | | | |
| (1,0) | | | |
| (1,1) | | | |

**Task 2:** If the gradient coming back is:
```
0.1  0.2
0.3  0.4
```
Which input positions receive non-zero gradients? Fill them in:

```
?  ?  ?  ?
?  ?  ?  ?
?  ?  ?  ?
?  ?  ?  ?
```
