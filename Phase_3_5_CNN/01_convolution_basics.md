# 3.5.1 — Understand 2D Convolution (By Hand)

> **Objective:** Compute a 2D convolution by hand to understand what the CUDA kernel does.
> Reference: Slide 3.5.2 of `phase3_5_cnn_mnist.html`

---

## The Concept

A 2D convolution slides a **filter** (kernel) over an **input** image. At each position, it multiplies filter weights by input pixels and sums them up.

```
Input (5×5)          Filter (3×3)        Output (3×3)
1  2  3  0  1        1  0 -1             ?  ?  ?
0  1  2  3  0    ×   0  1  0        =    ?  ?  ?
3  0  1  2  1       -1  0  1             ?  ?  ?
2  3  0  1  0
1  2  1  0  3

Output size = input_size - filter_size + 1 = 5 - 3 + 1 = 3
```

---

## Worked Example: First Output Pixel (position 0,0)

Place the 3×3 filter on the top-left of the input:

```
Input patch:     Filter:       Multiply:
1  2  3          1  0 -1       1×1 + 2×0 + 3×(-1) = 1 + 0 - 3 = -2
0  1  2    ×     0  1  0   =   0×0 + 1×1 + 2×0    = 0 + 1 + 0 = 1
3  0  1         -1  0  1       3×(-1)+ 0×0+ 1×1   = -3+ 0 + 1= -2
                                                      ─────────────
                                                      SUM = -3
```

Output[0,0] = -3

---

## Your Turn

**Task:** Compute the remaining 8 output values by hand.

| Position | Input Patch (top-left at row,col) | Show multiplications | Output |
|----------|----------------------------------|---------------------|--------|
| (0,0) | rows 0-2, cols 0-2 | 1×1+2×0+3×(-1)+0×0+1×1+2×0+3×(-1)+0×0+1×1 | **-3** |
| (0,1) | rows 0-2, cols 1-3 | | |
| (0,2) | rows 0-2, cols 2-4 | | |
| (1,0) | | | |
| (1,1) | | | |
| (1,2) | | | |
| (2,0) | | | |
| (2,1) | | | |
| (2,2) | | | |

**Task:** What does this specific filter detect? (Hint: look at the pattern of 1, 0, -1)

---

## From Hand Calculation to CUDA

The hand calculation above is exactly what the CUDA kernel does — but 294,912 times in parallel:

```cuda
float sum = bias[f];
for (int fh = 0; fh < K; fh++)           // K = filter size
    for (int fw = 0; fw < K; fw++)
        sum += input[(row+fh)*IN_W + (col+fw)] *  // <-- input pixel
               filters[f*K*K + fh*K + fw];         // <-- filter weight
output[...] = sum;  // one pixel done!
```

**Your Turn:** Circle the parts of the CUDA code that correspond to your hand calculation above. Which variable is the input pixel? Which is the filter weight?

---

## Key Formulas to Remember

| Formula | MNIST CNN |
|---------|-----------|
| Output height = input_h - filter_h + 1 | 28 - 5 + 1 = **24** |
| Output width = input_w - filter_w + 1 | 28 - 5 + 1 = **24** |
| Multiplications per output pixel = K² | 5 × 5 = **25** |
| Output pixels per image per filter = H_out × W_out | 24 × 24 = **576** |
| Total output values = batch × filters × H_out × W_out | 64 × 8 × 24 × 24 = **294,912** |
