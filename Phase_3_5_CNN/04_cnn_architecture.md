# 3.5.5 — CNN Architecture: Dimension Tracking

> **Objective:** Trace the exact dimensions through every layer of the CNN and compute the parameter count at each stage.
> Reference: Slide 3.5.5 of `phase3_5_cnn_mnist.html`

---

## The Complete Pipeline

```
Input(28×28×1) ──→ Conv2D(5×5, 8 filters) ──→ ReLU ──→ MaxPool(2×2) ──→ Flatten ──→ FC(1152→10) ──→ Softmax
```

---

## Dimension Tracking Table (COMPLETED)

| Step | Operation | Input Shape | Output Shape | Params Added |
|------|-----------|-------------|--------------|--------------|
| 1 | Conv2D (5×5, 8 filters) | 28×28×1 | 24×24×8 | 5×5×1×8 + 8 = 208 |
| 2 | ReLU | 24×24×8 | 24×24×8 | 0 |
| 3 | MaxPool (2×2, stride 2) | 24×24×8 | 12×12×8 | 0 |
| 4 | Flatten | 12×12×8 | 1152 | 0 |
| 5 | FC (1152→10) | 1152 | 10 | 1152×10 + 10 = 11,530 |
| 6 | Softmax | 10 | 10 | 0 |
| | **TOTAL** | | | **11,738** |

---

## Your Turn: Design Your Own CNN

**Task:** Fill in the dimensions for these architecture variations:

### Variation A: Add a second conv layer
```
Input(28×28×1) → Conv1(3×3, 16 filters) → ReLU → Pool(2×2) → Conv2(3×3, 32 filters) → ReLU → Pool(2×2) → Flatten → FC(??→10)

| Step | Input Shape | Output Shape | Params |
|------|-------------|--------------|--------|
| Conv1 | 28×28×1 | ? | ? |
| Pool1 | ? | ? | 0 |
| Conv2 | ? | ? | ? |
| Pool2 | ? | ? | 0 |
| Flatten | ? | ? | 0 |
| FC | ? | 10 | ? |
| TOTAL | | | ? |
```

### Variation B: Same filters, bigger kernel
What happens if we use 7×7 filters instead of 5×5? Fill in:

| Change | Effect |
|--------|--------|
| Conv output size | 28 - 7 + 1 = ? × ? |
| Pool output size | ? ÷ 2 = ? × ? |
| FC input size (after flatten) | ? × ? × 8 = ? |
| Conv params (8 filters, 7×7) | 7×7×1×8 + 8 = ? |
| FC params (input→10) | ? × 10 + 10 = ? |
| Total params | ? |

### Variation C: More filters
What if we keep 5×5 filters but use 16 filters instead of 8?

| Change | Effect |
|--------|--------|
| Conv output channels | 16 (instead of 8) |
| Pool output size | 12×12×16 |
| FC input | 12×12×16 = ? |
| Conv params | 5×5×1×16 + 16 = ? |
| FC params | ?×10 + 10 = ? |
| Total params | ? |

---

## Key Insight

The CNN has **11,738 parameters** — **9.3× fewer** than the MLP's 109,386 — yet achieves **higher accuracy** (96.29% vs 95.55%). Why?

> **Answer:** Weight sharing. The 5×5 conv filter has only 25 weights, reused 576 times (once per output position). The MLP gives every pixel→neuron connection its own unique weight. The CNN forces spatial structure — the same pattern detector works everywhere in the image. This is a stronger prior = better generalization with fewer parameters.
