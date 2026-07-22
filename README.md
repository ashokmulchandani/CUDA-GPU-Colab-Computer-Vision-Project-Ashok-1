# CUDA GPU & Computer Vision Project

## Notebooks (Open in Colab)

| Phase | Notebook | Open |
|-------|----------|------|
| Phase 1-3.5 | CUDA Fundamentals + MLP + CNN | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashokmulchandani/CUDA-GPU-Colab-Computer-Vision-Project-Ashok-1/blob/main/1_Ashok_CUDA.ipynb) |
| Phase 4 | YOLO + TensorRT | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashokmulchandani/CUDA-GPU-Colab-Computer-Vision-Project-Ashok-1/blob/main/2_Ashok_CUDA_YOLO.ipynb) — [🎓 Interactive Module](phase4_yolo_cuda.html) |
| Phase 7 | 3D Perception (LiDAR, Fusion, SLAM) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashokmulchandani/CUDA-GPU-Colab-Computer-Vision-Project-Ashok-1/blob/main/3_Ashok_CUDA_3D_Perception.ipynb) |
| Phase 6 | Multi-Stream CUDA | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ashokmulchandani/CUDA-GPU-Colab-Computer-Vision-Project-Ashok-1/blob/main/6_Multistream.ipynb) |

## Local Scripts

| Script | What it does |
|--------|-------------|
| `smart_camera.py` | Real-time YOLO + tracking + zone alerts + Telegram notifications |
| `yolo_webcam.py` | Simple YOLO webcam detection |

## Plan

See [CUDA_PLAN.md](CUDA_PLAN.md) for the full 9-phase learning plan.

## Progress

- ✅ Phase 1: CUDA Fundamentals
- ✅ Phase 2: Matrix Multiplication (694 GFLOPS)
- ✅ Phase 3: MLP on MNIST (95.55%, 30x faster than PyTorch)
- ✅ Phase 3.5: CNN on MNIST (96.29%, 7.4x faster than PyTorch)
- ✅ Phase 4: YOLO (137 FPS, TensorRT, live webcam)
- ✅ Phase 6: Production CUDA (TensorRT, Triton, Multi-stream 7x speedup)
- ✅ Phase 7: 3D Perception (LiDAR, PointPillars, Sensor Fusion, SLAM)
- ✅ Phase 9: Smart Camera (tracking, zones, Telegram alerts)
- ⬜ Phase 5: Chicken Disease (VGG16 + DVC)
- ⬜ Phase 8: Transformer from Scratch

## Interactive Learning Modules

| Phase | Module | Slides |
|-------|--------|--------|
| **3** | [🎓 MLP on MNIST — ReLU & Softmax](phase3_mlp_mnist.html) | 15 slides — Architecture, ReLU, Softmax, Accuracy, Loss + p_correct visual, R²/MAE/RMSE, Precision/Recall/F1, pipeline, detective analogy |
| **3** | [🧮 LR → Epochs Calculator](phase3_lr_convergence_calc.html) | Interactive — adjust learning rate, see epochs needed, accuracy trajectory bars, reference table |
| **3.5** | [🎓 CNN on MNIST — Convolution Kernels](phase3_5_cnn_mnist.html) | 13 slides — CNN vs FC, 2D convolution, Conv2D CUDA kernel code, Max Pooling, architecture, training loop, benchmarks (96.29%, 7.4×) + 4 hands-on exercise slides with Colab instructions |
| **3.5** | [📝 Hands-on Exercises](Phase_3_5_CNN/) | 6 templates — convolution by hand, kernel trace, max pooling, architecture design, Colab training + experiments, benchmarks + reflection |
| **4** | [🎓 YOLO + CUDA Pipeline](phase4_yolo_cuda.html) | 13 slides (8 concepts + 5 exercises) — YOLO architecture, CUDA preprocess kernel (419×), TensorRT FP16, CUDA NMS, 137 FPS pipeline, Triton serving. HANDS-ON: Colab + YOLO, CUDA kernels, TensorRT export, NMS, Smart Camera |
| **6** | [🎓 Production CUDA (6A-6F)](phase6_production_cuda.html) | 10 slides — TensorRT, Triton, Nsight, Multi-Stream 7×, CUDA Graphs, Custom Kernels + pybind11, Docker/K8s |

| **LLM** | [🧠 Where Are the Neurons in Llama?](llama_neuron_comparison.html) | Side-by-side: MLP neuron math (128×784) vs Llama 7B (11,008×4,096) — same formula, different scale |

> Use ← → arrow keys to navigate. Dark/light theme toggle in top-right. Each slide has a quiz.
