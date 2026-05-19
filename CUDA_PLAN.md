# FULL CUDA PROJECT PLAN OF ACTION

## Phase 1: CUDA Fundamentals (Vectors, Memory, Kernels)

| Step | Task |
|------|------|
| 1.1 | Install CUDA Toolkit + verify with `nvcc --version` |
| 1.2 | Understand GPU architecture — SMs, cores, warps (32 threads) |
| 1.3 | Hello World kernel — `__global__` function, `<<<1,1>>>` launch |
| 1.4 | Vector addition — 1D grid, `threadIdx.x + blockIdx.x * blockDim.x` |
| 1.5 | Thread/block/grid indexing — 1D, 2D, 3D configurations |
| 1.6 | Memory: `cudaMalloc`, `cudaMemcpy` (Host↔Device) |
| 1.7 | Pinned memory (`cudaMallocHost`) vs pageable — benchmark transfer speed |
| 1.8 | Unified memory (`cudaMallocManaged`) — simplify code |
| 1.9 | Error handling — `cudaGetLastError()`, `cudaDeviceSynchronize()` |
| 1.10 | Device query — print GPU specs (cores, memory, compute capability) |
| 1.11 | Timing with `cudaEvent_t` — measure kernel execution time |
| 1.12 | Python comparison — same vector add in NumPy vs CuPy vs raw CUDA |

## Phase 2: Matrix Multiplication (Naive → Tiled → Optimized)

| Step | Task |
|------|------|
| 2.1 | Naive matmul — each thread computes one element of C = A × B |
| 2.2 | Understand memory bottleneck — global memory latency |
| 2.3 | Shared memory (`__shared__`) — what it is, bank conflicts |
| 2.4 | Tiled matmul — load tiles into shared memory, sync with `__syncthreads()` |
| 2.5 | Benchmark naive vs tiled (256×256, 512×512, 1024×1024) |
| 2.6 | Memory coalescing — access patterns that maximize bandwidth |
| 2.7 | Occupancy — experiment with block sizes (128, 256, 512 threads) |
| 2.8 | Loop unrolling + `#pragma unroll` |
| 2.9 | Compare with cuBLAS `cublasSgemm` — see how close you get |
| 2.10 | Profile with `nvprof` or Nsight Compute — identify bottlenecks |

## Phase 3: MLP on MNIST (Fully Connected Layers in CUDA)

| Step | Task |
|------|------|
| 3.1 | Download MNIST dataset — parse binary format into arrays |
| 3.2 | Data loading — transfer images + labels to GPU |
| 3.3 | Forward pass kernel — matrix multiply (input × weights) + bias |
| 3.4 | Activation kernel — ReLU (`max(0, x)`) element-wise |
| 3.5 | Softmax kernel — compute probabilities for 10 classes |
| 3.6 | Cross-entropy loss kernel — compute loss value |
| 3.7 | Backpropagation kernel — compute gradients (dW, db) |
| 3.8 | Weight update kernel — SGD: `W -= lr * dW` |
| 3.9 | Training loop — forward → loss → backward → update (100 epochs) |
| 3.10 | Accuracy evaluation — predict on test set, compute % correct |
| 3.11 | Network architecture: 784 → 128 → 64 → 10 |
| 3.12 | Benchmark vs PyTorch MLP on same architecture |

## Phase 3.5: CNN on MNIST (Convolution Kernel in CUDA)

| Step | Task |
|------|------|
| 3.5.1 | Understand 2D convolution — filter sliding over image |
| 3.5.2 | Naive 2D convolution kernel — each thread computes one output pixel |
| 3.5.3 | Padding + stride handling in kernel |
| 3.5.4 | Max pooling kernel — 2×2 window, stride 2 |
| 3.5.5 | Tiled convolution — load input tile + halo into shared memory |
| 3.5.6 | Multi-channel convolution — handle RGB/multiple feature maps |
| 3.5.7 | Forward pass: Conv → ReLU → Pool → Conv → ReLU → Pool → FC → Softmax |
| 3.5.8 | Backprop through conv layer — gradient w.r.t. filters and input |
| 3.5.9 | Train CNN on MNIST — target >98% accuracy |
| 3.5.10 | Benchmark vs PyTorch CNN on same architecture |
| 3.5.11 | Profile conv kernel — compare with cuDNN performance |

## Phase 4: Object Detection (YOLO) — Real CNN in Action

| Step | Task |
|------|------|
| 4.1 | Understand YOLO architecture — grid cells, bounding boxes, class probs |
| 4.2 | Download pre-trained YOLOv5/v8 weights |
| 4.3 | Image preprocessing kernel — resize, normalize, HWC→CHW on GPU |
| 4.4 | CUDA inference pipeline — load weights → forward pass through layers |
| 4.5 | Non-Maximum Suppression (NMS) kernel — filter overlapping boxes on GPU |
| 4.6 | Post-processing — draw bounding boxes, class labels |
| 4.7 | Batch inference — process multiple images simultaneously |
| 4.8 | TensorRT optimization — convert model, compare latency |
| 4.9 | Real-time video inference — webcam → CUDA pipeline → display |
| 4.10 | Benchmark: PyTorch vs raw CUDA vs TensorRT (FPS comparison) |

## Phase 5: Chicken Disease Classification (VGG16 + DVC + Deployment)

| Step | Task |
|------|------|
| 5.1 | Load existing Chicken Disease dataset |
| 5.2 | VGG16 conv layers in CUDA — 3×3 filters, 64→128→256→512 channels |
| 5.3 | Batch normalization kernel |
| 5.4 | Implement full VGG16 forward pass in CUDA |
| 5.5 | Transfer learning — load pre-trained weights, fine-tune last layers |
| 5.6 | Training pipeline — CUDA kernels for forward + backward |
| 5.7 | DVC pipeline — `dvc.yaml` stages: data → train → evaluate |
| 5.8 | Model versioning with DVC — track weights in S3 |
| 5.9 | Export to ONNX → TensorRT for optimized inference |
| 5.10 | Flask/FastAPI endpoint — upload image → CUDA inference → return prediction |
| 5.11 | Docker container with CUDA runtime |
| 5.12 | Deploy to AWS EC2 (GPU instance) or Azure Container Instance |
| 5.13 | CI/CD — GitHub Actions: test → build → push → deploy |

## Phase 6: Production CUDA (Ship It — Real-World Deployment)

### 6A: TensorRT — Optimize Models for Production Inference

| Step | Task |
|------|------|
| 6A.1 | Understand TensorRT — what it does (graph optimization, layer fusion, kernel auto-tuning) |
| 6A.2 | Export PyTorch model to ONNX format |
| 6A.3 | Convert ONNX → TensorRT engine (`trtexec` CLI tool) |
| 6A.4 | FP32 vs FP16 vs INT8 precision — accuracy vs speed tradeoff |
| 6A.5 | Benchmark: PyTorch vs ONNX Runtime vs TensorRT (latency + throughput) |
| 6A.6 | Dynamic batching — handle variable batch sizes in production |
| 6A.7 | TensorRT plugins — write custom layer when TensorRT doesn't support an op |
| 6A.8 | Serialize engine to file — load pre-built engine for instant startup |

### 6B: NVIDIA Triton Inference Server — Production Model Serving

| Step | Task |
|------|------|
| 6B.1 | Understand Triton — model repository, backends, scheduling |
| 6B.2 | Set up Triton with Docker (`nvcr.io/nvidia/tritonserver`) |
| 6B.3 | Deploy TensorRT model to Triton — create `config.pbtxt` |
| 6B.4 | HTTP/gRPC client — send inference requests to Triton |
| 6B.5 | Dynamic batching in Triton — auto-batch multiple requests for throughput |
| 6B.6 | Model ensembles — chain preprocessing → model → postprocessing |
| 6B.7 | Multi-model serving — serve YOLO + VGG16 on same GPU |
| 6B.8 | Health checks + Prometheus metrics — monitor Triton in production |
| 6B.9 | A/B testing — serve two model versions, compare performance |

### 6C: Profiling & Optimization — Find and Fix Bottlenecks

| Step | Task |
|------|------|
| 6C.1 | Nsight Systems — timeline view of CPU↔GPU interaction |
| 6C.2 | Identify gaps — where is GPU idle? (data transfer? CPU bottleneck?) |
| 6C.3 | Nsight Compute — deep-dive into single kernel performance |
| 6C.4 | Memory throughput analysis — are you hitting bandwidth limits? |
| 6C.5 | Occupancy analysis — are SMs fully utilized? |
| 6C.6 | Warp stall reasons — what's blocking your threads? |
| 6C.7 | Roofline model — is your kernel compute-bound or memory-bound? |
| 6C.8 | Iterative optimization — profile → fix → profile again → measure improvement |

### 6D: Multi-Stream & Async Execution — Overlap Everything

| Step | Task |
|------|------|
| 6D.1 | Understand CUDA streams — independent execution queues |
| 6D.2 | Default stream vs custom streams — why default serializes everything |
| 6D.3 | Overlap data transfer + compute — copy batch N+1 while processing batch N |
| 6D.4 | Multi-stream pipeline — stream 1: upload, stream 2: compute, stream 3: download |
| 6D.5 | Events + synchronization — `cudaEventRecord`, `cudaStreamWaitEvent` |
| 6D.6 | Benchmark: single stream vs multi-stream (measure overlap benefit) |
| 6D.7 | CUDA Graphs — capture entire workflow, replay with minimal launch overhead |
| 6D.8 | Graph instantiation — reduce kernel launch cost from ~5μs to ~0.5μs |

### 6E: Custom CUDA Kernels in Production Pipelines

| Step | Task |
|------|------|
| 6E.1 | Image preprocessing kernel — resize + normalize + HWC→CHW in one kernel |
| 6E.2 | Video decode on GPU — NVDEC + custom frame processing |
| 6E.3 | Custom NMS kernel — faster than PyTorch's torchvision NMS |
| 6E.4 | Batched preprocessing — process 32 images simultaneously |
| 6E.5 | Integration with TensorRT — custom kernel as pre/post processing |
| 6E.6 | Python bindings (pybind11) — call your CUDA kernel from Python/FastAPI |
| 6E.7 | End-to-end pipeline: Video → GPU decode → preprocess → TensorRT → postprocess → output |

### 6F: Deployment & Infrastructure

| Step | Task |
|------|------|
| 6F.1 | Docker with CUDA — `nvidia/cuda:12.x-runtime` base image |
| 6F.2 | Multi-stage Docker build — compile CUDA in build stage, slim runtime image |
| 6F.3 | AWS ECS/EKS with GPU — deploy Triton on managed Kubernetes |
| 6F.4 | Auto-scaling — scale GPU instances based on request queue depth |
| 6F.5 | Cost optimization — spot instances, right-sizing GPU (T4 vs A10G vs A100) |
| 6F.6 | Monitoring — GPU utilization, memory usage, inference latency (CloudWatch/Grafana) |
| 6F.7 | Load testing — measure max throughput before latency degrades |
| 6F.8 | CI/CD for GPU models — build TensorRT engine in pipeline, deploy to Triton |

---

## Execution Order (Recommended)

| Session | What to Complete | Time |
|---------|-----------------|------|
| Session 1 | Phase 1: Steps 1.1–1.6 (install, hello world, vector add, memory) | 2-3 hrs |
| Session 2 | Phase 1: Steps 1.7–1.12 (pinned/unified memory, timing, Python comparison) | 2-3 hrs |
| Session 3 | Phase 2: Steps 2.1–2.5 (naive matmul, shared memory, tiled matmul) | 3-4 hrs |
| Session 4 | Phase 2: Steps 2.6–2.10 (coalescing, occupancy, profiling) | 2-3 hrs |
| Session 5 | Phase 3: Steps 3.1–3.6 (MNIST data, forward pass, softmax, loss) | 3-4 hrs |
| Session 6 | Phase 3: Steps 3.7–3.12 (backprop, training loop, benchmark) | 3-4 hrs |
| Session 7 | Phase 3.5: Steps 3.5.1–3.5.6 (2D conv kernel, pooling, multi-channel) | 3-4 hrs |
| Session 8 | Phase 3.5: Steps 3.5.7–3.5.11 (CNN training, backprop, profiling) | 3-4 hrs |
| Session 9 | Phase 4: Steps 4.1–4.5 (YOLO architecture, inference, NMS) | 3-4 hrs |
| Session 10 | Phase 4: Steps 4.6–4.10 (video inference, TensorRT, benchmarks) | 3-4 hrs |
| Session 11 | Phase 5: Steps 5.1–5.6 (VGG16 in CUDA, training) | 4-5 hrs |
| Session 12 | Phase 5: Steps 5.7–5.13 (DVC, deployment, CI/CD) | 3-4 hrs |
| Session 13 | Phase 6A: TensorRT optimization (steps 6A.1–6A.8) | 3-4 hrs |
| Session 14 | Phase 6B: Triton Inference Server (steps 6B.1–6B.9) | 3-4 hrs |
| Session 15 | Phase 6C: Profiling with Nsight (steps 6C.1–6C.8) | 3-4 hrs |
| Session 16 | Phase 6D: Multi-stream + CUDA Graphs (steps 6D.1–6D.8) | 3-4 hrs |
| Session 17 | Phase 6E: Custom production kernels (steps 6E.1–6E.7) | 3-4 hrs |
| Session 18 | Phase 6F: Deployment + infrastructure (steps 6F.1–6F.8) | 3-4 hrs |

---

## Tools & Technologies

| Category | Tools |
|----------|-------|
| GPU Programming | CUDA C/C++, nvcc compiler |
| Profiling | nvprof, Nsight Compute, Nsight Systems |
| Libraries | cuBLAS, cuDNN, TensorRT |
| Python GPU | CuPy, Numba CUDA, PyCUDA |
| Deep Learning | PyTorch (for benchmarking) |
| Data | MNIST, Chicken Disease dataset |
| MLOps | DVC, GitHub Actions |
| Deployment | Docker (CUDA runtime), FastAPI, AWS EC2 GPU |
| Models | MLP, CNN, VGG16, YOLOv5/v8 |
| Production Inference | TensorRT, ONNX Runtime, NVIDIA Triton |
| Streaming | CUDA Streams, CUDA Graphs, async execution |
| Video | NVDEC (GPU video decode) |
| Monitoring | Nsight Systems, Nsight Compute, Prometheus, Grafana |

---

## GPU Environment Setup

| Phase | Environment | GPU | Cost |
|-------|-------------|-----|------|
| Phase 1-3.5 | Google Colab (Free) | T4 16GB | Free |
| Phase 4 | Colab Pro | V100 16GB | ~$12/month |
| Phase 5 | AWS g5.xlarge or Colab Pro+ | A10G / A100 | ~$1/hr or ~$50/month |
| Phase 6 | AWS g5.xlarge or local GPU | A10G / T4 | ~$1/hr |

### Compile CUDA in Colab

```python
# Write CUDA file
%%writefile vector_add.cu
#include <stdio.h>
__global__ void add(int *a, int *b, int *c) { ... }

# Compile and run
!nvcc vector_add.cu -o vector_add && ./vector_add
```

### Cloud GPU Options (Phase 5 / Large Training)

| Provider | GPU | Cost |
|----------|-----|------|
| AWS p4d.24xlarge | 8x A100 (40GB) | ~$32/hr |
| Azure ND A100 v4 | 8x A100 | ~$27/hr |
| Lambda Cloud | A100 | ~$1.10/hr |
| RunPod | A100 / H100 | ~$1.64/hr |
| NVIDIA DGX Cloud | H100 pods | Contact sales |

---

## Prerequisites

| Requirement | How to Check |
|-------------|--------------|
| NVIDIA GPU | `nvidia-smi` |
| CUDA Toolkit | `nvcc --version` |
| cuDNN | Check `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\include\cudnn.h` |
| Visual Studio (Windows) | Needed for CUDA compilation on Windows |
| Python 3.8+ | `python --version` |
| PyTorch with CUDA | `python -c "import torch; print(torch.cuda.is_available())"` |

---

## Project Structure

```
CUDA_Projects-Ashok/
├── CUDA_PLAN.md                    # This file
├── CUDA_Transcripts/               # Reference transcripts
├── docs/                           # Notes, diagrams, architecture
├── notebooks/                      # Jupyter notebooks (Python GPU comparisons)
├── src/
│   ├── phase1/                     # Fundamentals
│   │   ├── hello_world.cu
│   │   ├── vector_add.cu
│   │   ├── memory_types.cu
│   │   ├── device_query.cu
│   │   └── timing.cu
│   ├── phase2/                     # Matrix multiplication
│   │   ├── matmul_naive.cu
│   │   ├── matmul_tiled.cu
│   │   └── matmul_cublas.cu
│   ├── phase3/                     # MLP on MNIST
│   │   ├── mnist_loader.cu
│   │   ├── forward.cu
│   │   ├── backward.cu
│   │   └── train.cu
│   ├── phase3_5/                   # CNN on MNIST
│   │   ├── conv2d.cu
│   │   ├── pooling.cu
│   │   └── cnn_train.cu
│   ├── phase4/                     # YOLO inference
│   │   ├── preprocess.cu
│   │   ├── inference.cu
│   │   └── nms.cu
│   ├── phase5/                     # Chicken Disease VGG16
│   │   ├── vgg16.cu
│   │   ├── train.cu
│   │   └── inference_api.py
│   └── phase6/                     # Production CUDA
│       ├── tensorrt/
│       │   ├── export_onnx.py
│       │   ├── build_engine.py
│       │   └── infer_trt.py
│       ├── triton/
│       │   ├── model_repository/
│       │   └── client.py
│       ├── profiling/
│       │   └── profile_notes.md
│       ├── streams/
│       │   ├── multi_stream.cu
│       │   └── cuda_graphs.cu
│       └── custom_kernels/
│           ├── preprocess.cu
│           ├── nms_custom.cu
│           └── python_bindings.cpp
├── data/                           # MNIST, chicken disease images
├── models/                         # Saved weights
└── Makefile                        # Build all CUDA files
```
