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

## Phase 7: Computer Vision & 3D Perception (LIDAR, Point Clouds, Sensor Fusion)

> Target role: Computer Vision / ML Engineer — end-to-end from raw sensor data to deployed solutions.
> Skills: LIDAR, cameras, multi-sensor inputs, point clouds, image processing, sensor fusion, spatial understanding.

### 7A: Point Cloud Processing on GPU

| Step | Task |
|------|------|
| 7A.1 | Understand point clouds — XYZ + intensity, unstructured 3D data |
| 7A.2 | Load LIDAR data (`.pcd`, `.ply`, `.las` formats) into GPU memory |
| 7A.3 | CUDA kernel: voxel grid downsampling — reduce millions of points to manageable size |
| 7A.4 | CUDA kernel: radius/KNN search — find nearest neighbors in 3D space |
| 7A.5 | CUDA kernel: ground plane segmentation — RANSAC on GPU |
| 7A.6 | CUDA kernel: clustering (DBSCAN on GPU) — group points into objects |
| 7A.7 | CUDA kernel: bounding box fitting — oriented 3D boxes around clusters |
| 7A.8 | Benchmark vs Open3D / PCL (CPU) — measure GPU speedup |

### 7B: 3D Object Detection (PointNet / PointPillars)

| Step | Task |
|------|------|
| 7B.1 | Understand PointNet architecture — per-point MLP + global max pooling |
| 7B.2 | Implement PointNet forward pass in CUDA — shared MLP kernels |
| 7B.3 | Understand PointPillars — convert point cloud to 2D pseudo-image |
| 7B.4 | CUDA kernel: pillar creation — scatter points into vertical columns |
| 7B.5 | CUDA kernel: pillar feature encoding — per-pillar PointNet |
| 7B.6 | 2D backbone (SSD-style) on encoded pillars — detect objects in BEV |
| 7B.7 | 3D NMS kernel — filter overlapping 3D bounding boxes |
| 7B.8 | TensorRT deployment of PointPillars — real-time 3D detection |

### 7C: Sensor Fusion (Camera + LIDAR)

| Step | Task |
|------|------|
| 7C.1 | Camera calibration — intrinsic/extrinsic matrices, distortion correction |
| 7C.2 | LIDAR-to-camera projection kernel — project 3D points onto 2D image plane |
| 7C.3 | Camera-to-LIDAR back-projection — lift 2D detections into 3D space |
| 7C.4 | Early fusion — concatenate image features + point cloud features on GPU |
| 7C.5 | Late fusion — run 2D detector (YOLO) + 3D detector (PointPillars), merge results |
| 7C.6 | Temporal fusion — track objects across frames using Kalman filter on GPU |
| 7C.7 | BEV (Bird's Eye View) fusion — project both sensors to common BEV grid |
| 7C.8 | End-to-end pipeline: LIDAR + Camera → fused detections → tracked objects |

### 7D: Depth Estimation & Stereo Vision

| Step | Task |
|------|------|
| 7D.1 | Stereo matching kernel — compute disparity map from left/right cameras |
| 7D.2 | Semi-Global Matching (SGM) on GPU — cost aggregation along multiple paths |
| 7D.3 | Monocular depth estimation — deploy MiDaS/DPT with TensorRT |
| 7D.4 | Depth to point cloud kernel — convert depth map to 3D points |
| 7D.5 | Occupancy grid kernel — build 2D/3D occupancy from depth data |
| 7D.6 | Benchmark: stereo vs LIDAR vs monocular depth accuracy |

### 7E: Image Processing Kernels for Perception

| Step | Task |
|------|------|
| 7E.1 | CUDA kernel: image undistortion — correct lens distortion in real-time |
| 7E.2 | CUDA kernel: color space conversion — RGB↔YUV↔HSV on GPU |
| 7E.3 | CUDA kernel: image warping — perspective transform, bird's eye view |
| 7E.4 | CUDA kernel: feature extraction — FAST/ORB keypoints on GPU |
| 7E.5 | CUDA kernel: optical flow — Lucas-Kanade or Farneback on GPU |
| 7E.6 | CUDA kernel: image stitching — multi-camera panorama in real-time |
| 7E.7 | Integration with GStreamer/DeepStream — GPU-accelerated video pipeline |

### 7F: Spatial Understanding & Mapping

| Step | Task |
|------|------|
| 7F.1 | Visual SLAM basics — feature matching + pose estimation on GPU |
| 7F.2 | ICP (Iterative Closest Point) kernel — align point clouds across frames |
| 7F.3 | Voxel-based mapping — build 3D map from accumulated LIDAR scans |
| 7F.4 | Semantic segmentation on point clouds — per-point classification |
| 7F.5 | Free-space detection — identify drivable/walkable areas |
| 7F.6 | Deploy full perception stack: sensors → preprocessing → detection → tracking → mapping |

## Phase 8: Transformer from Scratch in CUDA (Attention Is All You Need)

> The architecture behind GPT, Claude, Gemini, DALL-E, Whisper — the dominant AI architecture.

### 8A: Self-Attention Mechanism

| Step | Task |
|------|------|
| 8A.1 | Understand attention — "which parts of the input matter most for each output?" |
| 8A.2 | Query, Key, Value matrices — Q×K^T gives attention scores, multiply by V |
| 8A.3 | CUDA kernel: scaled dot-product attention — `softmax(Q×K^T / √d) × V` |
| 8A.4 | CUDA kernel: multi-head attention — split into 8 heads, attend in parallel |
| 8A.5 | Causal masking kernel — prevent looking at future tokens (for GPT-style) |
| 8A.6 | Benchmark attention kernel vs PyTorch `F.scaled_dot_product_attention` |

### 8B: Transformer Building Blocks

| Step | Task |
|------|------|
| 8B.1 | CUDA kernel: Layer Normalization — normalize across features (not batch) |
| 8B.2 | CUDA kernel: GELU activation — smooth ReLU used in transformers |
| 8B.3 | CUDA kernel: Feed-Forward Network (FFN) — two linear layers with GELU |
| 8B.4 | Residual connections — add input back to output (skip connections) |
| 8B.5 | Positional encoding — add position information (sinusoidal or learned) |
| 8B.6 | Token embedding kernel — convert token IDs to vectors |

### 8C: Full Transformer Encoder (BERT-style)

| Step | Task |
|------|------|
| 8C.1 | Stack: Embedding → [Attention → LayerNorm → FFN → LayerNorm] × N layers |
| 8C.2 | Implement 4-layer transformer encoder in CUDA |
| 8C.3 | Train on text classification task (sentiment analysis) |
| 8C.4 | Compare with HuggingFace BERT — accuracy and speed |

### 8D: Full Transformer Decoder (GPT-style)

| Step | Task |
|------|------|
| 8D.1 | Causal (autoregressive) attention — each token only sees previous tokens |
| 8D.2 | Implement GPT-style decoder (4 layers, 4 heads, 128 dim) |
| 8D.3 | Train on character-level text generation (Shakespeare / tiny dataset) |
| 8D.4 | Generate text token by token — sampling with temperature |
| 8D.5 | KV-Cache kernel — cache Key/Value for faster inference |
| 8D.6 | Benchmark: tokens/second generation speed |

### 8E: Optimization & Production

| Step | Task |
|------|------|
| 8E.1 | Flash Attention concept — fuse attention into one kernel, reduce memory |
| 8E.2 | Implement memory-efficient attention (tiled, no full N×N matrix) |
| 8E.3 | FP16 mixed precision — use half-precision for 2× speedup |
| 8E.4 | Quantization (INT8/INT4) — reduce model size for deployment |
| 8E.5 | TensorRT for transformer inference — optimize and deploy |
| 8E.6 | Benchmark: our CUDA vs PyTorch vs TensorRT (tokens/second) |

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
| Session 19 | Phase 7A: Point cloud processing on GPU (steps 7A.1–7A.8) | 4-5 hrs |
| Session 20 | Phase 7B: 3D object detection - PointPillars (steps 7B.1–7B.8) | 4-5 hrs |
| Session 21 | Phase 7C: Sensor fusion - Camera + LIDAR (steps 7C.1–7C.8) | 4-5 hrs |
| Session 22 | Phase 7D: Depth estimation + stereo (steps 7D.1–7D.6) | 3-4 hrs |
| Session 23 | Phase 7E: Image processing kernels (steps 7E.1–7E.7) | 3-4 hrs |
| Session 24 | Phase 7F: Spatial understanding + mapping (steps 7F.1–7F.6) | 4-5 hrs |
| Session 25 | Phase 8A: Self-attention mechanism (steps 8A.1–8A.6) | 3-4 hrs |
| Session 26 | Phase 8B: Transformer building blocks (steps 8B.1–8B.6) | 3-4 hrs |
| Session 27 | Phase 8C: Transformer encoder - BERT-style (steps 8C.1–8C.4) | 4-5 hrs |
| Session 28 | Phase 8D: Transformer decoder - GPT-style (steps 8D.1–8D.6) | 4-5 hrs |
| Session 29 | Phase 8E: Flash Attention + optimization (steps 8E.1–8E.6) | 3-4 hrs |

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
| Models | MLP, CNN, VGG16, YOLOv5/v8, PointNet, PointPillars, Transformer/GPT |
| Production Inference | TensorRT, ONNX Runtime, NVIDIA Triton |
| 3D / Perception | Open3D, PCL, LIDAR processing, stereo vision |
| Sensor Fusion | Camera calibration, LIDAR-camera projection, BEV fusion |
| Video Pipeline | GStreamer, NVIDIA DeepStream, NVDEC |
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
| Phase 7 | AWS g5.xlarge or Colab Pro+ | A10G / A100 | ~$1/hr or ~$50/month |

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
