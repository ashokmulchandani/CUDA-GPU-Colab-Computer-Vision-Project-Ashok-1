# Triton Model Repository — Practice

This is a practice Triton model repository. It follows the exact structure Triton expects.

## Folder Structure

```
triton_practice/
└── models/                    ← Point Triton here: --model-repository=/models
    └── yolo/                  ← Model name (used in API: /v2/models/yolo/infer)
        ├── config.pbtxt       ← Platform, I/O shapes, batching config
        ├── 1/                 ← Version 1 (initial deployment)
        │   └── model.plan     ← TensorRT engine (place your .engine file here)
        └── 2/                 ← Version 2 (for A/B testing)
            └── model.plan     ← Updated engine
```

## How to Use

### 1. Build the TensorRT engine (from Phase 6A)
```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='engine', half=True)  # creates yolov8n.engine
```

### 2. Copy the engine to the model repository
```bash
cp yolov8n.engine triton_practice/models/yolo/1/model.plan
```

### 3. Start Triton (requires Docker + NVIDIA GPU)
```bash
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/triton_practice/models:/models \
  nvcr.io/nvidia/tritonserver:24.01-py3 \
  tritonserver --model-repository=/models
```

### 4. Test
```bash
# Health check
curl http://localhost:8000/v2/health/ready

# List models
curl http://localhost:8000/v2/models
```

### A/B Testing Practice
1. Create an improved engine → copy to `models/yolo/2/model.plan`
2. Triton loads BOTH v1 and v2 automatically
3. Route 10% traffic to v2, compare latencies
4. If v2 is better → delete folder `1/` → instant promotion
5. If v2 fails → delete folder `2/` → instant rollback

## Where to Run

| What | Where |
|------|-------|
| Build TRT engine (6A) | **Colab** (free T4 GPU) |
| Triton server (6B) | **Cloud GPU VM** (AWS/Lambda/RunPod ~$0.50/hr) |
| Python client code | **Anywhere** — Colab, laptop, same VM |
