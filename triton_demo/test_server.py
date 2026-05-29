"""
Test Client — Sends an image to the inference server
Run this in a SEPARATE terminal while inference_server.py is running.

This simulates:
  - A mobile app sending a photo for detection
  - A security camera sending frames
  - Any client talking to Triton
"""

import requests
import json
import time

SERVER_URL = "http://localhost:8000"

# 1. Health check
print("1. Checking server health...")
try:
    r = requests.get(f"{SERVER_URL}/health", timeout=5)
    print(f"   Response: {r.json()}\n")
except:
    print("   ERROR: Server not running! Start inference_server.py first.\n")
    exit()

# 2. Send image for detection
print("2. Sending image for detection...")

# Use webcam to capture one frame (or use a file)
import cv2
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

if not ret:
    # If no webcam, create a dummy image
    import numpy as np
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Test Image", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

# Encode as JPEG
_, img_encoded = cv2.imencode('.jpg', frame)

# Send to server (same as sending to Triton!)
start = time.time()
r = requests.post(
    f"{SERVER_URL}/detect",
    files={"image": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
)
elapsed = (time.time() - start) * 1000

result = r.json()
print(f"   Response ({elapsed:.0f}ms total round-trip):")
print(f"   Detections: {result['count']}")
print(f"   Inference: {result['inference_ms']}ms")
for det in result['detections']:
    print(f"     • {det['class']} ({det['confidence']*100:.0f}%) at {det['box']}")

print(f"\n3. What just happened:")
print(f"   Client (this script) → HTTP POST → Server (inference_server.py) → YOLO → JSON back")
print(f"   Same pattern as: Mobile App → HTTP → Triton on AWS → detections back")
print(f"\n   The server could be on localhost, AWS, or anywhere!")
print(f"   Client doesn't need PyTorch, CUDA, or GPU — just HTTP!")
