"""
Example: How to send an image to Triton for inference
(This would work when Triton server is running)

In production:
  - Triton runs on AWS EC2 with GPU
  - This client runs anywhere (your app, phone, web server)
  - Communication via HTTP (like calling any web API)
"""

import numpy as np
import requests
import json

# Triton server URL (would be your EC2 IP in production)
TRITON_URL = "http://localhost:8000/v2/models/yolo_detector/infer"

def preprocess_image(image_path):
    """Same preprocessing as our CUDA kernel — resize, normalize, HWC→CHW"""
    import cv2
    img = cv2.imread(image_path)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0  # normalize
    img = img.transpose(2, 0, 1)  # HWC → CHW
    return img

def send_to_triton(image):
    """Send image to Triton and get detections back"""
    # Format request as JSON
    payload = {
        "inputs": [{
            "name": "images",
            "shape": [1, 3, 640, 640],
            "datatype": "FP32",
            "data": image.flatten().tolist()
        }],
        "outputs": [{
            "name": "output0"
        }]
    }
    
    # Send HTTP POST (like calling any web API!)
    response = requests.post(TRITON_URL, json=payload)
    result = response.json()
    return result

# Example usage:
# image = preprocess_image("test_image.jpg")
# detections = send_to_triton(image)
# print(f"Found {len(detections)} objects!")

print("""
=== TRITON CLIENT EXAMPLE ===

This is how ANY application talks to Triton:

  1. Preprocess image (resize, normalize)
  2. Send HTTP POST to Triton URL
  3. Get back detections as JSON

Same as calling any web API!
No GPU needed on the client side.
No PyTorch/CUDA needed on the client side.
Just HTTP requests.

In production:
  Mobile app → HTTP → Triton (GPU server) → HTTP → results back to app
  Web app → HTTP → Triton (GPU server) → HTTP → results back to browser
  IoT camera → HTTP → Triton (GPU server) → HTTP → alert system

That's it! Triton handles all the GPU complexity.
""")
