"""
Mini Inference Server — Same concept as NVIDIA Triton
Runs on localhost:8000, accepts images, returns detections.

This demonstrates:
  - Model loaded ONCE at startup (not per-request)
  - HTTP API (any client can send requests)
  - JSON response (universal format)
  - Same pattern as Triton, just simpler

Run: python inference_server.py
Test: python test_server.py (in another terminal)
"""

from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
import time

app = Flask(__name__)

# Load model ONCE at startup (like Triton does)
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')
print("✓ Model loaded and ready to serve!\n")

@app.route('/health', methods=['GET'])
def health():
    """Health check — same as Triton's /v2/health/ready"""
    return jsonify({"status": "ready", "model": "yolov8n"})

@app.route('/camera', methods=['GET'])
def camera():
    """Web page that opens phone camera and sends to /detect"""
    return '''
    <html>
    <head><title>Smart Camera</title></head>
    <body style="text-align:center; font-family:Arial;">
        <h2>Phone Camera → AI Detection</h2>
        <video id="video" width="320" height="240" autoplay></video><br><br>
        <button onclick="capture()" style="padding:15px 30px; font-size:18px;">📷 Detect Objects</button>
        <canvas id="canvas" width="320" height="240" style="display:none;"></canvas>
        <h3 id="result"></h3>
        <div id="detections"></div>
        <script>
            navigator.mediaDevices.getUserMedia({video: {facingMode: "environment"}})
                .then(stream => document.getElementById("video").srcObject = stream);
            
            function capture() {
                var canvas = document.getElementById("canvas");
                var video = document.getElementById("video");
                canvas.getContext("2d").drawImage(video, 0, 0, 320, 240);
                canvas.toBlob(function(blob) {
                    var form = new FormData();
                    form.append("image", blob, "frame.jpg");
                    document.getElementById("result").innerText = "Detecting...";
                    fetch("/detect", {method: "POST", body: form})
                        .then(r => r.json())
                        .then(data => {
                            document.getElementById("result").innerText = 
                                "Found " + data.count + " objects (" + data.inference_ms + "ms)";
                            var html = "";
                            data.detections.forEach(d => {
                                html += "<p>" + d.class + " (" + (d.confidence*100).toFixed(0) + "%)</p>";
                            });
                            document.getElementById("detections").innerHTML = html;
                        });
                }, "image/jpeg");
            }
        </script>
    </body>
    </html>
    '''

@app.route('/detect', methods=['POST'])
def detect():
    """
    Accept image, run YOLO, return detections.
    Same concept as: POST http://triton:8000/v2/models/yolo/infer
    """
    start = time.time()
    
    # Read image from request
    file = request.files.get('image')
    if not file:
        return jsonify({"error": "No image provided"}), 400
    
    # Decode image
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
    
    # Run YOLO inference
    results = model(img, verbose=False)[0]
    
    # Format detections as JSON
    detections = []
    for box in results.boxes:
        detections.append({
            "class": model.names[int(box.cls)],
            "confidence": round(float(box.conf), 3),
            "box": [round(x, 1) for x in box.xyxy[0].tolist()]
        })
    
    elapsed = (time.time() - start) * 1000
    
    return jsonify({
        "detections": detections,
        "count": len(detections),
        "inference_ms": round(elapsed, 1)
    })

if __name__ == '__main__':
    print("=" * 50)
    print("MINI INFERENCE SERVER (like Triton)")
    print("=" * 50)
    print(f"\nEndpoints:")
    print(f"  GET  http://localhost:8000/health  — check if server is ready")
    print(f"  POST http://localhost:8000/detect  — send image, get detections")
    print(f"\nThis is the SAME pattern as Triton:")
    print(f"  Triton:    POST http://server:8000/v2/models/yolo/infer")
    print(f"  Our server: POST http://localhost:8000/detect")
    print(f"\nPress Ctrl+C to stop\n")
    
    app.run(host='0.0.0.0', port=8000)
