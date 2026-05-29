"""Export YOLOv8 to ONNX format for Triton Inference Server"""
from ultralytics import YOLO

model = YOLO('yolov8n.pt')  # uses local file
model.export(format='onnx', imgsz=640, simplify=True)

# Move to Triton model repository
import shutil
shutil.copy('yolov8n.onnx', 'model_repository/yolo_detector/1/model.onnx')
print("✓ Exported and placed in model_repository/yolo_detector/1/model.onnx")
