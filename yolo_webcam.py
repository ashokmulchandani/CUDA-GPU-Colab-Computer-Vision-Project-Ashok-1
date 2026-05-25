from ultralytics import YOLO
import cv2

# Use OpenVINO for Intel GPU (faster) or CPU as fallback
# First run: pip install ultralytics opencv-python openvino
model = YOLO('yolov8n.pt')

# Open webcam (try multiple indices)
cap = None
for idx in [0, 1, 2]:
    test_cap = cv2.VideoCapture(idx)
    ret, frame = test_cap.read()
    if ret and frame is not None:
        cap = test_cap
        print(f"Found webcam at index {idx}")
        break
    test_cap.release()

if cap is None:
    print("ERROR: No webcam found. Make sure:")
    print("  1. Webcam is connected")
    print("  2. No other app is using it (close Teams/Zoom)")
    print("  3. Try running from terminal, not IDE")
    exit()

print("YOLO Webcam Detection Running!")
print("Press 'q' to quit")
print("-" * 40)

frame_count = 0
import time
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Run YOLO detection
    results = model(frame, verbose=False)
    
    # Draw boxes on frame
    annotated = results[0].plot()
    
    # Show FPS
    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    cv2.putText(annotated, f'FPS: {fps:.1f}', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display
    cv2.imshow('YOLO Real-time Detection', annotated)
    
    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\nProcessed {frame_count} frames at {fps:.1f} FPS average")
