"""
Phase 9: Smart Camera — YOLO + Tracking + Counting + Zones
Compatible with supervision 0.28+
Run: python smart_camera.py
Press 'q' to quit.
"""

from ultralytics import YOLO
import supervision as sv
import cv2
import numpy as np
import time
import requests

# === TELEGRAM CONFIG ===
TELEGRAM_TOKEN = "8500439596:AAEXFmzHLymijizbTeS_Dt5KCkxqbUavObU"
TELEGRAM_CHAT_ID = "1414560298"

def send_telegram_alert(frame, message):
    """Send photo + message to Telegram"""
    try:
        _, img_bytes = cv2.imencode('.jpg', frame)
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
        files = {'photo': ('alert.jpg', img_bytes.tobytes(), 'image/jpeg')}
        data = {'chat_id': TELEGRAM_CHAT_ID, 'caption': message}
        requests.post(url, files=files, data=data, timeout=5)
        print(f"  Telegram alert sent!")
    except Exception as e:
        print(f"  Telegram failed: {e}")

# === SETUP ===
print("Loading YOLO model...")
model = YOLO('yolov8n.pt')

# Tracker
tracker = sv.ByteTrack()

# Annotators
box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=0.5)
trace_annotator = sv.TraceAnnotator(thickness=2, trace_length=50)

# === COUNTING LINE (middle of frame) ===
LINE_START = sv.Point(0, 250)
LINE_END = sv.Point(640, 250)
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)

# === ALERT ZONE (right half of frame) ===
ZONE_POLYGON = np.array([[320, 50], [630, 50], [630, 430], [320, 430]])
zone = sv.PolygonZone(polygon=ZONE_POLYGON)

# === OPEN WEBCAM ===
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
    print("ERROR: No webcam found!")
    exit()

print("\n=== SMART CAMERA RUNNING ===")
print("  Move LEFT side = safe")
print("  Move RIGHT side = ZONE ALERT!")
print("  Cross middle line = counted")
print("  Press 'q' to quit")
print("-" * 40)

frame_count = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. DETECT (persons only)
    results = model(frame, verbose=False, classes=[0])[0]
    detections = sv.Detections.from_ultralytics(results)

    # 2. TRACK
    detections = tracker.update_with_detections(detections)

    # 3. LINE COUNTING
    line_zone.trigger(detections)

    # 4. ZONE CHECK
    in_zone = zone.trigger(detections)
    people_in_zone = int(np.sum(in_zone)) if in_zone is not None and len(in_zone) > 0 else 0
    
    # Debug: print when person detected
    if len(detections) > 0 and detections.xyxy is not None:
        for i, box in enumerate(detections.xyxy):
            center_x = int((box[0] + box[2]) / 2)
            if center_x > 320:
                people_in_zone = max(people_in_zone, 1)  # force alert if person center is in right half

    # 5. DRAW — trails
    frame = trace_annotator.annotate(scene=frame, detections=detections)

    # 6. DRAW — boxes + labels
    labels = []
    if detections.tracker_id is not None:
        for i, tid in enumerate(detections.tracker_id):
            zone_status = " [IN ZONE!]" if (in_zone is not None and i < len(in_zone) and in_zone[i]) else ""
            labels.append(f"Person #{tid}{zone_status}")
    frame = box_annotator.annotate(scene=frame, detections=detections)
    frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

    # 7. DRAW — counting line (manual since API varies)
    cv2.line(frame, (0, 250), (640, 250), (0, 255, 255), 2)  # yellow line
    cv2.putText(frame, "-- COUNTING LINE --", (220, 245),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 8. DRAW — zone rectangle (manual)
    cv2.polylines(frame, [ZONE_POLYGON], True, (0, 0, 255), 3)  # red border
    cv2.putText(frame, "ALERT ZONE", (350, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # 9. ZONE ALERT
    if people_in_zone > 0:
        cv2.putText(frame, "!! ZONE ALERT !!", (350, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
        # Flash red border
        cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 5)
        # Send Telegram alert (with cooldown so we don't spam)
        if frame_count % 150 == 0 or frame_count == 1:  # send every ~5 seconds max
            send_telegram_alert(frame, f"🚨 ZONE ALERT!\nPeople in zone: {people_in_zone}\nTime: {time.strftime('%H:%M:%S')}")

    # 10. Stats overlay
    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed if elapsed > 0 else 0

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Crossed In: {line_zone.in_count} | Out: {line_zone.out_count}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"People in Zone: {people_in_zone}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255) if people_in_zone > 0 else (200, 200, 200), 2)

    # Show
    cv2.imshow('Smart Camera - Phase 9', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n=== SESSION SUMMARY ===")
print(f"Duration: {elapsed:.1f}s | Frames: {frame_count} | FPS: {fps:.1f}")
print(f"Crossed IN: {line_zone.in_count} | OUT: {line_zone.out_count}")
