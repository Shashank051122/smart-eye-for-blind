import cv2
import numpy as np
import os
import time
from picamera2 import Picamera2
from ultralytics import YOLO

# --- Speak function ---
def speak(text):
    os.system(f'espeak "{text}" --stdout | aplay > /dev/null 2>&1')

# --- Load YOLO model ---
model = YOLO("yolo11s.pt")

# --- Initialize Picamera2 ---
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

print("🚀 Object Detection Started...")

last_spoken = ""
last_time = 0

while True:
    frame = picam2.capture_array()
    h, w = frame.shape[:2]

    results = model(frame, imgsz=480, conf=0.45)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # --- Calculate LEFT / FRONT / RIGHT ---
            center_x = (x1 + x2) / 2

            if center_x < w * 0.33:
                direction = "left"
            elif center_x > w * 0.66:
                direction = "right"
            else:
                direction = "front"

            speak_text = f"{label} on {direction}"

            # Speak intelligently (avoid repeating too much)
            now = time.time()
            if speak_text != last_spoken or (now - last_time > 9):
                print("🔊", speak_text)
                speak(speak_text)
                last_spoken = speak_text
                last_time = now

    time.sleep(0.01)  # prevent CPU overload

picam2.stop()
print("👋 Detection Stopped")
