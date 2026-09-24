# Raspberry Pi YOLOv5 Currency Detection - Headless version (systemd safe)

import os
import sys
import time
from pathlib import Path
import torch
import cv2
import numpy as np
from ultralytics.utils.plotting import Annotator, colors
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from picamera2 import Picamera2
import subprocess
import signal

# 🧠 CONFIG
MODEL_PATH = "/home/project/projecteye/yolov5/indian_currency_yolov5/best_linux.pt"
CONF_THRESHOLD = 0.25
IMG_SIZE = (416, 416)
DEVICE = "cpu"

# 🗣 Speak
def speak(text):
    try:
        subprocess.run(["espeak", text])
    except:
        pass

# Handle stop from main.py
running = True
def exit_gracefully(sig, frame):
    global running
    running = False

signal.signal(signal.SIGTERM, exit_gracefully)
signal.signal(signal.SIGINT, exit_gracefully)

# 🧩 Load model
print("🧠 Loading currency model...")
device = select_device(DEVICE)
model = DetectMultiBackend(MODEL_PATH, device=device, dnn=False, data=None, fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(IMG_SIZE, s=stride)
model.warmup(imgsz=(1, 3, *imgsz))
print("💰 Currency model ready")

# 📸 Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)
picam2.start()
print("📷 Currency detection started")

last_spoken = ""
last_time = 0

# 🔁 Loop
while running:
    frame = picam2.capture_array()
    im0 = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    img = cv2.resize(im0, imgsz)
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img)
    pred = non_max_suppression(pred, CONF_THRESHOLD, 0.45, None, False, max_det=1000)

    for det in pred:
        if len(det):
            det[:, :4] = scale_boxes(img.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                label = names[int(cls)]
                confidence = float(conf)

                print(f"💰 Detected: {label} ({confidence:.2f})")

                speak_text = f"{label} rupees detected"
                now = time.time()

                # Speak every 3 seconds or if new denomination
                if speak_text != last_spoken or (now - last_time > 3):
                    speak(speak_text)
                    last_spoken = speak_text
                    last_time = now

    time.sleep(0.01)  # reduce CPU load

# 🛑 Cleanup
picam2.stop()
print("🛑 Currency detection stopped")
sys.exit()
