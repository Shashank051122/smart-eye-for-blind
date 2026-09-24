import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import os
import time

# ---------- SPEAK ----------
def speak(text):
    os.system(f'espeak "{text}" --stdout | aplay > /dev/null 2>&1')

# ---------- LOAD COCO LABELS ----------
with open("/home/project/projecteye/object_onnx/coco.names") as f:
    class_names = [c.strip() for c in f.readlines()]

# ---------- LOAD MODEL ----------
MODEL_PATH = "/home/project/projecteye/object_onnx/yolov5n.onnx"
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
IMG_SIZE = 640  # YOLOv5n ONNX expects 640x640

print("🚀 YOLOv5n ONNX (GRID DECODE) Started")

# ---------- CAMERA ----------
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# ---------- YOLOv5 STRIDES ----------
STRIDES = [8, 16, 32]
ANCHORS = np.array([
    [10,13, 16,30, 33,23],
    [30,61, 62,45, 59,119],
    [116,90, 156,198, 373,326]
]).reshape(3, 3, 2)


def decode_output(outputs):

    # outputs shape: (1, 25200, 85)
    pred = outputs[0]

    # Each row: [cx, cy, w, h, obj, cls1..cls80]
    boxes = []
    for det in pred:
        obj_conf = det[4]
        cls_conf = np.max(det[5:])
        conf = obj_conf * cls_conf
        if conf < 0.5:
            continue

        cls_id = int(np.argmax(det[5:]))

        # Apply sigmoid, convert to pixel coords:
        cx = det[0]
        cy = det[1]
        bw = det[2]
        bh = det[3]

        # Skip invalid boxes
        if not np.isfinite(cx) or not np.isfinite(bw):
            continue

        boxes.append([cx, cy, bw, bh, conf, cls_id])

    return boxes


def get_direction(cx, w):
    if cx < w * 0.33:
        return "left"
    elif cx > w * 0.66:
        return "right"
    else:
        return "front"


last_speak = ""
last_time = 0

while True:

    frame = picam2.capture_array()
    img0 = frame.copy()
    h, w = frame.shape[:2]

    # Preprocess to 640x640
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1]  # BGR->RGB
    img = img.transpose(2, 0, 1)
    img = np.ascontiguousarray(img).astype(np.float16) / 255.0
    img = img[None]

    # Inference
    raw = session.run(None, {input_name: img})
    preds = decode_output(raw)

    for (cx, cy, bw, bh, conf, cls_id) in preds:

        # Convert from 640 space → camera resolution
        cx *= w
        cy *= h
        bw *= w
        bh *= h

        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)

        # Clamp
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        label = class_names[cls_id]
        direction = get_direction(cx, w)

        # Draw detection
        cv2.rectangle(img0, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(img0, f"{label} {conf:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255,255,255), 2)

        speak_text = f"{label} in {direction}"

        now = time.time()
        if speak_text != last_speak or now - last_time > 3:
            print(speak_text)
            speak(speak_text)
            last_speak = speak_text
            last_time = now

    cv2.imshow("YOLOv5n ONNX Detection", img0)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
