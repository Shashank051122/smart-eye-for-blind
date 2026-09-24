import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import os
import time

# ---------- SPEAK ----------
def speak(text):
    os.system(f'espeak "{text}" --stdout | aplay > /dev/null 2>&1')

# ---------- LOAD COCO NAMES ----------
with open("/home/project/projecteye/object_onnx/coco.names", "r") as f:
    class_names = [c.strip() for c in f.readlines()]

# ---------- LOAD ONNX MODEL ----------
MODEL_PATH = "/home/project/projecteye/object_onnx/yolov5n.onnx"

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape  # [1,3,640,640]
IMG_SIZE = input_shape[2]

print("🚀 YOLOv5n ONNX Detection Started")

# ---------- CAMERA ----------
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

def get_direction(cx, w):
    if cx < w * 0.33:
        return "left"
    elif cx > w * 0.66:
        return "right"
    else:
        return "front"

last_spoken = ""
last_time = 0

while True:
    frame = picam2.capture_array()
    h, w = frame.shape[:2]

    # ----- Preprocess -----
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR→RGB, HWC→CHW
    img = np.ascontiguousarray(img).astype(np.float16) / 255.0
    img = np.expand_dims(img, 0)

    # ----- ONNX INFER -----
    pred = session.run(None, {input_name: img})[0]  # (1,25200,85)
    pred = pred[0]

    for det in pred:
        x, y, bw, bh, obj, cls_scores = det[0], det[1], det[2], det[3], det[4], det[5:]
        cls_id = np.argmax(cls_scores)
        conf = cls_scores[cls_id]

        # Ignore low confidence
        if conf < 0.55:
            continue

        # ----- SAFE CHECK: avoid NaN / Inf -----
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(bw) or not np.isfinite(bh):
            continue

        # YOLO formats
        cx, cy = x * w, y * h
        bw, bh = bw * w, bh * h

        # Bounding box coords
        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)

        # ----- Clamp values -----
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        label = class_names[cls_id]
        direction = get_direction(cx, w)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # Voice output
        now = time.time()
        spoken_text = f"{label} in {direction}"

        if spoken_text != last_spoken or (now - last_time > 3):
            print(spoken_text)
            speak(spoken_text)
            last_spoken = spoken_text
            last_time = now

    cv2.imshow("YOLOv5n ONNX Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
print("👋 Detection stopped.")
