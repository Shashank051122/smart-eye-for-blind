import cv2
import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time
import os

# ---------- SPEAK FUNCTION ----------
def speak(text):
    os.system(f'espeak "{text}" --stdout | aplay > /dev/null 2>&1')


# ---------- MODEL PATH ----------
MODEL_PATH = "/home/project/projecteye/object_onnx/yolov5n.onnx"

print("🧠 Loading YOLOv5n ONNX model...")
session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ---------- CLASS NAMES ----------
class_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter",
    "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

# ---------- CAMERA SETUP ----------
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

print("🚀 YOLOv5n ONNX Detection Started")


# ---------- DIRECTION HELP FUNCTION ----------
def get_direction(x_center, frame_w):
    if x_center < frame_w * 0.33:
        return "left"
    elif x_center > frame_w * 0.66:
        return "right"
    else:
        return "center"


last_spoken = ""
last_time = 0

# ---------- MAIN LOOP ----------
while True:
    frame = picam2.capture_array()
    h, w, _ = frame.shape

    # Preprocess (YOLOv5 expects 640x640)
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1]             # BGR → RGB
    img = img.astype(np.float16) / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    img = np.expand_dims(img, axis=0)

    # Inference
    outputs = session.run(None, {input_name: img})[0]  # (1,25200,7)
    detections = outputs[0]

    for det in detections:
        x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det

        conf = float(obj_conf)
        if conf < 0.45:
            continue

        cls_id = int(cls_id)
        if cls_id >= len(class_names):
            continue

        label = class_names[cls_id]

        # Convert box back to camera resolution
        x1 = int(x1 / 640 * w)
        y1 = int(y1 / 640 * h)
        x2 = int(x2 / 640 * w)
        y2 = int(y2 / 640 * h)

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        # ---------- DIRECTION ----------
        x_center = (x1 + x2) // 2
        direction = get_direction(x_center, w)

        # Speak with timeout + avoid repeating
        now = time.time()
        sentence = f"{label} on your {direction}"

        if sentence != last_spoken or (now - last_time > 3):
            print(sentence)
            speak(sentence)
            last_spoken = sentence
            last_time = now

    # Show frame
    cv2.imshow("YOLOv5n ONNX Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
