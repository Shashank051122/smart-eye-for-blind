from ultralytics import YOLO
import cv2
import os

model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture("tcp://127.0.0.1:8888?overrun_nonfatal=1&fifo_size=50000000")

last_spoken = ""

def speak(text):
    os.system(f'espeak-ng "{text}" --stdout | aplay > /dev/null 2>&1')

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    results = model(frame)
    names = results[0].names
    detected = []

    for box in results[0].boxes:
        cls = int(box.cls)
        detected.append(names[cls])

    if detected:
        current = detected[0]
        if current != last_spoken:
            speak(current)
            last_spoken = current

    cv2.imshow("Smart Eye", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
