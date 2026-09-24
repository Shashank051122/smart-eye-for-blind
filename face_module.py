from picamera2 import Picamera2
import face_recognition
import cv2
import numpy as np
import os
import time
import signal
import sys

# -------------------- SPEAK FUNCTION --------------------
def speak(text):
    print(f"🗣️ {text}")
    os.system(f'espeak "{text}" --stdout | aplay > /dev/null 2>&1')

# -------------------- LOAD KNOWN FACES --------------------
print("🔍 Loading known faces...")
known_face_encodings = []
known_face_names = []

base_dir = "/home/project/projecteye/known_faces"
if not os.path.exists(base_dir):
    print("⚠️ 'known_faces' folder not found!")
    sys.exit()

for person_name in os.listdir(base_dir):
    person_folder = os.path.join(base_dir, person_name)
    if not os.path.isdir(person_folder):
        continue
    for filename in os.listdir(person_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(person_folder, filename)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                known_face_encodings.append(encodings[0])
                known_face_names.append(person_name)

if not known_face_names:
    print("⚠️ No faces found in dataset!")
    sys.exit()

print(f"✅ Loaded {len(known_face_names)} known faces: {set(known_face_names)}")

# -------------------- INITIALIZE CAMERA --------------------
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

print("📷 Face recognition started...")
time.sleep(2)

# -------------------- MAIN LOOP --------------------
last_spoken = ""
last_time = time.time()

running = True

def handle_exit(sig, frame):
    global running
    running = False

signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

while running:
    frame = picam2.capture_array()

    small_frame = cv2.resize(frame, (0, 0), fx=0.20, fy=0.20)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        if True in matches:
            match_index = np.argmin(face_recognition.face_distance(known_face_encodings, face_encoding))
            name = known_face_names[match_index]

        # Speak name (not more than once every 3 seconds)
        if name != last_spoken and time.time() - last_time > 3:
            text = f"{name} detected" if name != "Unknown" else "Unknown person detected"
            speak(text)
            print("Detected:", text)
            last_spoken = name
            last_time = time.time()

    time.sleep(0.01)   # reduces CPU load

picam2.stop()
print("👋 Face recognition stopped.")
sys.exit()
