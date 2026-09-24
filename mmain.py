import RPi.GPIO as GPIO
import time, threading, os, subprocess, datetime
from picamera2 import Picamera2

# 🔵 NEW IMPORTS FOR FACE ADD & VOICE
import cv2
import speech_recognition as sr

# ---------------- GPIO SETUP ----------------
GPIO.setmode(GPIO.BCM)

KEY1, KEY2, KEY3, KEY4 = 5, 6, 13, 19
TIME_CAPTURE_BUTTON = 26

for pin in [KEY1, KEY2, KEY3, KEY4, TIME_CAPTURE_BUTTON]:
    GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)

TRIG, ECHO, BUZZ = 23, 24, 17
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.setup(BUZZ, GPIO.OUT)
GPIO.output(TRIG, False)

tts_trigger = "/home/project/projecteye/tts_trigger.txt"

# ---------------- STATE FLAGS ----------------
face_process = object_process = tts_process = currency_process = None
face_running = object_running = tts_running = currency_running = False

# ---------------- SPEAK ----------------
def speak(text):
    os.system(f"espeak '{text}' --stdout | aplay > /dev/null 2>&1")
    time.sleep(0.3)

# ---------------- ULTRASONIC ----------------
def ultrasonic_loop():
    while True:
        GPIO.output(TRIG, True)
        time.sleep(0.00001)
        GPIO.output(TRIG, False)
        start = time.time()
        while GPIO.input(ECHO) == 0:
            start = time.time()
        while GPIO.input(ECHO) == 1:
            end = time.time()
        distance = (end - start) * 17150
        if 2 < distance < 80:
            GPIO.output(BUZZ, True)
        else:
            GPIO.output(BUZZ, False)
        time.sleep(0.15)

threading.Thread(target=ultrasonic_loop, daemon=True).start()

# ---------------- MODULE HANDLERS ----------------
def stop_all():
    if face_running: stop_face()
    if object_running: stop_object()
    if tts_running: stop_tts()
    if currency_running: stop_currency()
    time.sleep(0.3)

# --- Face Recognition ---
def start_face():
    global face_running, face_process
    stop_all()
    speak("Face recognition on")
    face_process = subprocess.Popen(["python3", "/home/project/projecteye/face_module.py"])
    face_running = True

def stop_face():
    global face_running, face_process
    if face_process:
        face_process.terminate()
        face_process.wait()
    face_running = False
    speak("Face recognition off")

# --- Object Detection ---
def start_object():
    global object_running, object_process
    stop_all()
    speak("Object detection on")
    object_process = subprocess.Popen(["python3", "/home/project/projecteye/object_detection.py"])
    object_running = True

def stop_object():
    global object_running, object_process
    if object_process:
        object_process.terminate()
        object_process.wait()
    object_running = False
    speak("Object detection off")

# --- Text to Speech (OCR) ---
def start_tts():
    global tts_running, tts_process
    stop_all()
    open(tts_trigger, "w").close()
    speak("Text reading mode on")
    tts_process = subprocess.Popen(["python3", "/home/project/projecteye/text_to_speech.py"])
    tts_running = True

def stop_tts():
    global tts_running, tts_process
    if tts_process:
        with open(tts_trigger, "w") as f:
            f.write("stop")
        time.sleep(0.5)
        tts_process.terminate()
        tts_process.wait()
    tts_running = False
    speak("Text reading mode off")

# --- Currency Detection (YOLOv5) ---
def start_currency():
    global currency_running, currency_process
    stop_all()
    speak("Currency detection on")
    currency_process = subprocess.Popen(["python3", "/home/project/projecteye/yolov5/detect_pi.py"])
    currency_running = True

def stop_currency():
    global currency_running, currency_process
    if currency_process:
        currency_process.terminate()
        currency_process.wait()
    currency_running = False
    speak("Currency detection off")

# ---------------- UTILITIES ----------------
def detect_press(pin):
    start = time.time()
    while GPIO.input(pin) == 0:
        time.sleep(0.01)
    return time.time() - start

def speak_time_date():
    now = datetime.datetime.now()
    speak(f"Time is {now.strftime('%I:%M %p')}")
    speak(f"Today is {now.strftime('%A, %B %d')}")

def capture_image():
    folder = "/home/project/projecteye/saved_images"
    os.makedirs(folder, exist_ok=True)
    filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
    path = os.path.join(folder, filename)
    try:
        picam2 = Picamera2()
        picam2.configure(picam2.create_still_configuration())
        picam2.start()
        time.sleep(1)
        picam2.capture_file(path)
        picam2.stop()
        picam2.close()
        speak("Image captured")
        print(f"📸 Saved: {path}")
    except Exception as e:
        print("⚠️ Camera error:", e)
        speak("Capture failed")

# 🔵 NEW FUNCTION — Listen for name using voice
def get_name_by_voice():
    speak("Please say the person's name")
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
    try:
        name = r.recognize_google(audio)
        name = name.strip().lower().replace(" ", "_")
        return name
    except:
        speak("I did not understand, try again")
        return None

# 🔵 NEW FUNCTION — Capture face dataset
def capture_new_face(name, img_count=30):
    save_dir = os.path.join("known_faces", name)
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    speak(f"Adding new person: {name}")
    count = 0
    while count < img_count:
        ret, frame = cap.read()
        if not ret: break
        cv2.imwrite(os.path.join(save_dir, f"{count}.jpg"), frame)
        count += 1
        cv2.imshow("Capturing new face", frame)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()
    speak("Face saved successfully")

# 🔵 NEW FUNCTION — Full add new face process
def add_new_face_flow():
    stop_face()  # Stop recognition before adding
    name = get_name_by_voice()
    if not name:
        return
    capture_new_face(name)
    speak("Restarting recognition")
    start_face()

# ---------------- STARTUP ----------------
speak("Device turned on")
print("✅ Smart Eye System Ready")

# ---------------- MAIN LOOP ----------------
try:
    while True:
        # 🔵 KEY 1 — Now supports LONG PRESS to add new face
        if GPIO.input(KEY1) == 0:
            press = detect_press(KEY1)
            if press >= 3:   # long press → add new face
                speak("Add new face mode")
                add_new_face_flow()
            else:           # short press → toggle recognition
                if not face_running: start_face()
                else: stop_face()

        # Key 2 - Object
        if GPIO.input(KEY2) == 0:
            time.sleep(0.25)
            if not object_running: start_object()
            else: stop_object()

        # Key 3 - TTS
        if GPIO.input(KEY3) == 0:
            press = detect_press(KEY3)
            if tts_running and press > 2:
                with open(tts_trigger, "w") as f: f.write("capture")
                speak("Scanning")
            elif press < 1:
                if not tts_running: start_tts()
                else: stop_tts()

        # Key 4 - Currency Detection
        if GPIO.input(KEY4) == 0:
            time.sleep(0.25)
            if not currency_running: start_currency()
            else: stop_currency()

        # Extra Button - Time / Capture
        if GPIO.input(TIME_CAPTURE_BUTTON) == 0:
            press = detect_press(TIME_CAPTURE_BUTTON)
            if press >= 2:
                capture_image()
            else:
                speak_time_date()

        time.sleep(0.1)

except KeyboardInterrupt:
    GPIO.cleanup()
    stop_all()
    speak("Device shutting down safely")
    print("👋 System stopped")
