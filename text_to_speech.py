import cv2
import pytesseract
import os
import time
from picamera2 import Picamera2
import signal
import sys

trigger_file = "/home/project/projecteye/tts_trigger.txt"

def speak(text):
    print(f"🗣️ {text}")
    os.system(f'espeak "{text}" --stdout | aplay > /dev/null 2>&1')

running = True

def handle_exit(sig, frame):
    global running
    running = False

signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT, handle_exit)

def text_to_speech():
    print("🟢 Text Reader Started")
    speak("Text reader started")

    # --- Initialize Camera ---
    picam2 = Picamera2()
    picam2.preview_configuration.main.size = (640, 480)
    picam2.preview_configuration.main.format = "RGB888"
    picam2.configure("preview")
    picam2.start()

    time.sleep(1)

    while running:
        frame = picam2.capture_array()

        # Check for trigger file
        if os.path.exists(trigger_file):
            with open(trigger_file) as f:
                command = f.read().strip()

            if command == "capture":
                print("📸 Capturing text...")
                speak("Capturing text")

                text = pytesseract.image_to_string(frame)

                if text.strip():
                    print("📝 Detected Text:\n", text)
                    speak(f"Detected text is {text}")
                else:
                    print("⚠️ No readable text found")
                    speak("No readable text found")

                open(trigger_file, "w").close()  # clear trigger

            elif command == "stop":
                speak("Stopping text reader")
                print("🛑 Text Reader Stopped")
                open(trigger_file, "w").close()
                break

        time.sleep(0.1)

    picam2.stop()
    sys.exit()

if __name__ == "__main__":
    text_to_speech()
