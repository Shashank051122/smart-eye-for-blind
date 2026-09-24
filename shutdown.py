import RPi.GPIO as GPIO
import os
import time

BUTTON_PIN = 3   # GPIO3 (pin 5)

GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON_PIN, GPIO.IN)

print("Shutdown monitor active")

while True:
    if GPIO.input(BUTTON_PIN) == 0:
        print("Shutdown button pressed")
        os.system("sudo /usr/sbin/shutdown -h now")
        time.sleep(3)  # allow shutdown command to execute
    time.sleep(0.1)
