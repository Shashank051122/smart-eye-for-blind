# Smart Eye for Blind People

An AI-powered assistive system designed to help visually impaired people understand their surroundings using computer vision, object detection, distance sensing, and audio feedback.

## Project Overview

Smart Eye combines a Raspberry Pi, camera, ultrasonic sensor, and audio feedback to provide useful information to the user.

The project includes computer-vision models and Python modules for object detection, face recognition, text-to-speech, Bluetooth communication, and related assistive functions.

## Hardware

* Raspberry Pi 4 Model B
* Raspberry Pi Camera Module 2
* HC-SR04 Ultrasonic Sensor
* Buzzer
* 4×1 Matrix Keypad
* Push Buttons
* Earphones
* Power Bank
* VR Box

## Software & Technologies

* Python
* Raspberry Pi OS
* OpenCV
* NumPy
* Picamera2
* PyTorch
* ONNX Runtime
* YOLO models
* VS Code

## Key Features

* Real-time object detection
* Distance and obstacle sensing
* Face-related recognition functionality
* Audio feedback and text-to-speech
* Bluetooth communication
* Support for multiple computer-vision models
* Raspberry Pi based edge processing

## Project Structure

The repository contains the main Python applications and supporting modules, model files, ONNX assets, object-detection implementations, and supporting resources.

Generated runtime outputs, Python virtual environments, and personal face-data directories are excluded using `.gitignore`.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Shashank051122/smart-eye-for-blind.git
cd smart-eye-for-blind
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

Install the required Python packages used by the project. Some components are intended to run on Raspberry Pi and may require Raspberry Pi-specific system packages.

### 4. Connect the Hardware

Connect the Raspberry Pi camera, ultrasonic sensor, buzzer, keypad, buttons, and audio output according to your hardware wiring.

### 5. Run the Application

The repository contains multiple Python entry points and experimental modules. Use the appropriate main script for the functionality you want to run on the Raspberry Pi.

## Notes

This project was developed as an assistive technology prototype combining embedded systems and AI/computer vision.

Hardware-specific configuration, GPIO wiring, model dependencies, and runtime settings may need to be adjusted for your Raspberry Pi setup.

## Author

**Shashank MY**

GitHub: https://github.com/Shashank051122

LinkedIn: https://linkedin.com/in/shashank-my051122

