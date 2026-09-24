import os
import shutil
import json

FACE_DIR = "/home/project/projecteye/known_faces"

def list_faces():
    data = {}
    if not os.path.exists(FACE_DIR):
        os.makedirs(FACE_DIR)
        return json.dumps(data)
    for person in os.listdir(FACE_DIR):
        person_folder = os.path.join(FACE_DIR, person)
        if os.path.isdir(person_folder):
            images = [img for img in os.listdir(person_folder)
                      if img.lower().endswith((".jpg", ".jpeg", ".png"))]
            data[person] = images
    return json.dumps(data)

def delete_face(name):
    folder = os.path.join(FACE_DIR, name)
    if os.path.exists(folder):
        shutil.rmtree(folder)
        return f"{name} deleted"
    return f"{name} not found"

def rename_face(old_name, new_name):
    old_folder = os.path.join(FACE_DIR, old_name)
    new_folder = os.path.join(FACE_DIR, new_name)
    if os.path.exists(old_folder):
        os.rename(old_folder, new_folder)
        return f"{old_name} renamed to {new_name}"
    return f"{old_name} not found"
