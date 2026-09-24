import os
import json
import base64
import shutil
from io import BytesIO

from PIL import Image
import bluetooth

KNOWN_FACES_DIR = "/home/project/projecteye/known_faces"
CAPTURED_DIR = "/home/project/projecteye/saved_images"


def ensure_dirs():
    os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
    os.makedirs(CAPTURED_DIR, exist_ok=True)


def list_faces_dict():
    data = {}
    if not os.path.exists(KNOWN_FACES_DIR):
        return data
    for person in os.listdir(KNOWN_FACES_DIR):
        pdir = os.path.join(KNOWN_FACES_DIR, person)
        if os.path.isdir(pdir):
            imgs = [
                f for f in os.listdir(pdir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            data[person] = imgs
    return data


def build_faces_payload():
    result = []
    faces = list_faces_dict()
    for person, images in faces.items():
        if not images:
            continue
        img_path = os.path.join(KNOWN_FACES_DIR, person, images[0])
        try:
            img = Image.open(img_path).resize((160, 160))
            buf = BytesIO()
            img.save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            result.append({"name": person, "image": img_b64})
        except Exception as e:
            print("Error encoding face image:", e)
    return json.dumps(result)


def handle_rename(cmd):
    try:
        payload = cmd.split(":", 1)[1]
        old, new = payload.split("|", 1)
    except Exception:
        print("Bad RENAME cmd:", cmd)
        return
    old_dir = os.path.join(KNOWN_FACES_DIR, old)
    new_dir = os.path.join(KNOWN_FACES_DIR, new)
    if os.path.exists(old_dir):
        os.rename(old_dir, new_dir)
        print(f"{old} renamed to {new}")
    else:
        print(f"{old} not found")


def handle_delete_face(cmd):
    try:
        name = cmd.split(":", 1)[1]
    except Exception:
        print("Bad DELETE cmd:", cmd)
        return
    folder = os.path.join(KNOWN_FACES_DIR, name)
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print(f"{name} deleted")
    else:
        print(f"{name} not found")


def build_captured_payload():
    data = []
    if not os.path.isdir(CAPTURED_DIR):
        return "[]"
    for fname in os.listdir(CAPTURED_DIR):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        path = os.path.join(CAPTURED_DIR, fname)
        try:
            img = Image.open(path).resize((160, 160))
            buf = BytesIO()
            img.save(buf, format="JPEG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            data.append({"name": fname, "image": img_b64})
        except Exception as e:
            print("Error encoding capture image:", e)
    return json.dumps(data)


def handle_delete_capture(cmd):
    try:
        fname = cmd.split(":", 1)[1]
    except Exception:
        print("Bad CAPDEL cmd:", cmd)
        return
    path = os.path.join(CAPTURED_DIR, fname)
    if os.path.exists(path):
        os.remove(path)
        print(f"{fname} deleted")
    else:
        print(f"{fname} not found")


# ---------- UPLOAD STYLE 1: "UPLOAD:<name>" + base64 + END ----------
def handle_upload_stream(client_sock, cmd):
    try:
        person = cmd.split(":", 1)[1].strip()
    except Exception:
        print("Bad UPLOAD cmd:", cmd)
        person = "unknown"

    folder = os.path.join(KNOWN_FACES_DIR, person)
    base_name = person  # samad -> samad1.jpg, samad2.jpg, ...
    print(f"⬆️ Streaming upload for '{person}'")

    os.makedirs(folder, exist_ok=True)

    existing = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    idx = len(existing) + 1
    saved = 0

    while True:
        chunk = client_sock.recv(70000)
        if not chunk:
            break
        text = chunk.decode(errors="ignore").strip()
        if text == "END":
            break
        try:
            raw = base64.b64decode(text)
            img = Image.open(BytesIO(raw)).convert("L")
            fname = os.path.join(folder, f"{base_name}{idx}.jpg")
            img.save(fname)
            print("Saved:", fname)
            idx += 1
            saved += 1
        except Exception as e:
            print("Error saving stream image:", e)

    try:
        client_sock.sendall(b"OK")
    except Exception:
        pass
    print(f"✅ Streaming upload finished for '{person}', images saved: {saved}")


# ---------- UPLOAD STYLE 2: "UPLOAD" + JSON {"name": "...", "images":[...]} ----------
def handle_upload_json(client_sock):
    print("⬆️ JSON upload started")
    buf = bytearray()
    payload = None

    while True:
        try:
            chunk = client_sock.recv(4096)
        except Exception as e:
            print("Upload JSON recv error:", e)
            break
        if not chunk:
            break
        buf.extend(chunk)
        try:
            payload = json.loads(buf.decode("utf-8"))
            break
        except json.JSONDecodeError:
            continue

    if payload is None:
        print("⚠️ Upload JSON: could not parse")
        return

    print("⬇️ JSON payload:", payload)

    name = payload.get("name", "unknown")
    base_name = name  # samad -> samad1.jpg...
    raw_images = payload.get("images", [])
    images_list = None

    if isinstance(raw_images, list):
        images_list = raw_images
    elif isinstance(raw_images, str):
        try:
            tmp = json.loads(raw_images)
            if isinstance(tmp, list):
                images_list = tmp
            else:
                images_list = [raw_images]
        except Exception:
            images_list = [raw_images]
    elif isinstance(payload.get("frames"), list):
        images_list = payload["frames"]

    if images_list is None:
        print("⚠️ Upload JSON: unsupported format for images")
        try:
            client_sock.sendall(b"OK")
        except Exception:
            pass
        return

    print(f"⬆️ Uploading {len(images_list)} images for '{name}' (JSON mode)")

    folder = os.path.join(KNOWN_FACES_DIR, name)
    os.makedirs(folder, exist_ok=True)

    existing = [
        f for f in os.listdir(folder)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    start_index = len(existing) + 1
    saved = 0

    for i, b64 in enumerate(images_list):
        try:
            raw = base64.b64decode(b64)
            img = Image.open(BytesIO(raw)).convert("L")
            index = start_index + i
            fname = os.path.join(folder, f"{base_name}{index}.jpg")
            img.save(fname)
            print("Saved:", fname)
            saved += 1
        except Exception as e:
            print("Error saving JSON image:", e)

    try:
        client_sock.sendall(b"OK")
    except Exception:
        pass
    print(f"✅ JSON upload finished for '{name}', images saved: {saved}")


def main():
    ensure_dirs()
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(("", 1))
    server_sock.listen(1)

    print("RFCOMM server started on channel 1 (NO SDP)")
    print("Pair phone -> then open app & use Sync / Faces / Capture & Upload / Gallery")

    try:
        while True:
            print("Waiting for connection...")
            client_sock, client_info = server_sock.accept()
            print("Connected:", client_info)

            try:
                raw = client_sock.recv(1024).decode(errors="ignore").strip()
                if not raw:
                    print("Empty command")
                    client_sock.close()
                    continue

                print("Received cmd:", raw)

                if raw == "SYNC":
                    payload = build_faces_payload()
                    client_sock.sendall(payload.encode("utf-8"))
                    print("Sent faces JSON")
                elif raw.startswith("RENAME:"):
                    handle_rename(raw)
                elif raw.startswith("DELETE:"):
                    handle_delete_face(raw)
                elif raw == "CAPLIST":
                    payload = build_captured_payload()
                    client_sock.sendall(payload.encode("utf-8"))
                    print("Sent captures JSON")
                elif raw.startswith("CAPDEL:"):
                    handle_delete_capture(raw)
                elif raw.startswith("UPLOAD:"):
                    handle_upload_stream(client_sock, raw)
                elif raw == "UPLOAD":
                    handle_upload_json(client_sock)
                else:
                    print("Unknown cmd:", raw)
                    try:
                        client_sock.sendall(b'{"status":"unknown_cmd"}')
                    except Exception:
                        pass

            except Exception as e:
                print("Error during client session:", e)
            finally:
                client_sock.close()
                print("Client disconnected")

    except KeyboardInterrupt:
        print("Server interrupted. Closing...")
    finally:
        server_sock.close()
        print("Server stopped")


if __name__ == "__main__":
    main()
