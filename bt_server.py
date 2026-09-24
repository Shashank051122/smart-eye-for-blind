import os
import json
import base64
from io import BytesIO

from PIL import Image
import bluetooth  # pybluez

KNOWN_FACES_DIR = "/home/project/projecteye/known_faces"

SERVICE_NAME = "SmartEyePi"
SERVICE_UUID = "00001101-0000-1000-8000-00805F9B34FB"  # Standard SPP UUID


def build_face_payload():
    """
    Scan known_faces/ and build JSON:
    [
      {"name": "ameen", "image": "<base64>"},
      ...
    ]
    Only the *first* image of each person is sent.
    """
    data = []

    if not os.path.isdir(KNOWN_FACES_DIR):
        print("⚠️ known_faces dir not found:", KNOWN_FACES_DIR)
        return "[]"

    for person in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person)
        if not os.path.isdir(person_dir):
            continue

        imgs = [f for f in os.listdir(person_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if not imgs:
            continue

        first_img_path = os.path.join(person_dir, imgs[0])
        try:
            img = Image.open(first_img_path)
            img = img.resize((160, 160))
            buf = BytesIO()
            img.save(buf, format="JPEG")
            img_bytes = buf.getvalue()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            data.append({
                "name": person,
                "image": img_b64
            })
        except Exception as e:
            print("Error loading image for", person, ":", e)

    return json.dumps(data)


def main():
    # Create RFCOMM socket
    server_sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    server_sock.bind(("", bluetooth.PORT_ANY))
    server_sock.listen(1)

    port = server_sock.getsockname()[1]

    # Advertise as Serial Port Profile
    bluetooth.advertise_service(
        server_sock,
        SERVICE_NAME,
        service_id=SERVICE_UUID,
        service_classes=[SERVICE_UUID, bluetooth.SERIAL_PORT_CLASS],
        profiles=[bluetooth.SERIAL_PORT_PROFILE]
    )

    print(f"📡 RFCOMM server started on channel {port}")
    print("📱 Make sure phone is paired with 'raspberrypi',")
    print("   then open app → Sync → Scan for device")

    try:
        client_sock, client_info = server_sock.accept()
        print("✅ Connection from", client_info)

        # Expect simple command from Android
        cmd = client_sock.recv(1024).decode(errors="ignore").strip()
        print("📥 Received:", cmd)

        if cmd.upper() == "SYNC":
            payload = build_face_payload()
            client_sock.send(payload.encode("utf-8"))
            print("📤 Sent face-data JSON (length:", len(payload), ")")
        else:
            print("⚠️ Unknown command")

    except Exception as e:
        print("❌ Server error:", e)
    finally:
        try:
            client_sock.close()
        except Exception:
            pass
        server_sock.close()
        print("🔻 Server stopped")


if __name__ == "__main__":
    main()
