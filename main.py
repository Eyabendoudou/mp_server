from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
from google.cloud import storage
from datetime import datetime

main = Flask(__name__)

# --- INIT Firebase Storage ---
BUCKET_NAME = "maxillo-app.firebasestorage.app"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "maxillo-app-firebase-adminsdk-fbsvc-33e4682258.json"

# --- INIT Mediapipe ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

def upload_bytes_to_storage(bytes_data: bytes, destination_path: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_path)
    blob.upload_from_string(bytes_data, content_type='image/jpeg')
    blob.make_public()  # ⚠️ seulement pour test, à désactiver après
    return blob.public_url

@main.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Server running!"})

@main.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image provided"}), 400

    file = request.files['image']
    patient_id = request.form.get('patientId', 'unknown')

    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"success": False, "message": "Invalid image"}), 400

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return jsonify({"success": False, "message": "No face detected"}), 200

    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=[],
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
        )

    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    dest = f"patients/{patient_id}/landmarks_{ts}.jpg"
    download_url = upload_bytes_to_storage(image_bytes, dest)

    return jsonify({
        "success": True,
        "message": "Processed and uploaded",
        "download_url": download_url
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    main.run(host="0.0.0.0", port=port)
