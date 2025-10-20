from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
from google.cloud import storage
from datetime import datetime

app = Flask(__name__)
SERVER_SECRET = os.environ.get("ANALYZER_SECRET", "a63b2f31bd9e236a5220bcaee53ea16e300454e3c27e95f8d6c46ceea6abe09e")  # set real secret in Railway env
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

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Server running!"})

@app.route('/analyze', methods=['POST'])
def analyze():
    auth_header = request.headers.get("Authorization")
    if auth_header is None or auth_header != f"Bearer {SERVER_SECRET}":
        return jsonify({"success": False, "message": "Unauthorized"}), 401
    
    if 'image' not in request.files:
        return jsonify({"success": False, "message": "No image provided"}), 400
    
    file = request.files['image']
    patient_id = request.form.get('patientId', 'unknown')

    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({"success": False, "message": "Invalid image"}), 400

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    landmark_list = []
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
    for lm in face_landmarks.landmark:
        landmark_list.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)})

    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    dest = f"patients/{patient_id}/landmarks_{ts}.jpg"
    download_url = upload_bytes_to_storage(image_bytes, dest)

    return jsonify({
        "success": True,
        "message": "Processed and uploaded",
        "download_url": download_url,
        "landmarks": landmark_list
        
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
