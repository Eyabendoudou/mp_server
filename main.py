# main.py
from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
from google.cloud import storage, firestore
from datetime import datetime
import math
import logging

# ---------- Configuration ----------
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

SERVER_SECRET = os.environ.get(
    "ANALYZER_SECRET",
    "a63b2f31bd9e236a5220bcaee53ea16e300454e3c27e95f8d6c46ceea6abe09e"
)

# Firebase / GCP
BUCKET_NAME = os.environ.get("BUCKET_NAME", "maxillo-app.firebasestorage.app")
# Point to service account JSON (make sure the file is present in your deployed container)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ.get(
    "GOOGLE_APPLICATION_CREDENTIALS", "maxillo-app-firebase-adminsdk-fbsvc-33e4682258.json"
)

# Mediapipe init
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# ---------- Helpers ----------

def upload_bytes_to_storage(bytes_data: bytes, destination_path: str) -> str:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_path)
    blob.upload_from_string(bytes_data, content_type='image/jpeg')
    blob.make_public()  # for testing only — recommended: use signed URLs or proper security
    return blob.public_url

def save_analysis_to_firestore(patient_id: str, analysis_id: str, payload: dict):
    db = firestore.Client()
    analyses_col = db.collection('patients').document(patient_id).collection('analyses')
    analyses_col.document(analysis_id).set(payload)
    # also update patient document's lastAnalysis
    db.collection('patients').document(patient_id).set({
        'lastAnalysis': payload
    }, merge=True)

# Utility math
def distance(p1, p2):
    if p1 is None or p2 is None:
        return None
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def angle_between(p1, p2):
    if p1 is None or p2 is None:
        return None
    delta_y = p2[1] - p1[1]
    delta_x = p2[0] - p1[0]
    return float(np.degrees(np.arctan2(delta_y, delta_x)))

# Convert normalized mediapipe landmarks to pixel coordinates (x,y)
def landmark_norm_to_pixel(lm, width, height):
    return (int(lm.x * width), int(lm.y * height))

# Clinical measures based on landmarks (adapted from your Colab code)
def compute_clinical_measures(landmarks_norm, image_shape):
    """
    landmarks_norm: list of {'x','y','z'} normalized coords (0..1)
    image_shape: (height, width, channels)
    returns: dict of measures (numbers or None)
    """
    h, w = image_shape[:2]
    # helper to safely get pixel point by index
    def get_pt(idx):
        if idx < 0 or idx >= len(landmarks_norm):
            return None
        lm = landmarks_norm[idx]
        return (lm['x'] * w, lm['y'] * h)  # keep as float for better precision

    # Indices chosen from your Colab script
    forehead_idx = 10
    chin_idx = 152
    eye_left_idx = 468  # left iris/eye (refined)
    eye_right_idx = 473
    jaw_left_idx = 234
    jaw_right_idx = 454

    forehead = get_pt(forehead_idx)
    chin = get_pt(chin_idx)
    eye_left = get_pt(eye_left_idx)
    eye_right = get_pt(eye_right_idx)
    jaw_left = get_pt(jaw_left_idx)
    jaw_right = get_pt(jaw_right_idx)

    measures = {}

    # Prepare all-landmarks length safe check
    # Compute interpupillary distance in px
    ipd_px = None
    if eye_left and eye_right:
        ipd_px = distance(eye_left, eye_right)

    if ipd_px and ipd_px > 0:
        # calibration constant: average adult IPD in mm (approx), user can adapt per population
        ipd_mm = 63.0
        px_to_mm = ipd_mm / ipd_px
        measures['px_to_mm_ratio'] = px_to_mm

        # face height / width
        measures['face_height_mm'] = distance(forehead, chin) * px_to_mm if forehead and chin else None
        measures['face_width_mm']  = distance(jaw_left, jaw_right) * px_to_mm if jaw_left and jaw_right else None
        if measures.get('face_height_mm') and measures.get('face_width_mm'):
            measures['height_width_ratio'] = float(measures['face_height_mm'] / measures['face_width_mm'])
        else:
            measures['height_width_ratio'] = None

        # chin deviation: difference from midline between forehead and chin
        if forehead and chin:
            axis_x = (forehead[0] + chin[0]) / 2.0
            measures['chin_deviation_mm'] = float((chin[0] - axis_x) * px_to_mm)
        else:
            measures['chin_deviation_mm'] = None

        # Mandibular lengths
        measures['jaw_left_mm'] = distance(jaw_left, chin) * px_to_mm if jaw_left and chin else None
        measures['jaw_right_mm'] = distance(jaw_right, chin) * px_to_mm if jaw_right and chin else None

        # avg asymmetry across some symmetric pairs
        symmetric_pairs_indices = [(234, 454), (93, 323), (132, 361), (58, 288)]
        asym_diffs = []
        for left_idx, right_idx in symmetric_pairs_indices:
            pl = get_pt(left_idx)
            pr = get_pt(right_idx)
            if pl and pr:
                diff_px = abs(pl[0] - pr[0])
                asym_diffs.append(diff_px * px_to_mm)
        measures['avg_asymmetry_mm'] = float(np.mean(asym_diffs)) if asym_diffs else None

    else:
        # fallback to pixel-based measures if IPD not available
        measures['face_height_px'] = distance(forehead, chin) if forehead and chin else None
        measures['face_width_px']  = distance(jaw_left, jaw_right) if jaw_left and jaw_right else None
        if measures.get('face_height_px') and measures.get('face_width_px'):
            measures['height_width_ratio'] = float(measures['face_height_px'] / measures['face_width_px'])
        else:
            measures['height_width_ratio'] = None

        if forehead and chin:
            axis_x = (forehead[0] + chin[0]) / 2.0
            measures['chin_deviation_px'] = float((chin[0] - axis_x))
        else:
            measures['chin_deviation_px'] = None

        symmetric_pairs_indices = [(234, 454), (93, 323), (132, 361), (58, 288)]
        asym_diffs_px = []
        for left_idx, right_idx in symmetric_pairs_indices:
            pl = get_pt(left_idx)
            pr = get_pt(right_idx)
            if pl and pr:
                diff_px = abs(pl[0] - pr[0])
                asym_diffs_px.append(diff_px)
        measures['avg_asymmetry_px'] = float(np.mean(asym_diffs_px)) if asym_diffs_px else None

    return measures

# ---------- API endpoints ----------

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Server running!"})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # --- auth ---
        auth_header = request.headers.get("Authorization")
        if auth_header is None or auth_header != f"Bearer {SERVER_SECRET}":
            return jsonify({"success": False, "message": "Unauthorized"}), 401

        # --- image ---
        if 'image' not in request.files:
            return jsonify({"success": False, "message": "No image provided"}), 400

        file = request.files['image']
        patient_id = request.form.get('patientId', 'unknown')

        npimg = np.frombuffer(file.read(), np.uint8)
        image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({"success": False, "message": "Invalid image"}), 400

        # --- mediapipe face mesh ---
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return jsonify({"success": False, "message": "No face detected"}), 200

        # draw landmarks on image (no connections)
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=[],
                landmark_drawing_spec=mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
            )

        # prepare normalized landmarks list and pixel list
        landmark_list_norm = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmark_list_norm.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)})

        # compute measures
        measures = compute_clinical_measures(landmark_list_norm, image.shape)

        # re-encode image and upload
        _, buffer = cv2.imencode('.jpg', image)
        image_bytes = buffer.tobytes()
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        dest = f"patients/{patient_id}/landmarks_{ts}.jpg"
        download_url = upload_bytes_to_storage(image_bytes, dest)

        # Prepare analysis payload to save in Firestore
        analysis_id = ts
        payload = {
            "createdAt": datetime.utcnow(),
            "patientId": patient_id,
            "download_url": download_url,
            "landmarks": landmark_list_norm,
            "measures": measures,
            "status": "done",
            "processedBy": "mediapipe-server"
        }

        # Save to Firestore
        try:
            save_analysis_to_firestore(patient_id, analysis_id, payload)
        except Exception as e:
            logging.exception("Failed to save analysis to Firestore")
            # don't fail the request — just warn in logs
            payload['firestore_error'] = str(e)

        # Return response
        # convert any non-serializable fields (datetime) — createdAt we used datetime object; convert to iso
        payload_resp = payload.copy()
        payload_resp['createdAt'] = payload_resp['createdAt'].isoformat()
        return jsonify({
            "success": True,
            "message": "Processed, measures computed, uploaded and saved",
            "analysisId": analysis_id,
            "download_url": download_url,
            "landmarks": landmark_list_norm,
            "measures": measures,
        })

    except Exception as e:
        logging.exception("Internal error")
        return jsonify({"success": False, "message": "Internal server error", "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
