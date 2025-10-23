from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
from google.cloud import storage
from google.cloud import firestore
from datetime import datetime

app = Flask(__name__)
SERVER_SECRET = os.environ.get("ANALYZER_SECRET", "a63b2f31bd9e236a5220bcaee53ea16e300454e3c27e95f8d6c46ceea6abe09e")
# --- INIT Firebase Storage + Firestore ---
BUCKET_NAME = "maxillo-app.firebasestorage.app"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "maxillo-app-firebase-adminsdk-fbsvc-33e4682258.json"

# --- clients
storage_client = storage.Client()
firestore_client = firestore.Client()

# --- INIT Mediapipe ---
mp_face_mesh = mp.solutions.face_mesh

def upload_bytes_to_storage(bytes_data: bytes, destination_path: str) -> str:
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(destination_path)
    blob.upload_from_string(bytes_data, content_type='image/jpeg')
    blob.make_public()  # ⚠️ only for testing, remove in production
    return blob.public_url

# --- utility functions for clinical measures (based on your colab) ---
def _dist(p1, p2):
    if p1 is None or p2 is None:
        return None
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))

def _get_px_point(landmarks, idx, width, height):
    try:
        pt = landmarks[idx]
        return (int(pt.x * width), int(pt.y * height))
    except Exception:
        return None

def calculate_clinical_measures_from_landmarks(landmarks, img_w, img_h):
    """
    landmarks: list of normalized landmarks (each has x,y,z)
    returns: dict of measures (keys similar to your colab)
    """
    # indices from your colab
    forehead_idx = 10
    chin_idx = 152
    eye_left_idx = 468
    eye_right_idx = 473
    jaw_left_idx = 234
    jaw_right_idx = 454

    # convert normalized to pixel points
    forehead = _get_px_point(landmarks, forehead_idx, img_w, img_h)
    chin = _get_px_point(landmarks, chin_idx, img_w, img_h)
    eye_left = _get_px_point(landmarks, eye_left_idx, img_w, img_h)
    eye_right = _get_px_point(landmarks, eye_right_idx, img_w, img_h)
    jaw_left = _get_px_point(landmarks, jaw_left_idx, img_w, img_h)
    jaw_right = _get_px_point(landmarks, jaw_right_idx, img_w, img_h)

    # store all landmarks in normalized form (x,y,z)
    all_landmarks = [{"x": float(l.x), "y": float(l.y), "z": float(l.z)} for l in landmarks]

    measures = {}
    # Interpupillary distance in pixels
    ipd_px = None
    if eye_left and eye_right:
        ipd_px = _dist(eye_left, eye_right)

    if ipd_px and ipd_px > 0:
        ipd_mm = 63.0  # calibration constant (adult average)
        px_to_mm = ipd_mm / ipd_px
        measures['px_to_mm_ratio'] = float(px_to_mm)

        measures['face_height_mm'] = _dist(forehead, chin) * px_to_mm if forehead and chin else None
        measures['face_width_mm'] = _dist(jaw_left, jaw_right) * px_to_mm if jaw_left and jaw_right else None
        if measures.get('face_height_mm') and measures.get('face_width_mm'):
            measures['height_width_ratio'] = float(measures['face_height_mm'] / measures['face_width_mm'])
        else:
            measures['height_width_ratio'] = None

        if forehead and chin:
            axis_x = (forehead[0] + chin[0]) // 2
            measures['chin_deviation_mm'] = float((chin[0] - axis_x) * px_to_mm)
        else:
            measures['chin_deviation_mm'] = None

        measures['jaw_left_mm'] = _dist(jaw_left, chin) * px_to_mm if jaw_left and chin else None
        measures['jaw_right_mm'] = _dist(jaw_right, chin) * px_to_mm if jaw_right and chin else None

        symmetric_pairs_indices = [(234, 454), (93, 323), (132, 361), (58, 288)]
        asym_diffs = []
        for left_idx, right_idx in symmetric_pairs_indices:
            pl = _get_px_point(landmarks, left_idx, img_w, img_h)
            pr = _get_px_point(landmarks, right_idx, img_w, img_h)
            if pl and pr:
                diff_px = abs(pl[0] - pr[0])
                asym_diffs.append(diff_px * px_to_mm)
        measures['avg_asymmetry_mm'] = float(np.mean(asym_diffs)) if asym_diffs else None
    else:
        # Fallback to pixel measures
        measures['face_height_px'] = _dist(forehead, chin) if forehead and chin else None
        measures['face_width_px'] = _dist(jaw_left, jaw_right) if jaw_left and jaw_right else None
        if measures.get('face_height_px') and measures.get('face_width_px'):
            measures['height_width_ratio'] = float(measures['face_height_px'] / measures['face_width_px'])
        else:
            measures['height_width_ratio'] = None

        if forehead and chin:
            axis_x = (forehead[0] + chin[0]) // 2
            measures['chin_deviation_px'] = float(chin[0] - axis_x)
        else:
            measures['chin_deviation_px'] = None

        symmetric_pairs_indices = [(234, 454), (93, 323), (132, 361), (58, 288)]
        asym_diffs_px = []
        for left_idx, right_idx in symmetric_pairs_indices:
            pl = _get_px_point(landmarks, left_idx, img_w, img_h)
            pr = _get_px_point(landmarks, right_idx, img_w, img_h)
            if pl and pr:
                diff_px = abs(pl[0] - pr[0])
                asym_diffs_px.append(diff_px)
        measures['avg_asymmetry_px'] = float(np.mean(asym_diffs_px)) if asym_diffs_px else None

    return measures, all_landmarks

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

    h, w = image.shape[:2]
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return jsonify({"success": False, "message": "No face detected"}), 200

    # use first face
    face_landmarks = results.multi_face_landmarks[0]
    # draw landmarks on image (same as before)
    mp.solutions.drawing_utils.draw_landmarks(
        image=image,
        landmark_list=face_landmarks,
        connections=[],
        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
    )

    # normalized landmarks list
    landmark_list = []
    for lm in face_landmarks.landmark:
        landmark_list.append({"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)})

    # calculate clinical measures
    clinical_measures, all_landmarks = calculate_clinical_measures_from_landmarks(face_landmarks.landmark, w, h)

    # encode annotated image
    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    dest = f"patients/{patient_id}/landmarks_{ts}.jpg"
    download_url = upload_bytes_to_storage(image_bytes, dest)

    # Persist an analysis document to Firestore (patients/{patientId}/analyses)
    try:
        doc_ref = firestore_client.collection('patients').document(patient_id).collection('analyses').document()
        analysis_doc = {
            'createdAt': firestore.SERVER_TIMESTAMP,
            'resultImageUrl': download_url,
            'landmarks': all_landmarks,
            'clinicalMeasures': clinical_measures,
            'status': 'done',
        }
        doc_ref.set(analysis_doc)

        # also write lastAnalysis on patient doc for compatibility
        patient_doc = firestore_client.collection('patients').document(patient_id)
        patient_doc.set({'lastAnalysis': analysis_doc}, merge=True)
    except Exception as e:
        # Log but don't fail the entire request
        print("Firestore write error:", e)

    return jsonify({
        "success": True,
        "message": "Processed and uploaded",
        "download_url": download_url,
        "landmarks": landmark_list,
        "clinicalMeasures": clinical_measures
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
