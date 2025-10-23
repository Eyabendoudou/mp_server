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

def calculate_proportionality_from_landmarks(landmarks, img_w, img_h):
    """
    Compute facial thirds (upper/middle/lower) in mm and produce an assessment.
    Returns dict with upper_third_mm, middle_third_mm, lower_third_mm, proportionality_assessment
    and pixel y-coordinates for drawing.
    """
    try:
        # build pixel points array
        pts = [(int(lm.x * img_w), int(lm.y * img_h)) for lm in landmarks]
        # eyebrow top: min y across a broad eyebrow region
        top_eyebrow_y = min(pts[i][1] for i in range(65, 296))
        glabella_y = pts[9][1]
        subnasale_y = pts[2][1]
        menton_y = pts[152][1]

        # pupils for calibration
        left_pupil = np.array(pts[468])
        right_pupil = np.array(pts[473])
        ipd_pixels = float(np.linalg.norm(left_pupil - right_pupil))
        if ipd_pixels == 0:
            return None

        mm_per_pixel = 63.0 / ipd_pixels

        # estimate trichion (top of forehead) by extrapolation
        estimated_upper_face_px = glabella_y - top_eyebrow_y
        trichion_y = max(int(top_eyebrow_y - estimated_upper_face_px), 0)

        # sort vertical coordinates robustly
        y_coords = sorted([trichion_y, glabella_y, subnasale_y, menton_y])
        trichion_y, glabella_y, subnasale_y, menton_y = y_coords

        upper_px = glabella_y - trichion_y
        middle_px = subnasale_y - glabella_y
        lower_px = menton_y - subnasale_y
        total_px = menton_y - trichion_y
        if total_px <= 0:
            return None

        upper_mm = float(upper_px * mm_per_pixel)
        middle_mm = float(middle_px * mm_per_pixel)
        lower_mm = float(lower_px * mm_per_pixel)

        tolerance_mm = 10.0
        if (abs(upper_mm - middle_mm) <= tolerance_mm and
                abs(middle_mm - lower_mm) <= tolerance_mm and
                abs(upper_mm - lower_mm) <= tolerance_mm):
            assessment = "Proportions harmonieuses"
        else:
            assessment = "Proportions non harmonieuses"

        return {
            'upper_third_mm': upper_mm,
            'middle_third_mm': middle_mm,
            'lower_third_mm': lower_mm,
            'proportionality_assessment': assessment,
            'mm_per_pixel': mm_per_pixel,
            # pixel coordinates for drawing on image
            'trichion_y_px': int(trichion_y),
            'glabella_y_px': int(glabella_y),
            'subnasale_y_px': int(subnasale_y),
            'menton_y_px': int(menton_y),
        }
    except Exception as e:
        print("Proportionality calc error:", e)
        return None

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

    # calculate proportionality from same landmarks
    proportionality = calculate_proportionality_from_landmarks(face_landmarks.landmark, w, h)

    # encode annotated landmark image (existing)
    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    dest = f"patients/{patient_id}/landmarks_{ts}.jpg"
    download_url = upload_bytes_to_storage(image_bytes, dest)

    # Create proportionality annotated image (horizontal lines + labels) if proportionality available
    proportion_image_url = None
    if proportionality:
        try:
            prop_img = image.copy()
            color_line = (0, 120, 255)  # orange-ish
            thickness = 2
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.6, min(w, h) / 800)  # scale with image size
            text_color = (255, 255, 255)
            # y positions
            t_y = int(proportionality.get('trichion_y_px', 0))
            g_y = int(proportionality.get('glabella_y_px', 0))
            s_y = int(proportionality.get('subnasale_y_px', 0))
            m_y = int(proportionality.get('menton_y_px', 0))
            # draw horizontal lines across the image width
            cv2.line(prop_img, (0, t_y), (w, t_y), color_line, thickness)
            cv2.line(prop_img, (0, g_y), (w, g_y), color_line, thickness)
            cv2.line(prop_img, (0, s_y), (w, s_y), color_line, thickness)
            cv2.line(prop_img, (0, m_y), (w, m_y), color_line, thickness)
            # write mm values near right side
            try:
                upper_mm = proportionality.get('upper_third_mm')
                middle_mm = proportionality.get('middle_third_mm')
                lower_mm = proportionality.get('lower_third_mm')
                if upper_mm is not None:
                    cv2.putText(prop_img, f"Upper: {upper_mm:.1f} mm", (10, max(15, t_y - 10)), font, font_scale, text_color, 2, cv2.LINE_AA)
                if middle_mm is not None:
                    cv2.putText(prop_img, f"Middle: {middle_mm:.1f} mm", (10, max(15, g_y - 10)), font, font_scale, text_color, 2, cv2.LINE_AA)
                if lower_mm is not None:
                    cv2.putText(prop_img, f"Lower: {lower_mm:.1f} mm", (10, max(15, s_y - 10)), font, font_scale, text_color, 2, cv2.LINE_AA)
                # assessment text at bottom
                assessment = proportionality.get('proportionality_assessment')
                if assessment:
                    cv2.putText(prop_img, assessment, (10, h - 10), font, max(0.7, font_scale), (220, 220, 220), 2, cv2.LINE_AA)
            except Exception:
                pass

            _, prop_buffer = cv2.imencode('.jpg', prop_img)
            prop_bytes = prop_buffer.tobytes()
            dest_prop = f"patients/{patient_id}/proportionality_{ts}.jpg"
            proportion_image_url = upload_bytes_to_storage(prop_bytes, dest_prop)
            # also attach image url into proportionality dict
            proportionality['imageUrl'] = proportion_image_url
        except Exception as e:
            print("Proportionality image generation/upload error:", e)
            proportion_image_url = None

    # Persist an analysis document to Firestore (patients/{patientId}/analyses)
    try:
        doc_ref = firestore_client.collection('patients').document(patient_id).collection('analyses').document()
        # embed proportionality (with imageUrl if generated)
        analysis_doc = {
            'createdAt': firestore.SERVER_TIMESTAMP,
            'resultImageUrl': download_url,
            'landmarks': all_landmarks,
            'clinicalMeasures': clinical_measures,
            'proportionality': proportionality,
            # keep backward-compatible key
            'proportion_image': proportion_image_url,
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
        "clinicalMeasures": clinical_measures,
        "proportionality": proportionality
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
