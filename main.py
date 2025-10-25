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

def test_symetrie_mm(landmarks_points, ipd_mm=63.0):
    """
    Compute average facial symmetry difference in mm and return (avg_mm, interpretation)
    Uses the pair indices from your Colab script.
    """
    try:
        if landmarks_points is None or len(landmarks_points) < 474:
            return None, "Landmarks insuffisants"

        center_x = landmarks_points[1][0]
        eye_left = landmarks_points[468]
        eye_right = landmarks_points[473]

        ipd_px = float(np.linalg.norm(np.array(eye_left) - np.array(eye_right)))
        px_to_mm = ipd_mm / ipd_px if ipd_px != 0 else 1.0

        symmetric_pairs = [
            (33, 263), (133, 362), (97, 326), (61, 291)
        ]

        total_diff_mm = 0.0
        valid = 0
        for left_idx, right_idx in symmetric_pairs:
            if left_idx < len(landmarks_points) and right_idx < len(landmarks_points):
                pl = landmarks_points[left_idx]
                pr = landmarks_points[right_idx]
                diff_mm = abs(abs(pl[0]-center_x) - abs(pr[0]-center_x)) * px_to_mm
                total_diff_mm += diff_mm
                valid += 1

        if valid == 0:
            return None, "Pas de paires valides"

        average_diff_mm = float(total_diff_mm / valid)

        if average_diff_mm < 3:
            sym_text = "Symetrie faciale bonne"
        elif average_diff_mm < 10:
            sym_text = "Symetrie moderee"
        else:
            sym_text = "Asymetrie notable"

        return average_diff_mm, sym_text
    except Exception as e:
        print("Symmetry calc error:", e)
        return None, "Erreur calcul symetrie"

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

    # create timestamp right away so all image generators can use the same ts safely
    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    # keep original frontal image untouched
    original_image = image.copy()

    h, w = image.shape[:2]
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return jsonify({"success": False, "message": "No face detected"}), 200

    # use first face
    face_landmarks = results.multi_face_landmarks[0]

    # --- create landmark image by drawing on a copy (so original remains untouched) ---
    landmark_img = original_image.copy()
    mp.solutions.drawing_utils.draw_landmarks(
        image=landmark_img,
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

    # --- SYMMETRY TEST (compute metrics + annotated image) ---
    symmetry = None
    symmetry_image_url = None
    try:
        # prepare pixel landmark points
        landmarks_points = [(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks.landmark]
        avg_diff_mm, sym_text = test_symetrie_mm(landmarks_points)

        symmetry = {
            'meanAsymmetry': float(avg_diff_mm) if avg_diff_mm is not None else None,
            'status': sym_text or '',
        }

        # Create symmetry annotated image (use original frontal image)
        try:
            sym_img = original_image.copy()
            center_x = landmarks_points[1][0] if len(landmarks_points) > 1 else w // 2
            # colors and thickness
            center_color = (255, 0, 0)  # blue-ish line (BGR)
            pair_color = (0, 255, 0)    # green for pair markers
            thickness = 3
            # draw vertical center line
            cv2.line(sym_img, (center_x, 0), (center_x, h), center_color, thickness, cv2.LINE_AA)

            # draw symmetric pairs connectors and markers
            pairs = [(33, 263), (133, 362), (97, 326), (61, 291)]
            for left_idx, right_idx in pairs:
                if left_idx < len(landmarks_points) and right_idx < len(landmarks_points):
                    pl = landmarks_points[left_idx]
                    pr = landmarks_points[right_idx]
                    # draw small circles
                    cv2.circle(sym_img, pl, 4, pair_color, -1, cv2.LINE_AA)
                    cv2.circle(sym_img, pr, 4, pair_color, -1, cv2.LINE_AA)
                    # draw line between pair
                    cv2.line(sym_img, pl, pr, pair_color, 2, cv2.LINE_AA)
                    # draw small horizontal markers to center distances
                    cv2.line(sym_img, (pl[0], pl[1]-8), (pl[0], pl[1]+8), (255,255,255), 1)
                    cv2.line(sym_img, (pr[0], pr[1]-8), (pr[0], pr[1]+8), (255,255,255), 1)

            # annotated avg and interpretation text (bold background)
            def put_bg_text(img, text, pos, scale=0.8, thickness_txt=2):
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness_txt)
                x, y = pos
                x1 = max(0, x - 6)
                y1 = max(0, y - th - 6)
                x2 = min(img.shape[1], x + tw + 6)
                y2 = min(img.shape[0], y + baseline + 6)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,0), -1)
                cv2.putText(img, text, (x, y), font, scale, (255,255,255), thickness_txt, cv2.LINE_AA)

            if avg_diff_mm is not None:
                put_bg_text(sym_img, f"Asym. moy: {avg_diff_mm:.2f} mm", (10, 30), 0.8, 2)
            put_bg_text(sym_img, sym_text, (10, h - 20), 0.8, 2)

            _, sym_buf = cv2.imencode('.jpg', sym_img)
            sym_bytes = sym_buf.tobytes()
            dest_sym = f"patients/{patient_id}/symmetry_{ts}.jpg"
            symmetry_image_url = upload_bytes_to_storage(sym_bytes, dest_sym)
            symmetry['imageUrl'] = symmetry_image_url
        except Exception as e:
            print("Symmetry image gen/upload error:", e)
            symmetry_image_url = None
    except Exception as e:
        print("Symmetry overall error:", e)
        symmetry = {'meanAsymmetry': None, 'status': 'Erreur de calcul'}

    # encode annotated landmark image (existing)
    _, buffer = cv2.imencode('.jpg', image)
    image_bytes = buffer.tobytes()

    # upload landmark image (resultImageUrl should point to landmark image)
    _, buffer_land = cv2.imencode('.jpg', landmark_img)
    landmark_bytes = buffer_land.tobytes()
    dest_land = f"patients/{patient_id}/landmarks_{ts}.jpg"
    landmark_url = upload_bytes_to_storage(landmark_bytes, dest_land)

    # Create proportionality annotated image from ORIGINAL frontal image (no landmarks overlay)
    proportion_image_url = None
    if proportionality:
        try:
            prop_img = original_image.copy()  # use original frontal image

            # Colors & thickness to match your Colab:
            yellow = (0, 255, 255)  # BGR yellow
            green = (0, 255, 0)     # BGR green
            line_thickness = 5      # thicker lines like Colab
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.8, min(w, h) / 700)  # readable scale
            text_color = (255, 255, 255)
            rect_bg_color = (0, 0, 0)

            # helper to draw bold text with background for legibility
            def put_bold_text(img, text, org, font, font_scale, text_color, thickness):
                (txt_w, txt_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                x, y = org
                # ensure background rectangle stays inside image
                x0 = max(0, x - 6)
                y0 = max(0, y - txt_h - 6)
                x1 = min(img.shape[1], x + txt_w + 6)
                y1 = min(img.shape[0], y + baseline + 6)
                cv2.rectangle(img, (x0, y0), (x1, y1), rect_bg_color, -1)
                cv2.putText(img, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

            # new helper: draw label slightly above a horizontal line while keeping it inside image
            def put_label_above_line(img, text, line_y, x, font, font_scale, text_color, thickness):
                (txt_w, txt_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                # desired baseline y just above the line
                desired_y = line_y - 8
                min_y = txt_h + 8  # minimum baseline to avoid clipping at top
                max_y = img.shape[0] - 8  # maximum baseline to avoid clipping at bottom
                if desired_y < min_y:
                    # not enough space above line — place below the line
                    desired_y = line_y + txt_h + 8
                # clamp to image
                desired_y = int(min(max(desired_y, min_y), max_y))
                # draw background rect + text (x used as left padding)
                x0 = max(0, x - 6)
                y0 = int(desired_y - txt_h - 6)
                x1 = int(min(img.shape[1], x + txt_w + 6))
                y1 = int(min(img.shape[0], desired_y + baseline + 6))
                cv2.rectangle(img, (x0, y0), (x1, y1), rect_bg_color, -1)
                cv2.putText(img, text, (x, desired_y), font, font_scale, text_color, thickness, cv2.LINE_AA)

            # y positions (safely fetch)
            t_y = int(proportionality.get('trichion_y_px', 0))
            g_y = int(proportionality.get('glabella_y_px', 0))
            s_y = int(proportionality.get('subnasale_y_px', 0))
            m_y = int(proportionality.get('menton_y_px', 0))

            # draw horizontal lines: top & bottom -> yellow, middle two -> green
            if 0 <= t_y < h:
                cv2.line(prop_img, (0, t_y), (w, t_y), yellow, line_thickness, cv2.LINE_AA)
            if 0 <= g_y < h:
                cv2.line(prop_img, (0, g_y), (w, g_y), green, line_thickness, cv2.LINE_AA)
            if 0 <= s_y < h:
                cv2.line(prop_img, (0, s_y), (w, s_y), green, line_thickness, cv2.LINE_AA)
            if 0 <= m_y < h:
                cv2.line(prop_img, (0, m_y), (w, m_y), yellow, line_thickness, cv2.LINE_AA)

            # write labels and measurements near each corresponding line,
            # using the new helper so text appears slightly above the line and not outside image
            try:
                upper_mm = proportionality.get('upper_third_mm')
                middle_mm = proportionality.get('middle_third_mm')
                lower_mm = proportionality.get('lower_third_mm')

                left_x = 10
                right_x = max(10, w - 200)

                if upper_mm is not None and 0 <= t_y < h:
                    put_label_above_line(prop_img, f"Haut: {upper_mm:.1f} mm", t_y, left_x, font, font_scale * 0.6, text_color, 1)
                    put_label_above_line(prop_img, "Trichion", t_y, right_x, font, font_scale * 0.6, text_color, 1)

                if middle_mm is not None and 0 <= g_y < h:
                    put_label_above_line(prop_img, f"Milieu: {middle_mm:.1f} mm", g_y, left_x, font, font_scale * 0.6, text_color, 1)
                    put_label_above_line(prop_img, "Glabella", g_y, right_x, font, font_scale * 0.6, text_color, 1)

                if lower_mm is not None and 0 <= s_y < h:
                    put_label_above_line(prop_img, f"Bas: {lower_mm:.1f} mm", s_y, left_x, font, font_scale * 0.6, text_color, 1)
                    put_label_above_line(prop_img, "Subnasale", s_y, right_x, font, font_scale * 0.6, text_color, 1)

                # Menton (bottom) label: ensure it's inside image and not overlapping face region
                if 0 <= m_y < h:
                    # prefer slightly below the bottom line if there's space; otherwise above
                    (txt_w_m, txt_h_m), baseline_m = cv2.getTextSize("Menton", font, font_scale * 0.8, 1)
                    y_m = m_y + txt_h_m + 12
                    if y_m > h - 6:
                        y_m = max(txt_h_m + 8, m_y - 8)
                    put_bold_text(prop_img, "Menton", (10, int(y_m)), font, max(0.65, font_scale * 0.8), text_color, 1)

                # assessment text at bottom with bold background
                assessment = proportionality.get('proportionality_assessment')
                if assessment:
                    put_bold_text(prop_img, assessment, (10, h - 12), font, max(0.8, font_scale), (230, 230, 230), 2)
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
            'resultImageUrl': landmark_url,
            'landmarks': all_landmarks,
            'clinicalMeasures': clinical_measures,
            'proportionality': proportionality,
            'symmetry': symmetry,
            # keep backward-compatible key
            'proportion_image': proportion_image_url,
            'symmetry_image': symmetry_image_url,   
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
        "download_url": landmark_url,
        "landmarks": landmark_list,
        "clinicalMeasures": clinical_measures,
        "proportionality": proportionality,
        "symmetry": symmetry
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
