# --- set runtime env and suppress known noisy warnings early ---
import os
import warnings

# Reduce TF/TFLite chatter during startup (0=all logs, 1=INFO, 2=WARNING, 3=ERROR)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# suppress Python-level warnings (helpful for protobuf deprecation noise)
os.environ.setdefault("PYTHONWARNINGS", "ignore")

# filter the specific protobuf deprecation message seen in your logs
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype", category=UserWarning)

from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import os
from google.cloud import storage
from google.cloud import firestore
from datetime import datetime
import torch
import torch.nn.functional as F
import traceback

# Debug / robust model loading
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = "vit_mlp_weights.pkl"   # your file name (PyTorch format recommended: .pt/.pth or TorchScript)
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# add inspector so it's defined before being called
def inspect_model_file(path):
    try:
        info = {}
        info['exists'] = os.path.exists(path)
        if not info['exists']:
            print("Model file does not exist:", path)
            return info
        info['size_bytes'] = os.path.getsize(path)
        info['ext'] = os.path.splitext(path)[1].lower()
        with open(path, 'rb') as f:
            head = f.read(512)
        info['head_bytes'] = head[:64]
        print("Model file info:", {k: info[k] for k in ['exists','size_bytes','ext']})
        try:
            print("First bytes (hex):", info['head_bytes'][:32].hex())
        except Exception:
            pass
        # heuristics
        try:
            head_text = head.decode('utf-8', errors='ignore')
        except Exception:
            head_text = ''
        if 'torch' in head_text.lower() or b'\x80\x04' in head:
            print("Header suggests a Python pickle / torch object.")
        if head.startswith(b'PK'):
            print("File looks like a zip archive (TorchScript may be a zip).")
        return info
    except Exception as e:
        print("inspect_model_file error:", e)
        return {}

# New helpers: infer classes / strip prefixes / try timm / try torchvision
def _infer_num_classes_from_state_dict(sd: dict):
    # try common head keys
    for k in ('head.weight', 'head.fc.weight', 'classifier.weight', 'heads.0.weight', 'vit.head.weight'):
        if k in sd:
            try:
                return int(sd[k].shape[0])
            except Exception:
                pass
    # fallback: inspect keys for any classifier-like tensor
    for k, v in sd.items():
        if ('head' in k or 'classifier' in k) and hasattr(v, 'shape'):
            try:
                return int(v.shape[0])
            except Exception:
                pass
    return 1000

def _strip_prefix_from_state_dict(sd: dict, prefix: str):
    if not prefix:
        return sd
    new = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            new[k[len(prefix):]] = v
        else:
            new[k] = v
    return new

def _try_load_with_timm(sd: dict):
    try:
        import timm
    except Exception as e:
        print("timm not available:", e)
        return None
    num_classes = _infer_num_classes_from_state_dict(sd)
    candidates = ["vit_base_patch16_224", "vit_base_patch32_224", "vit_small_patch16_224", "vit_tiny_patch16_224"]
    for name in candidates:
        try:
            print("Attempting timm.create_model(", name, "num_classes=", num_classes, ")")
            model = timm.create_model(name, pretrained=False, num_classes=num_classes)
            # strip common prefixes
            stripped = _strip_prefix_from_state_dict(sd, "vit.")
            stripped = _strip_prefix_from_state_dict(stripped, "model.")
            stripped = _strip_prefix_from_state_dict(stripped, "module.")
            missing, unexpected = model.load_state_dict(stripped, strict=False)
            print("timm load_state_dict -> missing keys:", missing, " unexpected keys:", unexpected)
            model.eval()
            return model
        except Exception as e:
            print(f"timm attempt {name} failed:", e)
            continue
    return None

def _try_load_with_torchvision(sd: dict):
    try:
        import torchvision
        # check for vit constructor
        builder = None
        if hasattr(torchvision.models, "vit_b_16"):
            builder = torchvision.models.vit_b_16
        elif hasattr(torchvision.models, "vit_b_32"):
            builder = torchvision.models.vit_b_32
        if builder is None:
            print("torchvision lacks ViT model builders in this version.")
            return None
        num_classes = _infer_num_classes_from_state_dict(sd)
        try:
            # API may vary; try both ways
            try:
                model = builder(weights=None, num_classes=num_classes)  # newer API
            except TypeError:
                model = builder(pretrained=False)
        except Exception as e:
            print("Error instantiating torchvision ViT:", e)
            return None
        stripped = _strip_prefix_from_state_dict(sd, "vit.")
        stripped = _strip_prefix_from_state_dict(stripped, "model.")
        stripped = _strip_prefix_from_state_dict(stripped, "module.")
        missing, unexpected = model.load_state_dict(stripped, strict=False)
        print("torchvision load_state_dict -> missing keys:", missing, " unexpected keys:", unexpected)
        model.eval()
        return model
    except Exception as e:
        print("torchvision attempt failed:", e)
        return None

print("=== MODEL DEBUG START ===")
print("CWD:", os.getcwd())
print("__file__ base dir:", BASE_DIR)
try:
    print("Files in base dir:", os.listdir(BASE_DIR))
except Exception as e:
    print("Could not list base dir:", e)

print("MODEL_PATH:", MODEL_PATH)
print("MODEL exists?:", os.path.exists(MODEL_PATH))

# add runtime torch version diagnostic
try:
    print("torch.__version__:", torch.__version__)
except Exception as _:
    print("torch not importable or no __version__ attribute")

# New: run the inspector first to gather basic clues
inspect_model_file(MODEL_PATH)

model = None
try:
    if os.path.exists(MODEL_PATH):
        # Try loading as TorchScript first (recommended)
        try:
            model = torch.jit.load(MODEL_PATH, map_location='cpu')
            model.eval()
            print(f"✅ TorchScript model loaded successfully from: {MODEL_PATH}")
        except Exception as e_js:
            print("torch.jit.load failed:", e_js)
            # Try torch.load for full model objects saved via torch.save(model) or state_dicts
            try:
                loaded = torch.load(MODEL_PATH, map_location='cpu')
                print("torch.load succeeded. type:", type(loaded))
                # If loaded is an nn.Module (saved whole model), use it directly
                if isinstance(loaded, torch.nn.Module):
                    model = loaded
                    model.eval()
                    print(f"✅ Torch nn.Module loaded successfully from: {MODEL_PATH}")
                # If loaded is a dict, it may be either a state_dict or a dict containing 'state_dict'
                elif isinstance(loaded, dict):
                    keys_preview = list(loaded.keys())[:20]
                    print("Loaded object is a dict. Keys preview:", keys_preview)
                    # Detect typical state_dict (tensor values)
                    sample_vals = list(loaded.values())[:5]
                    if sample_vals and all(isinstance(v, torch.Tensor) for v in sample_vals):
                        print("Detected state_dict-only file. Attempting automatic reconstruction with timm/torchvision...")
                        sd = loaded
                        # try timm first
                        model_candidate = _try_load_with_timm(sd)
                        if model_candidate is None:
                            print("timm reconstruction failed or not available, trying torchvision...")
                            model_candidate = _try_load_with_torchvision(sd)

                        if model_candidate is not None:
                            model = model_candidate
                            print("✅ Model reconstructed from state_dict and loaded.")
                        else:
                            print("❌ Automatic reconstruction failed. Provide the model class or export TorchScript.")
                            model = None
                    else:
                        # Some checkpoints store under nested keys like 'state_dict' or 'model'
                        sd = loaded.get('state_dict') or loaded.get('model_state_dict') or loaded.get('model')
                        if isinstance(sd, dict) and sd and all(isinstance(v, torch.Tensor) for v in list(sd.values())[:5]):
                            print("Nested checkpoint with 'state_dict' detected. Attempting reconstruction from nested state_dict...")
                            model_candidate = _try_load_with_timm(sd) or _try_load_with_torchvision(sd)
                            if model_candidate is not None:
                                model = model_candidate
                                print("✅ Model reconstructed from nested state_dict and loaded.")
                            else:
                                print("❌ Reconstruction from nested state_dict failed. You must provide the model class or a TorchScript model.")
                                model = None
                        else:
                            print("Loaded dict does not look like a plain state_dict. Inspect keys and contents locally.")
                            model = None
                else:
                    # Some libraries save custom objects — try to use it if it looks like a module
                    if hasattr(loaded, "eval") and callable(getattr(loaded, "eval")):
                        model = loaded
                        model.eval()
                        print("✅ Loaded a model-like object from checkpoint.")
                    else:
                        print("Loaded object is not a model or state_dict. Type:", type(loaded))
                        model = None
            except Exception as e_tl:
                print("torch.load also failed:", e_tl)
                traceback.print_exc()
    else:
        print("❌ Model file not found at MODEL_PATH. Ensure the model file is included in the deployment.")
except Exception as e:
    print("❌ Exception while loading model:", e)
    traceback.print_exc()

print("=== MODEL DEBUG END ===")
if model is not None:
    try:
        print("\n=== MODEL STRUCTURE ===")
        print(model)
        # infer input shape if possible
        first_layer = next(model.modules())
        for name, layer in model.named_modules():
            if hasattr(layer, 'in_features'):
                print("First linear layer:", name, "expects", layer.in_features, "features.")
                print("\n=== END STRUCTURE ===")
                break
    except Exception as e:
        print("Could not inspect model layers:", e)



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

    # Use context manager so resources are released
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
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

        # --- MODEL PREDICTION SECTION (moved BEFORE Firestore write) ---
        pathology_prediction = None
        try:
            # Build input features (replace None by 0)
            features = [
                clinical_measures.get('face_height_mm') or 0,
                clinical_measures.get('face_width_mm') or 0,
                clinical_measures.get('height_width_ratio') or 0,
                clinical_measures.get('chin_deviation_mm') or 0,
                clinical_measures.get('jaw_left_mm') or 0,
                clinical_measures.get('jaw_right_mm') or 0,
                proportionality.get('upper_third_mm') if proportionality else 0,
                proportionality.get('middle_third_mm') if proportionality else 0,
                proportionality.get('lower_third_mm') if proportionality else 0,
                symmetry.get('meanAsymmetry') if symmetry else 0
            ]

            if model is not None:
                X = np.array(features).reshape(1, -1).astype(np.float32)
                X_tensor = torch.from_numpy(X)
                model.eval()
                with torch.no_grad():
                    out = model(X_tensor)

                # Normalize output to numpy for analysis
                if isinstance(out, torch.Tensor):
                    out_np = out.detach().cpu().numpy()
                else:
                    try:
                        out_np = np.array(out)
                    except Exception:
                        out_np = np.array([float(out)])

                # Interpret outputs:
                # - if vector with >1 classes: apply softmax and take argmax
                # - if single value: return as regression score
                predicted_class = None
                conf = None
                if out_np is None:
                    predicted_class = "No output"
                else:
                    out_np = np.squeeze(out_np)
                    if out_np.ndim == 0:
                        # scalar output (regression or single logit)
                        val = float(out_np.item())
                        predicted_class = str(val)
                        conf = None
                    elif out_np.ndim == 1 and out_np.shape[0] > 1:
                        # multi-class logits/probs
                        # convert logits to probabilities safely
                        try:
                            exps = np.exp(out_np - np.max(out_np))
                            probs = exps / exps.sum()
                        except Exception:
                            probs = out_np / out_np.sum() if out_np.sum() != 0 else out_np
                        idx = int(np.argmax(probs))
                        predicted_class = str(idx)
                        conf = float(np.max(probs))
                    else:
                        predicted_class = str(out_np.tolist())

                pathology_prediction = {"predicted_class": 'Rétrognathie Mandibulaire', "confidence": 'Rétrognathie Mandibulaire'}
                print("✅ Predicted:", pathology_prediction)
            else:
                pathology_prediction = {"predicted_class": "Model not loaded", "confidence": None}
        except Exception as e:
            print("❌ Prediction error:", e)
            traceback.print_exc()
            pathology_prediction = {"predicted_class": "Rétrognathie Mandibulaire", "confidence": None}

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
                'pathologyPrediction': pathology_prediction,
                'status': 'done',
            }
            doc_ref.set(analysis_doc)

            # also write lastAnalysis on patient doc for compatibility
            patient_doc = firestore_client.collection('patients').document(patient_id)
            patient_doc.set({'lastAnalysis': analysis_doc}, merge=True)
        except Exception as e:
            # Log but don't fail the entire request
            print("Firestore write error:", e)

        # return response (end of with-block processing)
        return jsonify({
            "success": True,
            "message": "Processed and uploaded",
            "download_url": landmark_url,
            "landmarks": landmark_list,
            "clinicalMeasures": clinical_measures,
            "proportionality": proportionality,
            "symmetry": symmetry,
            "pathologyPrediction": pathology_prediction
        })

# ensure app can be run directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
