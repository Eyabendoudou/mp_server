# ==========================================
# app.py - Flask API adapté pour ViT+MLP
# Railway deployment ready
# ==========================================

# --- Suppress warnings early ---
import os
import warnings
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype", category=UserWarning)

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import timm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import StandardScaler
import traceback

# ========================================
# CONFIGURATION
# ========================================
PORT = int(os.environ.get('PORT', 5000))

# ⚠ IMPORTANT: Ajustez ces valeurs selon votre modèle
TABULAR_INPUT_DIM = 12  # Nombre de features tabulaires dans votre modèle
NUM_CLASSES = 2         # Nombre de classes (0=sain, 1=pathologie)
VIEWS = ['front', 'right', 'left']  # Vues d'images requises

# ========================================
# MODÈLE ViT+MLP FUSION (copie exacte de votre code)
# ========================================
class ViT_MLP_Fusion(nn.Module):
    def _init_(self, tabular_input_dim, num_classes=2, views=['front','left','right']):
        super()._init_()
        # 1️⃣ Load pretrained ViT
        self.vit = timm.create_model('vit_tiny_patch16_224', pretrained=False)
        self.vit.reset_classifier(0)  # remove head
        self.views = views

        # 2️⃣ MLP for tabular data
        self.mlp_tab = nn.Sequential(
            nn.Linear(tabular_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # 3️⃣ Get ViT output dimension dynamically
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            vit_feat = self.vit(dummy)
        self.vit_out_dim = vit_feat.shape[1]

        # 4️⃣ Fusion classifier
        fusion_input_dim = self.vit_out_dim * len(self.views) + 64
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, images, tabular):
        """
        images: (B, V, C, H, W) ou (B, C, H, W)
        tabular: (B, tabular_input_dim)
        """
        if images.ndim == 4:
            images = images.unsqueeze(1)
        B, V, C, H, W = images.shape

        img_feats = []
        for v in range(V):
            feat = self.vit(images[:, v, :, :, :])
            img_feats.append(feat)
        img_feats = torch.cat(img_feats, dim=1)  # (B, vit_out_dim * V)

        tab_feats = self.mlp_tab(tabular)
        x = torch.cat([img_feats, tab_feats], dim=1)
        return self.fusion(x)

# ========================================
# CHARGEMENT DU MODÈLE
# ========================================
print("="*50)
print("🔄 Initialisation du modèle ViT+MLP...")
print("="*50)

BASE_DIR = os.path.dirname(os.path.abspath(_file_))
MODEL_PATH = os.path.join(BASE_DIR, "best_vit_mlp.pt")

device = torch.device('cpu')  # Railway utilise CPU
model = None

try:
    if os.path.exists(MODEL_PATH):
        print(f"✅ Fichier modèle trouvé: {MODEL_PATH}")
        print(f"   Taille: {os.path.getsize(MODEL_PATH) / (1024**2):.2f} MB")
        
        # Initialiser le modèle
        model = ViT_MLP_Fusion(
            tabular_input_dim=TABULAR_INPUT_DIM,
            num_classes=NUM_CLASSES,
            views=VIEWS
        )
        
        # Charger les poids
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(device)
        
        print("✅ Modèle ViT+MLP chargé avec succès!")
        print(f"   Architecture: ViT-Tiny + MLP Fusion")
        print(f"   Input: {len(VIEWS)} images + {TABULAR_INPUT_DIM} features")
        print(f"   Output: {NUM_CLASSES} classes")
    else:
        print(f"❌ Fichier modèle introuvable: {MODEL_PATH}")
        print("   Vérifiez que 'best_vit_mlp.pt' est dans le dossier")
        
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle: {e}")
    traceback.print_exc()

print("="*50)

# ========================================
# PREPROCESSING IMAGES (Albumentations)
# ========================================
transform = A.Compose([
    A.Resize(224, 224),
    ToTensorV2()
])

def preprocess_image(image_bytes):
    """Convertit bytes → tensor PyTorch (3, 224, 224)"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(img)
        transformed = transform(image=img_array)['image']
        return transformed.float() / 255.0
    except Exception as e:
        print(f"Erreur preprocessing image: {e}")
        return None

# ========================================
# FLASK APP
# ========================================
app = Flask(_name_)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'ok',
        'message': 'ViT+MLP Medical AI API',
        'model': 'ViT_MLP_Fusion',
        'model_loaded': model is not None,
        'endpoints': {
            'health': '/health (GET)',
            'predict': '/predict (POST - multimodal)',
            'predict_from_measures': '/predict_from_measures (POST - tabular only)'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'tabular_input_dim': TABULAR_INPUT_DIM,
        'num_classes': NUM_CLASSES,
        'views_required': VIEWS
    })

# ========================================
# ROUTE 1: Prédiction multimodale (images + tabular)
# ========================================
@app.route('/predict', methods=['POST'])
def predict():
    """
    Prédiction avec images + données tabulaires
    
    Requête:
    - front, right, left: images (fichiers)
    - tabular: string "val1,val2,val3,..." (TABULAR_INPUT_DIM valeurs)
    
    Réponse:
    - prediction: classe prédite (0 ou 1)
    - confidence: probabilité de la classe prédite
    - probabilities: toutes les probabilités
    """
    if model is None:
        return jsonify({'error': 'Modèle non chargé'}), 500
    
    try:
        # 1️⃣ Vérifier les images
        images = []
        for view in VIEWS:
            if view not in request.files:
                return jsonify({'error': f'Image {view} manquante'}), 400
            
            img_file = request.files[view]
            img_tensor = preprocess_image(img_file.read())
            
            if img_tensor is None:
                return jsonify({'error': f'Image {view} invalide'}), 400
            
            images.append(img_tensor)
        
        # 2️⃣ Récupérer les données tabulaires
        tabular_str = request.form.get('tabular', '')
        if not tabular_str:
            return jsonify({'error': 'Données tabulaires manquantes (paramètre: tabular)'}), 400
        
        tabular = np.array([float(x) for x in tabular_str.split(',')], dtype=np.float32)
        
        if len(tabular) != TABULAR_INPUT_DIM:
            return jsonify({
                'error': f'Nombre de features incorrect',
                'expected': TABULAR_INPUT_DIM,
                'received': len(tabular)
            }), 400
        
        # 3️⃣ Préparer les tensors
        images_tensor = torch.stack(images).unsqueeze(0).to(device)  # (1, 3, 3, 224, 224)
        tabular_tensor = torch.tensor(tabular).unsqueeze(0).to(device)  # (1, TABULAR_INPUT_DIM)
        
        # 4️⃣ Prédiction
        with torch.no_grad():
            outputs = model(images_tensor, tabular_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = outputs.argmax(1).item()
            confidence = probabilities[0][prediction].item()
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'confidence': float(confidence),
            'probabilities': probabilities[0].tolist(),
            'class_name': 'Pathologie' if prediction == 1 else 'Sain'
        })
    
    except Exception as e:
        print(f"Erreur prediction: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ========================================
# ROUTE 2: Prédiction avec mesures cliniques uniquement
# ========================================
@app.route('/predict_from_measures', methods=['POST'])
def predict_from_measures():
    """
    Prédiction avec uniquement les mesures cliniques (sans images)
    
    Requête JSON:
    {
        "face_height_mm": 180.5,
        "face_width_mm": 130.2,
        "height_width_ratio": 1.38,
        "chin_deviation_mm": 2.5,
        "jaw_left_mm": 95.3,
        "jaw_right_mm": 94.8,
        "upper_third_mm": 60.1,
        "middle_third_mm": 59.8,
        "lower_third_mm": 60.5,
        "avg_asymmetry_mm": 3.2
    }
    
    Note: Si vous n'avez pas d'images, créez des images noires (placeholder)
    """
    if model is None:
        return jsonify({'error': 'Modèle non chargé'}), 500
    
    try:
        data = request.get_json()
        
        # Extraire les features dans le bon ordre
        features = [
            data.get('face_height_mm', 0),
            data.get('face_width_mm', 0),
            data.get('height_width_ratio', 0),
            data.get('chin_deviation_mm', 0),
            data.get('jaw_left_mm', 0),
            data.get('jaw_right_mm', 0),
            data.get('upper_third_mm', 0),
            data.get('middle_third_mm', 0),
            data.get('lower_third_mm', 0),
            data.get('avg_asymmetry_mm', 0)
        ]
        
        if len(features) != TABULAR_INPUT_DIM:
            return jsonify({
                'error': f'Nombre de features incorrect',
                'expected': TABULAR_INPUT_DIM,
                'received': len(features)
            }), 400
        
        # Créer des images placeholder (noires)
        dummy_images = torch.zeros(1, 3, 3, 224, 224).to(device)
        
        # Préparer les features tabulaires
        tabular_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        
        # Prédiction
        with torch.no_grad():
            outputs = model(dummy_images, tabular_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = outputs.argmax(1).item()
            confidence = probabilities[0][prediction].item()
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'confidence': float(confidence),
            'probabilities': probabilities[0].tolist(),
            'class_name': 'Pathologie' if prediction == 1 else 'Sain',
            'note': 'Prédiction basée uniquement sur les mesures cliniques'
        })
    
    except Exception as e:
        print(f"Erreur prediction from measures: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# ========================================
# LANCER L'APPLICATION
# ========================================
if _name_ == '_main_':
    print("\n" + "="*50)
    print(f"🚀 Démarrage du serveur sur le port {PORT}")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=PORT, debug=False)