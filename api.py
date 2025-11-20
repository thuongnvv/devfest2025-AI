"""
Multi-Model Healthcare AI API
API endpoints for skin, teeth, and nail disease detection
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import timm
import numpy as np
import base64
import io
import requests
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224

# Model configurations
MODEL_CONFIGS = {
    "dermnet": {
        "path": "best_medagen_resnet18_vits_cbam.pth",
        "description": "Skin disease detection using ResNet18 + ViT + CBAM",
        "architecture": "ResNet18 + Vision Transformer + CBAM Attention",
        "threshold": 0.40,
        "classes": [
            "Acne and Rosacea Photos",
            "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
            "Atopic Dermatitis Photos",
            "Cellulitis Impetigo and other Bacterial Infections",
            "Eczema Photos",
            "Hair Loss Photos Alopecia and other Hair Diseases",
            "Melanoma Skin Cancer Nevi and Moles",
            "Nail Fungus and other Nail Disease",
            "Poison Ivy Photos and other Contact Dermatitis",
            "Psoriasis pictures Lichen Planus and related diseases",
            "Scabies Lyme Disease and other Infestations and Bites",
            "Seborrheic Keratoses and other Benign Tumors",
            "Tinea Ringworm Candidiasis and other Fungal Infections",
            "Warts Molluscum and other Viral Infections"
        ]
    },
    "teeth": {
        "path": "best_teeth_model.pth",
        "description": "Teeth disease detection using MobileNetV2",
        "architecture": "MobileNetV2",
        "threshold": 0.60,
        "classes": [
            "Calculus",
            "Mouth Ulcer", 
            "Tooth Discoloration",
            "caries",
            "hypodontia"
        ]
    },
    "nail": {
        "path": "best_nail_model.pth",
        "description": "Nail disease detection using MobileNetV2", 
        "architecture": "MobileNetV2",
        "threshold": 0.70,
        "classes": [
            "Acral_Lentiginous_Melanoma",
            "Healthy_Nail",
            "Onychogryphosis",
            "blue_finger",
            "clubbing",
            "pitting"
        ]
    }
}

# Global variable to store loaded models
loaded_models = {}

# Image preprocessing
image_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============= MODEL CLASSES =============

class CBAM(nn.Module):
    """CBAM Attention Module for DermNet"""
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 3):
        super().__init__()
        # Channel attention
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        # Spatial attention
        self.spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                 padding=spatial_kernel // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()

        # Channel attention
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        ch_att = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(b, c, 1, 1)
        x = x * ch_att

        # Spatial attention
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)
        sp_att = torch.sigmoid(self.spatial(s))
        x = x * sp_att
        return x

class ResNet18_ViTS_CBAM(nn.Module):
    """DermNet Model: ResNet18 + ViT + CBAM"""
    def __init__(self, num_classes: int):
        super().__init__()

        # ResNet18 backbone
        rn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet_backbone = nn.Sequential(*list(rn.children())[:-1])
        res_dim = 512

        # ViT small
        self.vit = timm.create_model("vit_small_patch16_224", pretrained=True)
        vit_dim = self.vit.embed_dim
        if hasattr(self.vit, "head"):
            self.vit.reset_classifier(0)

        fused_dim = res_dim + vit_dim
        self.cbam = CBAM(fused_dim, reduction=16, spatial_kernel=3)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ResNet branch
        r = self.resnet_backbone(x)
        r = r.view(r.size(0), -1)

        # ViT branch
        v = self.vit(x)

        # Fusion + CBAM
        feat = torch.cat([r, v], dim=1)
        feat_4d = feat.unsqueeze(-1).unsqueeze(-1)
        feat_4d = self.cbam(feat_4d)
        feat = feat_4d.view(feat_4d.size(0), -1)
        out = self.classifier(feat)
        return out

def create_mobilenet_model(num_classes: int):
    """Create MobileNetV2 model for teeth/nail"""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model

# ============= MODEL LOADING =============

def load_model(model_name: str):
    """Load specific model"""
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    
    config = MODEL_CONFIGS[model_name]
    model_path = config["path"]
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    try:
        logger.info(f"Loading {model_name} model from {model_path}")
        
        # Create model based on type
        if model_name == "dermnet":
            model = ResNet18_ViTS_CBAM(num_classes=len(config["classes"]))
        else:  # teeth or nail
            model = create_mobilenet_model(num_classes=len(config["classes"]))
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state' in checkpoint:
                model.load_state_dict(checkpoint['model_state'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(DEVICE)
        model.eval()
        
        # Cache model
        loaded_models[model_name] = model
        
        logger.info(f"Successfully loaded {model_name} model")
        return model
        
    except Exception as e:
        logger.error(f"Error loading {model_name} model: {str(e)}")
        raise

# ============= UTILITY FUNCTIONS =============

def download_image_from_url(image_url: str) -> Image.Image:
    """Download image from URL"""
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Error downloading image from URL: {str(e)}")

def decode_base64_image(base64_string: str) -> Image.Image:
    """Decode base64 image"""
    try:
        # Remove data URL prefix if present
        if base64_string.startswith('data:image'):
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Error decoding base64 image: {str(e)}")

def predict_with_model(model, image: Image.Image, model_name: str, top_n: int = 5):
    """Make prediction with model"""
    try:
        # Preprocess image
        input_tensor = image_transforms(image).unsqueeze(0).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
        
        # Get results
        probs = probabilities[0].cpu().numpy()
        classes = MODEL_CONFIGS[model_name]["classes"]
        threshold = MODEL_CONFIGS[model_name]["threshold"]
        
        # Get top N predictions
        top_indices = np.argsort(probs)[-top_n:][::-1]
        top_predictions = [
            {
                "class": classes[idx],
                "confidence": float(probs[idx]),
                "rank": i + 1
            }
            for i, idx in enumerate(top_indices)
        ]
        
        # Top prediction
        top_prediction = top_predictions[0]
        max_confidence = top_prediction["confidence"]
        
        # Out of domain detection
        is_out_of_domain = max_confidence < threshold
        
        return {
            "model": model_name,
            "top_prediction": top_prediction["class"],
            "confidence": max_confidence,
            "is_out_of_domain": is_out_of_domain,
            "threshold": threshold,
            "top_predictions": top_predictions,
            "logits_range": {
                "min": float(logits.min()),
                "max": float(logits.max())
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction error for {model_name}: {str(e)}")
        raise

# ============= API ENDPOINTS =============

@app.route("/", methods=["GET"])
def home():
    """API home page"""
    return jsonify({
        "message": "Multi-Model Healthcare AI API",
        "version": "1.0.0",
        "models": list(MODEL_CONFIGS.keys()),
        "endpoints": [
            "/health",
            "/models", 
            "/models/{model_name}/classes",
            "/predict"
        ],
        "documentation": "https://github.com/thuongnvv/devfest2025-AI"
    })

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    try:
        # Check if model files exist
        model_status = {}
        for model_name, config in MODEL_CONFIGS.items():
            model_status[model_name] = {
                "file_exists": os.path.exists(config["path"]),
                "loaded": model_name in loaded_models
            }
        
        return jsonify({
            "status": "healthy",
            "device": DEVICE,
            "models_status": model_status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            "status": "error", 
            "error": str(e)
        }), 500

@app.route("/models", methods=["GET"])
def list_models():
    """List all available models"""
    models_info = {}
    for model_name, config in MODEL_CONFIGS.items():
        models_info[model_name] = {
            "description": config["description"],
            "architecture": config["architecture"],
            "num_classes": len(config["classes"]),
            "confidence_threshold": config["threshold"],
            "file_exists": os.path.exists(config["path"])
        }
    
    return jsonify({
        "models": models_info,
        "total_models": len(MODEL_CONFIGS)
    })

@app.route("/models/<model_name>/classes", methods=["GET"])
def list_model_classes(model_name):
    """List classes for specific model"""
    if model_name not in MODEL_CONFIGS:
        return jsonify({"error": f"Model '{model_name}' not found"}), 404
    
    config = MODEL_CONFIGS[model_name]
    return jsonify({
        "model": model_name,
        "description": config["description"], 
        "architecture": config["architecture"],
        "classes": config["classes"],
        "num_classes": len(config["classes"]),
        "confidence_threshold": config["threshold"]
    })

@app.route("/predict", methods=["POST"])
def predict():
    """Make prediction on image"""
    try:
        data = request.get_json()
        
        # Validate request
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        model_name = data.get("model")
        if not model_name:
            return jsonify({"error": "Model name is required"}), 400
            
        if model_name not in MODEL_CONFIGS:
            return jsonify({"error": f"Unknown model: {model_name}"}), 400
        
        # Get top N (default 5)
        top_n = min(data.get("n", 5), len(MODEL_CONFIGS[model_name]["classes"]))
        
        # Get image
        image = None
        if "image_url" in data:
            image = download_image_from_url(data["image_url"])
        elif "image" in data:
            image = decode_base64_image(data["image"])
        else:
            return jsonify({"error": "Either 'image_url' or 'image' (base64) is required"}), 400
        
        # Load model
        model = load_model(model_name)
        
        # Make prediction
        result = predict_with_model(model, image, model_name, top_n)
        
        return jsonify({
            "success": True,
            "result": result
        })
        
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    print("🚀 Starting Multi-Model Healthcare AI API")
    print(f"Device: {DEVICE}")
    print(f"Available models: {list(MODEL_CONFIGS.keys())}")
    
    # Pre-load models (optional)
    # for model_name in MODEL_CONFIGS.keys():
    #     try:
    #         load_model(model_name)
    #         print(f"✅ Pre-loaded {model_name}")
    #     except Exception as e:
    #         print(f"❌ Failed to pre-load {model_name}: {e}")
    
    app.run(host="0.0.0.0", port=5000, debug=False)