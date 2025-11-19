from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import timm
import os
import base64
import io
from pyngrok import ngrok

app = Flask(__name__)
CORS(app)

# ============= CONFIG =============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_medagen_resnet18_vits_cbam.pth"
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.40
NGROK_TOKEN = "35W4RdCdRA10R0NSBN6Vzxyr379_2vceBj14fqSEMKQHRDe4B"

# ============= CBAM MODULE =============
class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                                 padding=spatial_kernel // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        ch_att = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(b, c, 1, 1)
        x = x * ch_att

        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)
        sp_att = torch.sigmoid(self.spatial(s))
        x = x * sp_att
        return x

# ============= FUSION MODEL =============
class ResNet18_ViTS_CBAM(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        rn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet_backbone = nn.Sequential(*list(rn.children())[:-1])
        res_dim = 512

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
        r = self.resnet_backbone(x)
        r = r.view(r.size(0), -1)
        v = self.vit(x)
        feat = torch.cat([r, v], dim=1)
        feat_4d = feat.unsqueeze(-1).unsqueeze(-1)
        feat_4d = self.cbam(feat_4d)
        feat = feat_4d.view(feat_4d.size(0), -1)
        out = self.classifier(feat)
        return out

# ============= LOAD MODEL =============
def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    classes = checkpoint["classes"]
    model = ResNet18_ViTS_CBAM(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(DEVICE)
    model.eval()
    return model, classes

# ============= TRANSFORMS =============
transforms_inference = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ============= PREDICTION FUNCTION =============
def predict_from_image(image, model, class_names, confidence_threshold=0.40):
    image_tensor = transforms_inference(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        max_prob, max_idx = torch.max(probabilities, dim=0)
        max_confidence = float(max_prob)
        
        if max_confidence < confidence_threshold:
            return {
                "status": "out_of_domain",
                "message": f"Out of domain (confidence: {max_confidence*100:.2f}% < {confidence_threshold*100:.0f}%)",
                "max_confidence": max_confidence,
                "threshold": confidence_threshold
            }
        
        top3 = torch.topk(probabilities, k=min(3, len(class_names)))
        predictions = []
        for i in range(top3.indices.size(0)):
            idx = top3.indices[i].item()
            prob = float(top3.values[i])
            predictions.append({
                "rank": i + 1,
                "class": class_names[idx],
                "confidence": prob,
                "confidence_percent": prob * 100
            })
        
        return {
            "status": "success",
            "predictions": predictions,
            "max_confidence": max_confidence,
            "threshold": confidence_threshold
        }

# ============= FLASK ROUTES =============
@app.route('/', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "DermNet AI API is running",
        "model_loaded": model is not None,
        "device": DEVICE,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({"error": "No image provided"}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No image selected"}), 400
        
        # Load and process image
        image = Image.open(file.stream).convert('RGB')
        
        # Make prediction
        result = predict_from_image(image, model, class_names, CONFIDENCE_THRESHOLD)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({"error": "No base64 image provided"}), 400
        
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Make prediction
        result = predict_from_image(image, model, class_names, CONFIDENCE_THRESHOLD)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({
        "classes": class_names,
        "total_classes": len(class_names)
    })

if __name__ == '__main__':
    print("🔥 Loading model...")
    model, class_names = load_model()
    print(f"✅ Model loaded! Classes: {len(class_names)}")
    
    print("🌐 Setting up ngrok...")
    ngrok.set_auth_token(NGROK_TOKEN)
    public_url = ngrok.connect(5000)
    print(f"🚀 Public URL: {public_url}")
    
    print("🎯 API Endpoints:")
    print(f"  - Health: {public_url}")
    print(f"  - Predict: {public_url}/predict (POST with image file)")
    print(f"  - Predict Base64: {public_url}/predict_base64 (POST with JSON)")
    print(f"  - Classes: {public_url}/classes")
    
    app.run(host='0.0.0.0', port=5000, debug=False)