import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import timm
import io
import json
import requests
from io import BytesIO

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224

# Model-specific confidence thresholds (from verified API)
CONFIDENCE_THRESHOLDS = {
    "dermnet": 0.40,  # 40% threshold for skin diseases
    "teeth": 0.60,    # 60% threshold for teeth  
    "nail": 0.70      # 70% threshold for nails
}

# Updated model configurations (exact paths from API)
MODEL_CONFIGS = {
    "dermnet": {
        "path": "best_medagen_swin_convnext_cbam_23classes.pth",
        "description": "Skin disease detection using Swin Tiny + ConvNeXt + CBAM",
        "architecture": "Swin Transformer + ConvNeXt + CBAM Fusion",
        "type": "vit_cnn_hybrid",
        "classes": [
            "Acne and Rosacea Photos",
            "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
            "Atopic Dermatitis Photos",
            "Bullous Disease Photos",
            "Cellulitis Impetigo and other Bacterial Infections",
            "Eczema Photos",
            "Exanthems and Drug Eruptions",
            "Hair Loss Photos Alopecia and other Hair Diseases",
            "Herpes HPV and other STDs Photos",
            "Light Diseases and Disorders of Pigmentation",
            "Lupus and other Connective Tissue diseases",
            "Melanoma Skin Cancer Nevi and Moles",
            "Nail Fungus and other Nail Disease",
            "Poison Ivy Photos and other Contact Dermatitis",
            "Psoriasis pictures Lichen Planus and related diseases",
            "Scabies Lyme Disease and other Infestations and Bites",
            "Seborrheic Keratoses and other Benign Tumors",
            "Systemic Disease",
            "Tinea Ringworm Candidiasis and other Fungal Infections",
            "Urticaria Hives",
            "Vascular Tumors",
            "Vasculitis Photos",
            "Warts Molluscum and other Viral Infections"
        ]
    },
    "teeth": {
        "path": "resnet18_cbam_teeth_best.pth",
        "description": "Teeth disease detection using ResNet18 + CBAM",
        "architecture": "ResNet18 + CBAM Attention",
        "type": "cbam_resnet18",
        "classes": [
            "Calculus",
            "Mouth Ulcer",
            "Tooth Discoloration", 
            "caries",
            "hypodontia",
            "Unknown"
        ]
    },
    "nail": {
        "path": "resnet18_cbam_nail_best.pth", 
        "description": "Nail disease detection using ResNet18 + CBAM",
        "architecture": "ResNet18 + CBAM Attention",
        "type": "cbam_resnet18",
        "classes": [
            "Acral_Lentiginous_Melanoma",
            "Healthy_Nail",
            "Onychogryphosis",
            "blue_finger",
            "clubbing",
            "pitting",
            "Unknown"
        ]
    }
}



# Global variable to store loaded models
loaded_models = {}

# Image transformations
transforms_inference = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# CBAM Module (EXACT từ training)
class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
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

        # ----- Channel attention -----
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c)
        ch_att = torch.sigmoid(self.mlp(avg_pool) + self.mlp(max_pool)).view(b, c, 1, 1)
        x = x * ch_att

        # ----- Spatial attention -----
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        s = torch.cat([avg, mx], dim=1)   # [B, 2, H, W]
        sp_att = torch.sigmoid(self.spatial(s))
        x = x * sp_att
        return x

# CBAMBlock for ViTCNNHybrid (from DermNet 23 classes training)
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAMBlock, self).__init__()
        # Channel attention
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # Spatial attention
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_att(x)
        x = x * ca
        # Spatial attention
        sa = torch.cat([
            torch.mean(x, dim=1, keepdim=True), 
            torch.max(x, dim=1, keepdim=True)[0]
        ], dim=1)
        sa = self.spatial_att(sa)
        x = x * sa
        return x

# ViTCNNHybrid Model (Swin Tiny + ConvNeXt Tiny + CBAM) for DermNet 23 classes
class ViTCNNHybrid(nn.Module):
    def __init__(self, num_classes, use_cbam=True):
        super(ViTCNNHybrid, self).__init__()
        
        self.vit = timm.create_model(
            'swin_tiny_patch4_window7_224', pretrained=True, num_classes=0, drop_rate=0.3
        )
        self.vit_out_features = 768
        
        # ConvNeXt-Tiny
        self.cnn = timm.create_model(
            'convnext_tiny', pretrained=True, num_classes=0, drop_rate=0.3, global_pool=''
        )
        self.cnn_out_features = 768
        self.cnn_pool = nn.AdaptiveAvgPool2d((7, 7))
        
        # Gates
        self.vit_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.vit_out_features, self.vit_out_features // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.vit_out_features // 16, self.vit_out_features, 1),
            nn.Sigmoid()
        )
        self.cnn_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.cnn_out_features, self.cnn_out_features // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.cnn_out_features // 16, self.cnn_out_features, 1),
            nn.Sigmoid()
        )
        
        self.match_dim = nn.Conv2d(self.vit_out_features, self.cnn_out_features, 1)
        
        # Learnable α for dynamic fusion
        self.alpha_param = nn.Parameter(torch.tensor(0.5))
        
        # Fusion
        fusion_layers = [
            nn.Conv2d(self.cnn_out_features, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        ]
        if use_cbam:
            fusion_layers.append(CBAMBlock(256))
        fusion_layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.fusion = nn.Sequential(*fusion_layers)
        
        # FC
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # ViT branch
        vit_out = self.vit(x)
        vit_out = vit_out.view(-1, self.vit_out_features, 1, 1).expand(-1, -1, 7, 7)
        vit_out = vit_out * self.vit_gate(vit_out)
        
        # CNN branch
        cnn_out = self.cnn(x)
        cnn_out = self.cnn_pool(cnn_out)
        cnn_out = cnn_out * self.cnn_gate(cnn_out)
        
        # Dynamic Fusion
        alpha = torch.sigmoid(self.alpha_param)
        combined = alpha * vit_out + (1 - alpha) * cnn_out
        
        combined = self.fusion(combined)
        combined = combined.view(combined.size(0), -1)
        out = self.fc(combined)
        return out

# ResNet18_ViTS_CBAM Model (EXACT from training notebook)
class ResNet18_ViTS_CBAM(nn.Module):
    def __init__(self, num_classes=14):
        super(ResNet18_ViTS_CBAM, self).__init__()
        
        # ResNet18 backbone (EXACT từ training)
        from torchvision.models import resnet18, ResNet18_Weights
        rn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet_backbone = nn.Sequential(*list(rn.children())[:-1])  # [B,512,1,1]
        res_dim = 512
        
        # ViT small patch16 224 (timm)
        self.vit = timm.create_model("vit_small_patch16_224", pretrained=True)
        vit_dim = self.vit.embed_dim
        # bỏ head, chỉ lấy embedding
        if hasattr(self.vit, "head"):
            self.vit.reset_classifier(0)
        
        # CBAM attention with spatial_kernel=3 (from training code)
        fused_dim = res_dim + vit_dim
        self.cbam = CBAM(fused_dim, reduction=16, spatial_kernel=3)
        
        # Classifier combining ResNet and ViT features
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # ResNet branch
        r = self.resnet_backbone(x)              # [B,512,1,1]
        r = r.view(r.size(0), -1)                # [B,512]
        
        # ViT branch
        v = self.vit(x)                          # [B,vit_dim]
        
        # Fusion + CBAM
        feat = torch.cat([r, v], dim=1)          # [B, C]
        feat_4d = feat.unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        feat_4d = self.cbam(feat_4d)             # CBAM attention
        feat = feat_4d.view(feat_4d.size(0), -1)
        out = self.classifier(feat)
        return out

# CBAMResNet18 for Nail Model (from nail training notebook)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Safe ratio for small channels
        safe_ratio = ratio if in_planes >= ratio else 1
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // safe_ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // safe_ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))

class CBAMModule(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAMModule, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        out = out * self.sa(out)
        return out

class CBAMResNet18(nn.Module):
    def __init__(self, num_classes):
        super(CBAMResNet18, self).__init__()
        # Load ResNet18 pretrained
        from torchvision.models import resnet18, ResNet18_Weights
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        # ResNet18 channel configs: 64, 128, 256, 512
        self.cbam1 = CBAMModule(64)
        self.cbam2 = CBAMModule(128)
        self.cbam3 = CBAMModule(256)
        self.cbam4 = CBAMModule(512)
        
        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # Stem
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Layer 1 + CBAM
        x = self.backbone.layer1(x)
        x = self.cbam1(x)

        # Layer 2 + CBAM
        x = self.backbone.layer2(x)
        x = self.cbam2(x)

        # Layer 3 + CBAM
        x = self.backbone.layer3(x)
        x = self.cbam3(x)

        # Layer 4 + CBAM
        x = self.backbone.layer4(x)
        x = self.cbam4(x)

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def load_model(model_name):
    """Load a specific model"""
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    config = MODEL_CONFIGS[model_name]
    model_path = config["path"]
    
    try:
        # Load state dict first to inspect structure
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state' in checkpoint:
                # Checkpoint with wrapper
                state_dict = checkpoint['model_state']
                print(f"📦 Found model_state in checkpoint for {model_name}")
            elif 'state_dict' in checkpoint:
                # Standard checkpoint format
                state_dict = checkpoint['state_dict']
                print(f"📦 Found state_dict in checkpoint for {model_name}")
            else:
                # Direct state dict
                state_dict = checkpoint
                print(f"📦 Using direct state_dict for {model_name}")
        else:
            # Old format - direct model
            state_dict = checkpoint
            print(f"📦 Using legacy format for {model_name}")
        
        # Create model based on type (exact API implementation)
        if config["type"] == "vit_cnn_hybrid":
            # DermNet model - ViTCNNHybrid (Swin + ConvNeXt + CBAM)
            model = ViTCNNHybrid(num_classes=len(config["classes"]))
            print(f"🔧 Created ViTCNNHybrid for {model_name} with {len(config['classes'])} classes")
        elif config["type"] == "cbam_resnet18":
            # Nail and Teeth models - CBAMResNet18 architecture
            model = CBAMResNet18(num_classes=len(config["classes"]))
            print(f"🔧 Created CBAMResNet18 for {model_name} with {len(config['classes'])} classes")
        
        # Try to load state dict with different strategies
        try:
            model.load_state_dict(state_dict, strict=True)
            print(f"✅ Loaded {model_name} with strict=True")
        except RuntimeError as e:
            print(f"⚠️ Strict loading failed for {model_name}, trying non-strict...")
            try:
                missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                if missing_keys:
                    print(f"⚠️ Missing keys: {missing_keys[:3]}...")
                if unexpected_keys:
                    print(f"⚠️ Unexpected keys: {unexpected_keys[:3]}...")
                print(f"✅ Loaded {model_name} with strict=False")
            except Exception as e2:
                print(f"❌ Failed to load {model_name}: {str(e2)}")
                return None
        
        model = model.to(DEVICE)
        model.eval()
        
        loaded_models[model_name] = model
        print(f"✅ Successfully loaded and cached {model_name} model")
        return model
        
    except Exception as e:
        print(f"❌ Error loading {model_name} model: {str(e)}")
        return None

def download_image_from_url(url):
    """Download image from URL with browser headers to avoid blocking"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # Convert bytes to PIL Image
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except Exception as e:
        print(f"⚠️ Failed to download image from URL: {str(e)}")
        return None

def predict_image_from_pil(pil_image, model_name, top_n):
    """Predict from PIL Image directly"""
    try:
        # Load model
        model = load_model(model_name)
        if model is None:
            return f"❌ **Error:** Failed to load {model_name} model."
        
        config = MODEL_CONFIGS[model_name]
        class_names = config["classes"]
        
        # Preprocess and predict
        image_tensor = transforms_inference(pil_image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            
            # Get top N predictions
            top_n = min(top_n, len(class_names))
            topk = torch.topk(probabilities, k=top_n)
            
            # Build JSON response
            predictions = []
            for prob, idx in zip(topk.values, topk.indices):
                predictions.append({
                    "class": class_names[int(idx)],
                    "confidence": float(prob)
                })
            
            result = {
                "success": True,
                "model": model_name,
                "architecture": config['architecture'],
                "description": config['description'],
                "predictions": predictions
            }
            
            return json.dumps(result, indent=2)
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e)
        }
        return json.dumps(error_result, indent=2)

def predict_image(image, model_name, top_n):
    """Predict using uploaded image"""
    if image is None:
        return "❌ **Error:** Please upload an image first."
    
    try:
        # Load model if not already loaded
        model = load_model(model_name)
        if model is None:
            return f"❌ **Error:** Failed to load {model_name} model."
        
        # Get model config
        config = MODEL_CONFIGS[model_name]
        class_names = config["classes"]
        
        # Preprocess image - handle different input types
        try:
            if isinstance(image, str):
                # Check if it's a URL
                if image.startswith(('http://', 'https://')):
                    # Download from URL
                    pil_image = download_image_from_url(image)
                    if pil_image is None:
                        return f"❌ **Error:** Cannot download image from URL. Please check the link or try uploading the file directly."
                else:
                    # Local file path - open with validation
                    try:
                        pil_image = Image.open(image)
                        pil_image.verify()  # Verify it's a valid image
                        pil_image = Image.open(image).convert('RGB')  # Reopen after verify
                    except Exception as e:
                        return f"❌ **Error:** Invalid or corrupted image file: {str(e)}"
            elif isinstance(image, Image.Image):
                # Already PIL Image
                pil_image = image.convert('RGB')
            else:
                # Try to convert to PIL Image
                try:
                    pil_image = Image.fromarray(image).convert('RGB')
                except Exception as e:
                    return f"❌ **Error:** Cannot convert to PIL Image: {str(e)}"
        except Exception as img_err:
            return f"❌ **Error:** Failed to process image: {str(img_err)}"
            
        image_tensor = transforms_inference(pil_image).unsqueeze(0).to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            logits = model(image_tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            
            # Get top N predictions
            top_n = min(top_n, len(class_names))
            topk = torch.topk(probabilities, k=top_n)
            
            # Build JSON response
            predictions = []
            for prob, idx in zip(topk.values, topk.indices):
                predictions.append({
                    "class": class_names[int(idx)],
                    "confidence": float(prob)
                })
            
            result = {
                "success": True,
                "model": model_name,
                "architecture": config['architecture'],
                "description": config['description'],
                "predictions": predictions
            }
            
            return json.dumps(result, indent=2)
            
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e)
        }
        return json.dumps(error_result, indent=2)

def get_model_info(model_name):
    """Get information about a specific model"""
    if model_name not in MODEL_CONFIGS:
        return "❌ Model not found"
    
    config = MODEL_CONFIGS[model_name]
    
    info = f"**Model:** {model_name.upper()}\n"
    info += f"**Description:** {config['description']}\n"
    info += f"**Architecture:** {config['type']}\n"
    info += f"**Classes:** {len(config['classes'])}\n"
    info += f"**Confidence Threshold:** {CONFIDENCE_THRESHOLDS[model_name]*100:.0f}%\n"
    
    is_loaded = model_name in loaded_models
    info += f"**Status:** {'✅ Loaded' if is_loaded else '⏳ Not loaded'}\n\n"
    
    classes = config['classes']
    info += "**Supported Conditions:**\n"
    for i, cls in enumerate(classes, 1):
        info += f"{i}. {cls}\n"
    
    return info

# Create Gradio Interface
def create_interface():
    """Create the main Gradio interface"""
    
    with gr.Blocks(title="🏥 Multi-Model Healthcare AI") as demo:
        
        # Header
        gr.Markdown("""
        # 🏥 Multi-Model Healthcare AI
        
        AI-powered medical image analysis for **skin diseases**, **teeth conditions**, and **nail disorders**.
        
        Upload an image and select a model to get AI predictions with confidence scores.
        
        ⚠️ **Medical Disclaimer:** This AI system is for educational and research purposes only. Always consult qualified healthcare professionals for medical advice.
        """)
        
        with gr.Tabs():
            # Tab 1: Main Prediction Interface
            with gr.Tab("🔬 Prediction"):
                with gr.Row():
                    # Left Column - Input
                    with gr.Column(scale=1):
                        gr.Markdown("### 📤 Upload & Configure")
                        
                        image_input = gr.Textbox(
                            label="Image URL",
                            placeholder="Enter image URL (e.g., https://example.com/image.jpg)",
                            lines=1
                        )
                        
                        model_dropdown = gr.Dropdown(
                            choices=list(MODEL_CONFIGS.keys()),
                            value="dermnet",
                            label="Select AI Model",
                            info="Choose the appropriate model for your image type"
                        )
                        
                        top_n_slider = gr.Slider(
                            minimum=1,
                            maximum=5,
                            value=3,
                            step=1,
                            label="Number of Predictions",
                            info="How many top predictions to show"
                        )
                        
                        predict_btn = gr.Button("🔬 Analyze Image", variant="primary")
                        
                    # Right Column - Output  
                    with gr.Column(scale=1):
                        gr.Markdown("### 📊 AI Analysis Results")
                        
                        prediction_output = gr.Markdown(
                            value="Upload an image and click 'Analyze Image' to see predictions..."
                        )
                        
                        model_info_output = gr.Markdown(
                            value=get_model_info("dermnet")
                        )
            
            # Tab 2: List Models API
            with gr.Tab("📋 Models List"):
                list_models_btn = gr.Button("📋 Get All Models", variant="primary")
                models_list_output = gr.Markdown()
            
            # Tab 3: Get Classes API
            with gr.Tab("🏷️ Model Classes"):
                classes_model_dropdown = gr.Dropdown(
                    choices=list(MODEL_CONFIGS.keys()),
                    value="dermnet",
                    label="Select Model"
                )
                get_classes_btn = gr.Button("🏷️ Get Classes", variant="primary")
                classes_output = gr.Markdown()
        
        # Footer
        gr.Markdown("""
        ---
        
        **🔬 Models:** DermNet (Skin), Teeth Disease Detection, Nail Disease Detection  
        **🛡️ Features:** Out-of-domain detection, Confidence thresholds, Multi-model support
        **🎯 Accuracy:** Built with ResNet18+ViT+CBAM and MobileNetV2 architectures
        
        *Created for DevFest 2025 AI Challenge*
        """)
        
        # Event Handlers
        def format_json_to_markdown(json_str):
            """Format JSON result to Markdown for UI display"""
            try:
                result = json.loads(json_str)
                
                if not result.get("success", False):
                    return f"❌ **Error:** {result.get('error', 'Unknown error')}"
                
                # Format predictions as Markdown
                output = f"### 🎯 Top {len(result['predictions'])} Predictions\n\n"
                
                for i, pred in enumerate(result['predictions'], 1):
                    class_name = pred['class']
                    confidence = pred['confidence'] * 100
                    output += f"{i}. **{class_name}** - {confidence:.1f}%\n"
                
                output += f"\n---\n"
                output += f"*Model: {result['description']} | Architecture: {result['architecture']}*"
                
                return output
            except Exception as e:
                return f"❌ **Error formatting result:** {str(e)}"
        
        def handle_prediction(url_input, model, top_n):
            """Handle prediction from URL or file path - returns JSON"""
            pil_image = None
            
            # CASE 1: Dict from Gradio client (contains file path after auto-download)
            if isinstance(url_input, dict):
                file_path = url_input.get('path', '')
                if file_path:
                    try:
                        pil_image = Image.open(file_path).convert('RGB')
                    except Exception as e:
                        error = {"success": False, "error": f"Cannot open image file: {e}"}
                        return json.dumps(error, indent=2)
            
            # CASE 2: String - could be URL or stringified dict
            elif isinstance(url_input, str):
                url_input = url_input.strip()
                
                # Direct HTTP URL
                if url_input.startswith(('http://', 'https://')):
                    pil_image = download_image_from_url(url_input)
                    if pil_image is None:
                        error = {"success": False, "error": "Failed to download image from URL"}
                        return json.dumps(error, indent=2)
                
                # Stringified dict
                elif url_input.startswith('{'):
                    try:
                        import ast
                        parsed = ast.literal_eval(url_input)
                        return handle_prediction(parsed, model, top_n)
                    except:
                        error = {"success": False, "error": "Invalid input format"}
                        return json.dumps(error, indent=2)
                else:
                    error = {"success": False, "error": "Please enter a valid HTTP/HTTPS URL"}
                    return json.dumps(error, indent=2)
            
            if pil_image is None:
                error = {"success": False, "error": "No image provided"}
                return json.dumps(error, indent=2)
            
            # Predict using the PIL image (returns JSON)
            return predict_image_from_pil(pil_image, model, top_n)
        
        def handle_prediction_ui(url_input, model, top_n):
            """Handle prediction for UI - converts JSON to Markdown"""
            json_result = handle_prediction(url_input, model, top_n)
            return format_json_to_markdown(json_result)
        
        def list_models():
            """List all available models with their info"""
            output = "# 🏥 Available AI Models\n\n"
            output += f"**Total Models:** {len(MODEL_CONFIGS)}\n\n"
            
            for model_name, config in MODEL_CONFIGS.items():
                output += f"## 🔬 {model_name.upper()}\n"
                output += f"- **Description:** {config['description']}\n"
                output += f"- **Architecture:** {config['architecture']}\n"
                output += f"- **Classes:** {len(config['classes'])}\n"
                output += f"- **Confidence Threshold:** {CONFIDENCE_THRESHOLDS[model_name]*100}%\n\n"
            
            return output
        
        def get_classes(model):
            """Get list of classes for a specific model"""
            if model not in MODEL_CONFIGS:
                return "❌ **Error:** Invalid model name"
            
            classes = MODEL_CONFIGS[model]["classes"]
            output = f"# 📋 {model.upper()} Classes\n\n"
            output += f"**Total Classes:** {len(classes)}\n\n"
            
            for i, cls in enumerate(classes, 1):
                output += f"{i}. {cls}\n"
            
            return output
        
        # Connect event handlers
        # UI button - shows Markdown
        predict_btn.click(
            fn=handle_prediction_ui,
            inputs=[image_input, model_dropdown, top_n_slider],
            outputs=prediction_output
        )
        
        # API endpoint - returns JSON
        # Create hidden textbox just for API registration
        hidden_json_output = gr.Textbox(visible=False, label="json_output")
        hidden_trigger = gr.Textbox(visible=False, label="hidden")
        
        hidden_trigger.submit(
            fn=handle_prediction,
            inputs=[image_input, model_dropdown, top_n_slider],
            outputs=hidden_json_output,
            api_name="handle_prediction"
        )
        
        model_dropdown.change(
            fn=get_model_info,
            inputs=model_dropdown,
            outputs=model_info_output,
            api_name="get_model_info"
        )
        
        list_models_btn.click(
            fn=list_models,
            inputs=[],
            outputs=models_list_output,
            api_name="list_models"
        )
        
        get_classes_btn.click(
            fn=get_classes,
            inputs=[classes_model_dropdown],
            outputs=classes_output,
            api_name="get_classes"
        )
    
    return demo

# Launch the interface
if __name__ == "__main__":
    # Pre-load models for faster inference
    print("🤖 Loading AI models...")
    for model_name in MODEL_CONFIGS.keys():
        load_model(model_name)
    
    print("🚀 Starting Gradio interface...")
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        share=True  # Required for HF Spaces
    )