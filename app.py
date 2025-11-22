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
        "path": "best_medagen_resnet18_vits_cbam.pth",
        "description": "Skin disease detection using ResNet18 + ViT + CBAM",
        "architecture": "ResNet18 + Vision Transformer + CBAM Attention",
        "type": "fusion_model",
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
        "type": "mobilenet",
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
        "type": "mobilenet",
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
        if config["type"] == "fusion_model":
            # DermNet model - exact architecture
            model = ResNet18_ViTS_CBAM(num_classes=len(config["classes"]))
        else:
            # MobileNetV2 for teeth and nail models
            from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
            model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            model.classifier[1] = nn.Linear(model.last_channel, len(config["classes"]))
            print(f"🔧 Created MobileNetV2 for {model_name}")
        
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
            max_prob, max_idx = torch.max(probabilities, dim=0)
            max_confidence = float(max_prob)
            
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))
            normalized_entropy = float(entropy / torch.log(torch.tensor(len(class_names))))
            
            threshold = CONFIDENCE_THRESHOLDS.get(model_name, 0.50)
            is_out_of_domain = (max_confidence < threshold) or (normalized_entropy > 0.8)
            
            if is_out_of_domain:
                reasons = []
                if max_confidence < threshold:
                    reasons.append(f"low confidence ({max_confidence*100:.1f}% < {threshold*100:.0f}%)")
                if normalized_entropy > 0.8:
                    reasons.append(f"high uncertainty (entropy: {normalized_entropy:.3f})")
                
                output = f"🚫 **Out of Domain**\n\n"
                output += f"**Reason:** {' and '.join(reasons)}\n"
                output += f"**Suggestion:** This image may not contain {config['description'].lower()}."
                return output
            
            top_n = min(top_n, len(class_names))
            topk = torch.topk(probabilities, k=top_n)
            
            output = f"✅ **Prediction Successful**\n\n"
            output += f"**Model:** {config['description']}\n"
            output += f"**Max Confidence:** {max_confidence*100:.1f}%\n\n"
            output += "**Top Predictions:**\n"
            for i, (prob, idx) in enumerate(zip(topk.values, topk.indices)):
                confidence = float(prob) * 100
                class_name = class_names[int(idx)]
                output += f"**{i+1}.** {class_name} - **{confidence:.1f}%**\n"
            
            return output
    except Exception as e:
        return f"❌ **Error:** {str(e)}"

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
            max_prob, max_idx = torch.max(probabilities, dim=0)
            max_confidence = float(max_prob)
            
            # Calculate entropy for uncertainty detection
            entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8))
            normalized_entropy = float(entropy / torch.log(torch.tensor(len(class_names))))
            
            # Apply model-specific confidence threshold
            threshold = CONFIDENCE_THRESHOLDS.get(model_name, 0.50)
            
            # Out-of-domain detection
            is_out_of_domain = (max_confidence < threshold) or (normalized_entropy > 0.8)
            
            if is_out_of_domain:
                reasons = []
                if max_confidence < threshold:
                    reasons.append(f"low confidence ({max_confidence*100:.1f}% < {threshold*100:.0f}%)")
                if normalized_entropy > 0.8:
                    reasons.append(f"high uncertainty (entropy: {normalized_entropy:.3f})")
                
                output = f"🚫 **Out of Domain**\n\n"
                output += f"**Reason:** {' and '.join(reasons)}\n"
                output += f"**Confidence:** {max_confidence*100:.1f}%\n"
                output += f"**Threshold:** {threshold*100:.0f}%\n\n"
                output += f"**Suggestion:** This image may not contain {config['description'].lower()}. Please try uploading a relevant medical image."
                return output
            
            # Get top N predictions
            top_n = min(top_n, len(class_names))
            topk = torch.topk(probabilities, k=top_n)
            
            # Format successful prediction
            output = f"✅ **Prediction Successful**\n\n"
            output += f"**Model:** {config['description']}\n"
            output += f"**Max Confidence:** {max_confidence*100:.1f}%\n"
            output += f"**Entropy:** {normalized_entropy:.3f}\n\n"
            
            output += "**Top Predictions:**\n"
            for i, (prob, idx) in enumerate(zip(topk.values, topk.indices)):
                confidence = float(prob) * 100
                class_name = class_names[int(idx)]
                output += f"**{i+1}.** {class_name} - **{confidence:.1f}%**\n"
            
            return output
            
    except Exception as e:
        return f"❌ **Error:** {str(e)}"

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
        def handle_prediction(url_input, model, top_n):
            """Handle prediction from URL or file path"""
            pil_image = None
            
            # CASE 1: Dict from Gradio client (contains file path after auto-download)
            if isinstance(url_input, dict):
                file_path = url_input.get('path', '')
                if file_path:
                    try:
                        pil_image = Image.open(file_path).convert('RGB')
                    except Exception as e:
                        return f"❌ **Error:** Cannot open image file: {e}"
            
            # CASE 2: String - could be URL or stringified dict
            elif isinstance(url_input, str):
                url_input = url_input.strip()
                
                # Direct HTTP URL
                if url_input.startswith(('http://', 'https://')):
                    pil_image = download_image_from_url(url_input)
                    if pil_image is None:
                        return "❌ **Error:** Failed to download image from URL"
                
                # Stringified dict
                elif url_input.startswith('{'):
                    try:
                        import ast
                        parsed = ast.literal_eval(url_input)
                        return handle_prediction(parsed, model, top_n)
                    except:
                        return "❌ **Error:** Invalid input format"
                else:
                    return "❌ **Error:** Please enter a valid HTTP/HTTPS URL"
            
            if pil_image is None:
                return "❌ **Error:** No image provided"
            
            # Predict using the PIL image
            return predict_image_from_pil(pil_image, model, top_n)
        
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
        predict_btn.click(
            fn=handle_prediction,
            inputs=[image_input, model_dropdown, top_n_slider],
            outputs=prediction_output,
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