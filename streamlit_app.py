import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import timm
import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============= CONFIG =============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_medagen_resnet18_vits_cbam.pth"
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.40

# ============= PAGE CONFIG =============
st.set_page_config(
    page_title="DermNet AI - Skin Disease Detection",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= CUSTOM CSS =============
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    .out-of-domain {
        background: linear-gradient(135deg, #ff9a56 0%, #ff6b6b 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
    .success-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

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
@st.cache_resource
def load_model():
    """Load model with caching"""
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found: {MODEL_PATH}")
        st.stop()
    
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
    """Make prediction from PIL image"""
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
            "threshold": confidence_threshold,
            "all_probabilities": probabilities.cpu().numpy()
        }

def create_confidence_chart(predictions):
    """Create confidence chart using Plotly"""
    classes = [pred["class"] for pred in predictions]
    confidences = [pred["confidence_percent"] for pred in predictions]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    fig = go.Figure(data=[
        go.Bar(
            x=confidences,
            y=classes,
            orientation='h',
            marker_color=colors[:len(classes)],
            text=[f'{conf:.1f}%' for conf in confidences],
            textposition='inside'
        )
    ])
    
    fig.update_layout(
        title="Top 3 Predictions - Confidence Levels",
        xaxis_title="Confidence (%)",
        yaxis_title="Disease Class",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    
    return fig

# ============= MAIN APP =============
def main():
    # Header
    st.markdown('<h1 class="main-header">🔬 DermNet AI - Skin Disease Detection</h1>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner("🔄 Loading AI model..."):
        model, class_names = load_model()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ Information")
        st.info(f"""
        **Model**: ResNet18 + ViT + CBAM
        **Device**: {DEVICE}
        **Classes**: {len(class_names)}
        **Confidence Threshold**: {CONFIDENCE_THRESHOLD*100:.0f}%
        """)
        
        st.header("📋 Supported Conditions")
        for i, class_name in enumerate(class_names, 1):
            st.write(f"{i}. {class_name}")
        
        st.header("🎯 How it works")
        st.write("""
        1. Upload a skin image
        2. AI analyzes the condition
        3. Get top 3 predictions
        4. If confidence < 40%, marked as "Out of Domain"
        """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a skin image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of the skin condition"
        )
        
        if uploaded_file is not None:
            # Display image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Add analysis button
            if st.button("🔍 Analyze Skin Condition", type="primary", use_container_width=True):
                with st.spinner("🧠 AI is analyzing the image..."):
                    # Make prediction
                    result = predict_from_image(image, model, class_names, CONFIDENCE_THRESHOLD)
                    
                    # Store result in session state
                    st.session_state['analysis_result'] = result
    
    with col2:
        st.header("📊 Analysis Results")
        
        if 'analysis_result' in st.session_state:
            result = st.session_state['analysis_result']
            
            if result["status"] == "out_of_domain":
                st.markdown(f"""
                <div class="out-of-domain">
                    <h3>⚠️ Out of Domain</h3>
                    <p>{result["message"]}</p>
                    <p>This image may not be a skin condition or is not within our trained dataset.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show confidence meter
                st.progress(result["max_confidence"])
                st.write(f"Maximum confidence: {result['max_confidence']*100:.2f}%")
                
            elif result["status"] == "success":
                st.markdown(f"""
                <div class="success-card">
                    <h3>✅ Analysis Complete</h3>
                    <p>Top 3 Predictions (Confidence ≥ {result['threshold']*100:.0f}%)</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show predictions
                for i, pred in enumerate(result["predictions"]):
                    with st.container():
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h4>{pred['rank']}. {pred['class']}</h4>
                            <p><strong>Confidence: {pred['confidence_percent']:.2f}%</strong></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Progress bar for confidence
                        st.progress(pred['confidence'])
                
                # Show confidence chart
                if len(result["predictions"]) > 1:
                    st.plotly_chart(
                        create_confidence_chart(result["predictions"]), 
                        use_container_width=True
                    )
        else:
            st.info("👆 Please upload an image and click 'Analyze' to see results")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔬 DermNet AI - Advanced Skin Disease Detection using Deep Learning</p>
        <p>Powered by ResNet18 + Vision Transformer + CBAM Attention</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()