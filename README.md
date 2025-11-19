# 🔬 DermNet AI - Skin Disease Detection

AI-powered skin disease detection API using ResNet18 + Vision Transformer + CBAM.

**API URL**: `https://unbenefited-lura-animatingly.ngrok-free.dev`

## 🚀 Quick Start

### Start API Server
```bash
pip install -r requirements.txt
python app.py
```

### Test API
```bash
curl https://unbenefited-lura-animatingly.ngrok-free.dev/
```

## 📋 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check |
| `/predict` | POST | Upload image file |
| `/predict_base64` | POST | Send base64 image |
| `/classes` | GET | Get supported diseases |

## 💻 Frontend Integration

See `FRONTEND_DOCS.md` for complete integration guide with React, Vanilla JS examples.

### Quick Example
```javascript
const formData = new FormData();
formData.append('image', fileInput.files[0]);

fetch('https://unbenefited-lura-animatingly.ngrok-free.dev/predict', {
  method: 'POST', 
  body: formData
})
.then(res => res.json())
.then(result => {
  if (result.status === 'success') {
    // Show predictions
    result.predictions.forEach(pred => {
      console.log(`${pred.class} - ${pred.confidence_percent}%`);
    });
  } else {
    console.log('Out of domain'); 
  }
});
```

## 🧠 Model Architecture

- **ResNet18**: Feature extraction backbone
- **Vision Transformer (ViT-Small)**: Global attention mechanism  
- **CBAM**: Convolutional Block Attention Module
- **Fusion Layer**: Combines ResNet and ViT features
- **Classification Head**: Final prediction layer

## ⚙️ Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)
- 4GB+ RAM
- Internet connection (for downloading pre-trained weights)

## 📊 Model Performance

- **Confidence Threshold**: 40%
- **Out-of-Domain Detection**: Automatic filtering of non-skin images
- **Top-3 Predictions**: Ranked results with confidence scores

## 🔧 Configuration

Edit the following variables in the scripts:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "best_medagen_resnet18_vits_cbam.pth"
CONFIDENCE_THRESHOLD = 0.40
IMAGE_SIZE = 224
```

## 🌐 Public Access with ngrok

The Flask API includes ngrok integration for public access:

```python
NGROK_TOKEN = "your_ngrok_token_here"
```

## 📝 Notes

- Ensure the model file `best_medagen_resnet18_vits_cbam.pth` is in the project directory
- For best results, use clear, well-lit images of skin conditions
- The model is trained on the DermNet dataset with 14 selected classes
- Out-of-domain detection helps filter non-relevant images

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## ⚠️ Disclaimer

This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with a qualified healthcare provider for medical advice.

## 📄 License

This project is licensed under the MIT License.