# Healthcare AI API

Multi-model AI API for healthcare image analysis supporting skin diseases, dental conditions, and nail disorders.

## 🩺 Models

- **DermNet**: Skin disease classification (ResNet18 + ViT + CBAM) - 14 classes
- **Teeth**: Dental condition detection (MobileNetV2) - 5 classes  
- **Nail**: Nail disorder classification (MobileNetV2) - 6 classes

## 🚀 Quick Start

```bash
# Make executable and start API
chmod +x start_api.sh
./start_api.sh
```

Or manually:

```bash
# Install dependencies
pip install flask flask-cors torch torchvision timm pillow requests numpy

# Start API
python3 api.py
```

## 📡 API Endpoints

### Health Check
```http
GET /health
```

### List Models
```http
GET /models
```

### Get Model Classes
```http
GET /models/{model_name}/classes
```

### Make Prediction
```http
POST /predict
Content-Type: application/json

{
  "model": "dermnet|teeth|nail",
  "image": "base64_encoded_image",
  "n": 5
}
```

Or with image URL:
```http
POST /predict
Content-Type: application/json

{
  "model": "dermnet",
  "image_url": "https://example.com/image.jpg",
  "n": 3
}
```

## 🧪 Testing

```bash
# Full API testing
python3 test_api_client.py

# Test with real images
python3 test_real_images.py
```

## 📋 Model Details

### DermNet (Skin Diseases)
- **Architecture**: ResNet18 + Vision Transformer + CBAM Attention
- **Classes**: 14 skin conditions
- **Confidence Threshold**: 40%
- **Input**: 224x224 RGB images

### Teeth (Dental Conditions)  
- **Architecture**: MobileNetV2
- **Classes**: 5 dental conditions
- **Confidence Threshold**: 60%
- **Input**: 224x224 RGB images

### Nail (Nail Disorders)
- **Architecture**: MobileNetV2  
- **Classes**: 6 nail conditions
- **Confidence Threshold**: 70%
- **Input**: 224x224 RGB images

## 🔧 Model Files

Place these model files in the project directory:
- `best_medagen_resnet18_vits_cbam.pth`
- `best_teeth_model.pth` 
- `best_nail_model.pth`

## 📊 Response Format

```json
{
  "success": true,
  "model_info": {
    "name": "dermnet",
    "architecture": "ResNet18_ViTS_CBAM",
    "num_classes": 14,
    "confidence_threshold": 0.4
  },
  "result": {
    "top_prediction": "Acne and Rosacea",
    "confidence": 0.85,
    "is_out_of_domain": false,
    "threshold": 0.4,
    "top_predictions": [
      {
        "rank": 1,
        "class": "Acne and Rosacea", 
        "confidence": 0.85
      }
    ]
  },
  "processing_time": 0.234
}
```

## 🛡️ Error Handling

The API includes comprehensive error handling for:
- Invalid model names
- Missing/corrupt images
- Network timeouts
- Model loading failures
- Out-of-domain predictions

## 💡 Usage Examples

### Python Client
```python
import requests
import base64

# Health check
response = requests.get("http://localhost:5000/health")

# Make prediction
with open("image.jpg", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:5000/predict", json={
    "model": "dermnet",
    "image": image_b64,
    "n": 3
})

result = response.json()
print(f"Prediction: {result['result']['top_prediction']}")
print(f"Confidence: {result['result']['confidence']:.2%}")
```

### cURL
```bash
# Health check
curl http://localhost:5000/health

# List models
curl http://localhost:5000/models

# Get classes
curl http://localhost:5000/models/dermnet/classes
```

## 🔍 Model Architectures

### CBAM Attention Module
- Channel attention with average and max pooling
- Spatial attention with configurable kernel size
- Reduction ratio: 16

### ResNet18_ViTS_CBAM
- ResNet18 backbone with CBAM attention
- Vision Transformer integration
- Custom classifier head

### MobileNetV2 Models
- Pre-trained MobileNetV2 backbone
- Custom classifier for specific conditions
- Efficient inference for mobile deployment

## 📈 Performance

- Average response time: ~0.2-0.5 seconds
- Supports concurrent requests
- GPU acceleration when available
- Automatic CPU fallback

## 🔒 Security

- Input validation for all endpoints
- File size limits for image uploads
- Timeout protection for external URLs
- Error sanitization in responses

## ⚠️ Medical Disclaimer

This AI system is for **educational and research purposes only**. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical decisions.
