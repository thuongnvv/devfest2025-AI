# Healthcare AI API Documentation

🩺 Multi-model medical AI API for skin diseases, dental conditions, and nail disorders.

## 🚀 Quick Start

```bash
# Start API
python3 api.py

# Test API
python3 test_api_client.py
```

**API Base URL**: `http://localhost:5000`

## 📡 Endpoints

### 1. Health Check
```http
GET /health
```
**Response**:
```json
{
  "status": "healthy",
  "device": "cpu",
  "models_status": {
    "dermnet": {"loaded": true, "file_exists": true},
    "teeth": {"loaded": true, "file_exists": true},
    "nail": {"loaded": true, "file_exists": true}
  }
}
```

### 2. List Models
```http
GET /models
```
**Response**:
```json
{
  "total_models": 3,
  "models": {
    "dermnet": {
      "description": "Skin disease detection",
      "architecture": "ResNet18 + ViT + CBAM",
      "num_classes": 14,
      "confidence_threshold": 0.4
    },
    "teeth": {...},
    "nail": {...}
  }
}
```

### 3. Get Model Classes
```http
GET /models/{model}/classes
```
**Example**: `/models/dermnet/classes`

**Response**:
```json
{
  "model": "dermnet",
  "num_classes": 14,
  "classes": [
    "Acne and Rosacea Photos",
    "Melanoma",
    "Eczema",
    "..."
  ]
}
```

### 4. Make Prediction
```http
POST /predict
Content-Type: application/json
```

**Request Body (Base64)**:
```json
{
  "model": "dermnet",
  "image": "base64_encoded_image_string",
  "n": 5
}
```

**Request Body (URL)**:
```json
{
  "model": "teeth", 
  "image_url": "https://example.com/image.jpg",
  "n": 3
}
```

**Response**:
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
      },
      {
        "rank": 2, 
        "class": "Eczema",
        "confidence": 0.12
      }
    ]
  },
  "processing_time": 0.234
}
```

## 🏥 Models

| Model | Domain | Classes | Threshold | Architecture |
|-------|--------|---------|-----------|--------------|
| `dermnet` | Skin diseases | 14 | 40% | ResNet18+ViT+CBAM |
| `teeth` | Dental conditions | 5 | 60% | MobileNetV2 |
| `nail` | Nail disorders | 6 | 70% | MobileNetV2 |

## 💻 Code Examples

### Python
```python
import requests
import base64

# Health check
response = requests.get("http://localhost:5000/health")
print(response.json())

# Predict with base64 image
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

### JavaScript
```javascript
// Health check
fetch('http://localhost:5000/health')
  .then(response => response.json())
  .then(data => console.log(data));

// Predict with URL
fetch('http://localhost:5000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    model: 'teeth',
    image_url: 'https://example.com/tooth.jpg',
    n: 5
  })
})
.then(response => response.json())
.then(data => console.log(data.result));
```

### cURL
```bash
# Health check
curl http://localhost:5000/health

# List models
curl http://localhost:5000/models

# Get classes
curl http://localhost:5000/models/dermnet/classes

# Predict (with file)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dermnet",
    "image": "'$(base64 -w 0 image.jpg)'",
    "n": 5
  }'
```

## ⚠️ Error Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `400` | Bad request (invalid model, missing image, etc.) |
| `500` | Server error (model loading failed, etc.) |

**Error Response**:
```json
{
  "success": false,
  "error": "Invalid model name",
  "details": "Model 'invalid' not found. Available: dermnet, teeth, nail"
}
```

## 🔧 Configuration

**Image Requirements**:
- Format: JPEG, PNG, WebP
- Size: Any (auto-resized to 224x224)
- Max file size: 10MB

**Response Times**:
- DermNet: ~0.2s
- Teeth/Nail: ~0.05s

## 📝 Notes

- **Out-of-domain detection**: Images with confidence below threshold are marked as out-of-domain
- **GPU support**: Automatically uses GPU if available, falls back to CPU
- **CORS enabled**: Can be called from web browsers
- **Concurrent requests**: Supports multiple simultaneous predictions

## 🧪 Testing

```bash
# Full API test suite
python3 test_api_client.py

# Test with real medical images  
python3 test_real_images.py
```

## ⚠️ Medical Disclaimer

This API is for **educational and research purposes only**. Not for medical diagnosis or treatment. Always consult healthcare professionals.