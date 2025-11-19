# 🔬 DermNet AI API

API for AI-powered skin disease detection using deep learning.

**API URL**: `https://unbenefited-lura-animatingly.ngrok-free.dev`

---

## 📋 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check |
| `/predict` | POST | Upload image file |
| `/predict_base64` | POST | Send base64 image |
| `/classes` | GET | Get supported diseases |

---

## 🐍 Python Example

```python
import requests

# Upload and predict skin disease
with open('skin_image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post(
        'https://unbenefited-lura-animatingly.ngrok-free.dev/predict', 
        files=files
    )
    result = response.json()
    
    if result['status'] == 'success':
        print("Top 3 Predictions:")
        for pred in result['predictions']:
            print(f"  {pred['rank']}. {pred['class']} - {pred['confidence_percent']:.1f}%")
    else:
        print(f"Out of domain: {result['message']}")
```

---

## � Response Structure

### Success Response (Confidence ≥ 40%)
```json
{
  "status": "success",
  "predictions": [
    {
      "rank": 1,
      "class": "Melanoma Skin Cancer Nevi and Moles",
      "confidence": 0.85,
      "confidence_percent": 85.0
    },
    {
      "rank": 2,
      "class": "Eczema Photos",
      "confidence": 0.12,
      "confidence_percent": 12.0
    },
    {
      "rank": 3,
      "class": "Acne and Rosacea Photos",
      "confidence": 0.03,
      "confidence_percent": 3.0
    }
  ],
  "max_confidence": 0.85,
  "threshold": 0.4
}
```

### Out of Domain Response (< 40%)
```json
{
  "status": "out_of_domain",
  "message": "Out of domain (confidence: 35% < 40%)",
  "max_confidence": 0.35,
  "threshold": 0.4
}
```

### Error Response
```json
{
  "error": "No image provided"
}
```

---

## 🏥 Supported Diseases (14 types)

1. Acne and Rosacea Photos
2. Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions
3. Atopic Dermatitis Photos
4. Cellulitis Impetigo and other Bacterial Infections
5. Eczema Photos
6. Hair Loss Photos Alopecia and other Hair Diseases
7. Melanoma Skin Cancer Nevi and Moles
8. Nail Fungus and other Nail Disease
9. Poison Ivy Photos and other Contact Dermatitis
10. Psoriasis pictures Lichen Planus and related diseases
11. Scabies Lyme Disease and other Infestations and Bites
12. Seborrheic Keratoses and other Benign Tumors
13. Tinea Ringworm Candidiasis and other Fungal Infections
14. Warts Molluscum and other Viral Infections

---

## ⚠️ Notes

- **File formats**: JPEG, PNG, JPG only
- **File size**: Under 10MB
- **Timeout**: 10-15 seconds recommended
- **Medical disclaimer**: For educational use only