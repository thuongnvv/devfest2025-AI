# 🔬 DermNet AI - Frontend Integration Guide

Simple API for AI-powered skin disease detection.

**API URL**: `https://unbenefited-lura-animatingly.ngrok-free.dev`

---

## 🚀 Quick Start

### 1. Check API Health
```javascript
fetch('https://unbenefited-lura-animatingly.ngrok-free.dev/')
  .then(res => res.json())
  .then(data => console.log(data));
```

### 2. Upload & Predict
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
    // Show top 3 predictions
    result.predictions.forEach(pred => {
      console.log(`${pred.rank}. ${pred.class} - ${pred.confidence_percent.toFixed(1)}%`);
    });
  } else {
    // Out of domain (confidence < 40%)
    console.log(result.message);
  }
});
```

---

## 📋 API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Health check |
| `/predict` | POST | Upload image file |
| `/predict_base64` | POST | Send base64 image |
| `/classes` | GET | Get supported diseases |

---

## 📤 Request Format

### File Upload
```javascript
const formData = new FormData();
formData.append('image', file); // File object from input
```

### Base64
```javascript
const payload = {
  image: "base64_string_here" // Remove data:image/jpeg;base64, prefix
};
```

---

## 📥 Response Format

### Success (Confidence ≥ 40%)
```json
{
  "status": "success",
  "predictions": [
    {
      "rank": 1,
      "class": "Melanoma Skin Cancer Nevi and Moles",
      "confidence": 0.85,
      "confidence_percent": 85.0
    }
  ],
  "max_confidence": 0.85,
  "threshold": 0.4
}
```

### Out of Domain (< 40%)
```json
{
  "status": "out_of_domain",
  "message": "Out of domain (confidence: 35% < 40%)",
  "max_confidence": 0.35,
  "threshold": 0.4
}
```

### Error
```json
{
  "error": "No image provided"
}
```

---

## 🎯 Frontend Implementation

### React Example
```jsx
import { useState } from 'react';

function SkinAnalyzer() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const analyzeImage = async (file) => {
    setLoading(true);
    const formData = new FormData();
    formData.append('image', file);

    try {
      const response = await fetch(
        'https://unbenefited-lura-animatingly.ngrok-free.dev/predict',
        { method: 'POST', body: formData }
      );
      const data = await response.json();
      setResult(data);
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <input 
        type="file" 
        onChange={(e) => analyzeImage(e.target.files[0])}
        accept="image/*"
      />
      
      {loading && <p>Analyzing...</p>}
      
      {result && (
        <div>
          {result.status === 'success' ? (
            <div>
              <h3>Top Predictions:</h3>
              {result.predictions.map(pred => (
                <div key={pred.rank}>
                  {pred.rank}. {pred.class} - {pred.confidence_percent.toFixed(1)}%
                </div>
              ))}
            </div>
          ) : (
            <p>{result.message}</p>
          )}
        </div>
      )}
    </div>
  );
}
```

### Vanilla JS Example
```html
<!DOCTYPE html>
<html>
<head>
    <title>Skin Disease Detector</title>
</head>
<body>
    <input type="file" id="imageInput" accept="image/*">
    <div id="result"></div>

    <script>
        document.getElementById('imageInput').addEventListener('change', async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            const formData = new FormData();
            formData.append('image', file);

            try {
                const response = await fetch(
                    'https://unbenefited-lura-animatingly.ngrok-free.dev/predict',
                    { method: 'POST', body: formData }
                );
                const result = await response.json();
                
                const resultDiv = document.getElementById('result');
                if (result.status === 'success') {
                    resultDiv.innerHTML = result.predictions
                        .map(p => `${p.rank}. ${p.class} - ${p.confidence_percent.toFixed(1)}%`)
                        .join('<br>');
                } else {
                    resultDiv.innerHTML = result.message;
                }
            } catch (error) {
                document.getElementById('result').innerHTML = 'Error: ' + error.message;
            }
        });
    </script>
</body>
</html>
```

---

## 🏥 Supported Diseases (14 types)

1. Acne and Rosacea
2. Skin Cancer / Melanoma
3. Eczema
4. Psoriasis
5. Fungal Infections
6. Bacterial Infections
7. Viral Infections (Warts)
8. Hair Loss / Alopecia
9. Nail Diseases
10. Benign Tumors
11. Contact Dermatitis
12. Atopic Dermatitis
13. Parasitic Infections
14. Malignant Lesions

---

## ⚠️ Important Notes

- **File Size**: Keep images under 10MB
- **Format**: JPEG, PNG, JPG only  
- **Quality**: Clear, well-lit skin images work best
- **Timeout**: Set 10-15 second timeout for requests
- **CORS**: Enabled for all origins
- **Medical Disclaimer**: For educational use only, not medical diagnosis

---

## 🔧 Error Handling

```javascript
async function predictSkinDisease(file) {
  try {
    const formData = new FormData();
    formData.append('image', file);
    
    const response = await fetch(API_URL + '/predict', {
      method: 'POST',
      body: formData,
      // Add timeout
      signal: AbortSignal.timeout(15000)
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    
    const result = await response.json();
    
    if (result.error) {
      throw new Error(result.error);
    }
    
    return result;
    
  } catch (error) {
    console.error('Prediction failed:', error);
    return { error: error.message };
  }
}
```

That's it! Simple integration for any frontend framework. 🎉