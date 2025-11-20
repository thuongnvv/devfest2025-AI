# API Quick Reference

## Start API
```bash
python3 api.py
# Access: http://localhost:5000
```

## Endpoints
```http
GET  /health                     # Check API status
GET  /models                     # List all models  
GET  /models/{model}/classes     # Get model classes
POST /predict                    # Make prediction
```

## Models
- `dermnet` - Skin diseases (14 classes, 40% threshold)
- `teeth` - Dental conditions (5 classes, 60% threshold)  
- `nail` - Nail disorders (6 classes, 70% threshold)

## Predict Request
```json
{
  "model": "dermnet|teeth|nail",
  "image": "base64_string",     // OR
  "image_url": "http://...",    // Image URL
  "n": 5                        // Top N predictions
}
```

## Predict Response
```json
{
  "success": true,
  "result": {
    "top_prediction": "Acne and Rosacea",
    "confidence": 0.85,
    "is_out_of_domain": false,
    "top_predictions": [...]
  },
  "processing_time": 0.234
}
```

## Python Example
```python
import requests, base64

# Predict
with open("image.jpg", "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()

response = requests.post("http://localhost:5000/predict", json={
    "model": "dermnet", "image": img_b64, "n": 3
})
print(response.json()["result"]["top_prediction"])
```

## cURL Example
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"model": "teeth", "image_url": "https://example.com/tooth.jpg", "n": 5}'
```

## Test
```bash
python3 test_api_client.py      # Full tests
python3 test_real_images.py     # Real image tests
```