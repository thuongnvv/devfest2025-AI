# HuggingFace Gradio API Documentation

🩺 Multi-model medical AI on HuggingFace Spaces using Gradio Client

**⚠️ BREAKING CHANGE (v2.0):** API now returns JSON instead of Markdown for easier parsing.

## 🚀 Quick Start

```python
from gradio_client import Client
import json

# Connect to Space
client = Client("thuonguyenvan/medagenn")

# Make prediction (returns JSON string)
result_json = client.predict(
    "https://example.com/image.jpg",  # image URL
    "dermnet",                          # model: dermnet/teeth/nail
    3,                                  # top N predictions
    api_name="/handle_prediction"
)

# Parse JSON
result = json.loads(result_json)
print(f"Success: {result['success']}")
print(f"Top prediction: {result['predictions'][0]['class']}")
print(f"Confidence: {result['predictions'][0]['confidence']*100:.1f}%")
```

## 📋 Response Format

### Success Response

```json
{
  "success": true,
  "model": "dermnet",
  "architecture": "ResNet18 + ViT + CBAM",
  "description": "Skin disease detection using ResNet18 + ViT + CBAM",
  "predictions": [
    {
      "class": "Acne and Rosacea Photos",
      "confidence": 0.524
    },
    {
      "class": "Atopic Dermatitis Photos",
      "confidence": 0.160
    }
  ]
}
```

### Error Response

```json
{
  "success": false,
  "error": "Failed to download image from URL"
}
```

## 📡 API Endpoints

### 1. 🔬 Predict Image (`/handle_prediction`)

**Returns**: JSON string

**Parameters**:
- `image_url` (str): HTTP/HTTPS image URL
- `select_ai_model` (str): `"dermnet"` | `"teeth"` | `"nail"`
- `number_of_predictions` (int): 1-5

**Success Response**:
```json
{
  "success": true,
  "model": "dermnet",
  "architecture": "Swin Transformer + ConvNeXt + CBAM Fusion",
  "description": "Skin disease detection using Swin Tiny + ConvNeXt + CBAM",
  "predictions": [
    {"class": "Acne and Rosacea Photos", "confidence": 0.524},
    {"class": "Atopic Dermatitis Photos", "confidence": 0.160}
  ]
}
```

**Error Response**:
```json
{
  "success": false,
  "error": "Failed to download image from URL"
}
```

**Example**:
```python
import json
result_json = client.predict(
    "https://example.com/image.jpg",
    "dermnet",
    3,
    api_name="/handle_prediction"
)
result = json.loads(result_json)
if result['success']:
    for pred in result['predictions']:
        print(f"{pred['class']}: {pred['confidence']*100:.1f}%")
```

---

### 2. ℹ️ Get Model Info (`/get_model_info`)

**Returns**: Markdown string

**Parameters**:
- `select_ai_model` (str): Model name

**Example**:
```python
info = client.predict("dermnet", api_name="/get_model_info")
print(info)
```

---

### 3. 📋 List All Models (`/list_models`)

**Returns**: Markdown string

**Parameters**: None

**Example**:
```python
models = client.predict(api_name="/list_models")
print(models)
```

---

### 4. 🏷️ Get Model Classes (`/get_classes`)

**Returns**: Markdown string

**Parameters**:
- `select_model` (str): Model name

**Example**:
```python
classes = client.predict("dermnet", api_name="/get_classes")
print(classes)
```

## 🏥 Available Models

| Model | Domain | Architecture | Classes | File Size |
|-------|--------|--------------|---------|-----------|
| `dermnet` | Skin diseases | Swin Transformer + ConvNeXt + CBAM Fusion | 23 | 216MB |
| `teeth` | Dental conditions | ResNet18 + CBAM Attention | 6 (incl. Unknown) | 45MB |
| `nail` | Nail disorders | ResNet18 + CBAM Attention | 7 (incl. Unknown) | 45MB |

**Model Details:**
- **DermNet**: Advanced hybrid model combining Vision Transformer (Swin Tiny) and CNN (ConvNeXt Tiny) with gated fusion and CBAM attention. Covers 23 comprehensive skin disease categories.
- **Teeth & Nail**: Efficient ResNet18-based models with CBAM (Convolutional Block Attention Module), including Unknown class for out-of-domain detection.

## 💻 Code Examples

### Basic Usage

```python
from gradio_client import Client
import json

# Initialize client
client = Client("thuonguyenvan/medagenn")

# Test image URL
url = "https://example.com/skin_lesion.jpg"

# Get predictions
result_json = client.predict(url, "dermnet", 3, api_name="/handle_prediction")

# Parse JSON
result = json.loads(result_json)
if result['success']:
    print(f"Model: {result['model']}")
    print(f"Architecture: {result['architecture']}")
    print("\nPredictions:")
    for i, pred in enumerate(result['predictions'], 1):
        print(f"{i}. {pred['class']}: {pred['confidence']*100:.1f}%")
else:
    print(f"Error: {result['error']}")
```

### With Error Handling

```python
from gradio_client import Client
import json

def predict_image(image_url, model="dermnet", top_n=3):
    try:
        client = Client("thuonguyenvan/medagenn")
        result_json = client.predict(
            image_url,
            model,
            top_n,
            api_name="/handle_prediction"
        )
        result = json.loads(result_json)
        
        if result['success']:
            return result['predictions']
        else:
            print(f"API Error: {result['error']}")
            return None
    except Exception as e:
        print(f"Connection Error: {str(e)}")
        return None

# Use it
predictions = predict_image("https://example.com/skin_lesion.jpg", "dermnet")
if predictions:
    for pred in predictions:
        print(f"{pred['class']}: {pred['confidence']*100:.1f}%")
```

### Batch Processing

```python
from gradio_client import Client
import json

client = Client("thuonguyenvan/medagenn")

# Multiple images
image_urls = [
    "https://example.com/image1.jpg",
    "https://example.com/image2.jpg",
    "https://example.com/image3.jpg"
]

# Process all
results = []
for url in image_urls:
    result_json = client.predict(url, "dermnet", 3, api_name="/handle_prediction")
    result = json.loads(result_json)
    results.append(result)

# Display
for i, result in enumerate(results):
    print(f"\n=== Image {i+1} ===")
    if result['success']:
        for pred in result['predictions']:
            print(f"  {pred['class']}: {pred['confidence']*100:.1f}%")
    else:
        print(f"  Error: {result['error']}")
```

### Get All Models

```python
from gradio_client import Client

client = Client("thuonguyenvan/medagenn")

# List all models
models_info = client.predict(api_name="/list_models")
print(models_info)

# Get classes for specific model
classes = client.predict("dermnet", api_name="/get_classes")
print(classes)
```

## 📋 View Available APIs

```python
from gradio_client import Client

client = Client("thuonguyenvan/medagenn")

# List all available endpoints
print(client.view_api())
```

**Output**:
```
Named API endpoints: 4

 - predict(image_url, select_ai_model, number_of_predictions, api_name="/handle_prediction") -> json_output
    Returns: [Textbox] str (JSON)

 - predict(select_ai_model, api_name="/get_model_info") -> value_15
    Returns: [Markdown] str

 - predict(api_name="/list_models") -> value_18
    Returns: [Markdown] str

 - predict(select_model, api_name="/get_classes") -> value_22
    Returns: [Markdown] str
```

## 📊 API Summary

| Endpoint | Output Format | Use Case |
|----------|---------------|----------|
| `/handle_prediction` | JSON string | API clients, programmatic access |
| `/get_model_info` | Markdown | Model details |
| `/list_models` | Markdown | Browse models |
| `/get_classes` | Markdown | Class names |

## ⚠️ Image URL Requirements

- **Format**: JPEG, PNG, WebP
- **Protocol**: HTTP or HTTPS only
- **Accessibility**: Must be publicly accessible
- **Size**: Auto-resized to 224x224

## 🔧 Installation

```bash
pip install gradio-client
```

## ⚠️ Medical Disclaimer

This API is for **educational and research purposes only**. Not for medical diagnosis. Always consult qualified healthcare professionals.
