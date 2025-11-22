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

### 1. Predict Image (`/handle_prediction`)

**Parameters**:
- `image_url` (str): HTTP/HTTPS image URL
- `select_ai_model` (str): Model name - `"dermnet"` | `"teeth"` | `"nail"`
- `number_of_predictions` (int): Number of top predictions (1-5)

**Returns**: JSON string with prediction results

**Example**:
```python
result_json = client.predict(
    "https://example.com/image.jpg",
    "dermnet",
    3,
    api_name="/handle_prediction"
)

# Parse JSON
import json
result = json.loads(result_json)
if result['success']:
    for pred in result['predictions']:
        print(f"{pred['class']}: {pred['confidence']*100:.1f}%")
```

### 2. Get Model Info (`/get_model_info`)

**Parameters**:
- `select_ai_model` (str): Model name

**Returns**: Markdown-formatted model information

### 3. List Models (`/list_models`)

**Returns**: Markdown-formatted list of all available models

### 4. Get Classes (`/get_classes`)

**Parameters**:
- `select_ai_model` (str): Model name

**Returns**: Markdown-formatted list of classes for the specified model

## 🏥 Available Models

| Model | Domain | Architecture | Classes | File Size |
|-------|--------|--------------|---------|-----------|
| `dermnet` | Skin diseases | Swin Tiny + ConvNeXt + CBAM | 23 | 216MB |
| `teeth` | Dental conditions | ResNet18 + CBAM | 6 (incl. Unknown) | 45MB |
| `nail` | Nail disorders | ResNet18 + CBAM | 7 (incl. Unknown) | 45MB |

**Model Details:**
- **DermNet**: Advanced hybrid model combining Vision Transformer (Swin) and CNN (ConvNeXt) with gated fusion and CBAM attention. Covers 23 comprehensive skin disease categories.
- **Teeth & Nail**: Efficient ResNet18-based models with CBAM attention, including Unknown class for out-of-domain detection.

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
Named API endpoints: 2

- predict(..., api_name="/handle_prediction") -> value
  Parameters:
   - image_url: str
   - select_ai_model: Literal[dermnet, teeth, nail]
   - number_of_predictions: float (1-5)

- predict(..., api_name="/get_model_info") -> value
  Parameters:
   - select_ai_model: Literal[dermnet, teeth, nail]
```

## 🔧 Installation

```bash
pip install gradio-client
```

## ⚠️ Image URL Requirements

- **Format**: JPEG, PNG, WebP
- **Protocol**: Must be HTTP or HTTPS
- **Accessibility**: URL must be publicly accessible
- **Size**: Any (auto-resized to 224x224)

## 📝 Response Format

Predictions are returned as **Markdown-formatted strings**:

```markdown
### 🎯 Top 3 Predictions

1. **Acne and Rosacea Photos** - 85.2%
2. **Eczema Photos** - 12.4%
3. **Psoriasis pictures** - 2.1%

---
*Confidence Threshold: 40% | Model: DermNet (ResNet18+ViT+CBAM)*
```

Out-of-domain detection:
```markdown
🚫 **Out of Domain**

The image does not match any known condition with sufficient confidence.
- **Highest confidence:** 32.4% (Acne)
- **Threshold:** 40%

💡 **Suggestions:**
- Ensure image shows relevant medical condition
- Try different lighting/angle
- Select correct AI model
```

## 🧪 Testing

Complete test notebook: `test_hf_api.ipynb`

```python
# Cell 1: Connect
from gradio_client import Client
client = Client("thuonguyenvan/medagenn")

# Cell 2: Test URL
test_url = "https://example.com/image.jpg"

# Cell 3: Test all models
for model in ["dermnet", "teeth", "nail"]:
    result = client.predict(test_url, model, 3, api_name="/handle_prediction")
    print(f"\n{model}: {result}")
```

## 🌐 Space URL

**Web Interface**: https://huggingface.co/spaces/thuonguyenvan/medagenn

## ⚠️ Medical Disclaimer

This API is for **educational and research purposes only**. Not for medical diagnosis or treatment. Always consult qualified healthcare professionals for medical decisions.

## 📚 Related Files

- `test_hf_api.ipynb` - Interactive testing notebook
- `app.py` - Gradio application source
- `API_DOCS.md` - Flask REST API documentation (localhost)
