# HuggingFace Gradio API Documentation

🩺 Multi-model medical AI on HuggingFace Spaces using Gradio Client

## 🚀 Quick Start

```python
from gradio_client import Client

# Connect to Space
client = Client("thuonguyenvan/medagenn")

# Make prediction
result = client.predict(
    "https://example.com/image.jpg",  # image URL
    "dermnet",                          # model: dermnet/teeth/nail
    3,                                  # top N predictions
    api_name="/handle_prediction"
)

print(result)
```

## 📡 API Endpoints

### 1. Predict Image (`/handle_prediction`)

**Parameters**:
- `image_url` (str): HTTP/HTTPS image URL
- `select_ai_model` (str): Model name - `"dermnet"` | `"teeth"` | `"nail"`
- `number_of_predictions` (int): Number of top predictions (1-5)

**Returns**: Markdown-formatted string with predictions

### 2. Get Model Info (`/get_model_info`)

**Parameters**:
- `select_ai_model` (str): Model name

**Returns**: Model architecture and configuration info

## 🏥 Available Models

| Model | Domain | Architecture | Classes | File Size |
|-------|--------|--------------|---------|-----------|
| `dermnet` | Skin diseases | ResNet18 + ViT + CBAM | 14 | 128MB |
| `teeth` | Dental conditions | ResNet18 + CBAM | 6 (incl. Unknown) | 45MB |
| `nail` | Nail disorders | ResNet18 + CBAM | 7 (incl. Unknown) | 45MB |

**Note**: All models now use advanced architectures with CBAM attention mechanism. Teeth and nail models include Unknown class for out-of-domain detection.

## 💻 Code Examples

### Basic Usage

```python
from gradio_client import Client

# Initialize client
client = Client("thuonguyenvan/medagenn")

# Test image URL
url = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/480px-Cat03.jpg"

# Get predictions for all models
for model in ["dermnet", "teeth", "nail"]:
    result = client.predict(url, model, 3, api_name="/handle_prediction")
    print(f"\n{model.upper()} Results:\n{result}")
```

### With Error Handling

```python
from gradio_client import Client

def predict_image(image_url, model="dermnet", top_n=3):
    try:
        client = Client("thuonguyenvan/medagenn")
        result = client.predict(
            image_url,
            model,
            top_n,
            api_name="/handle_prediction"
        )
        return result
    except Exception as e:
        return f"Error: {str(e)}"

# Use it
result = predict_image("https://example.com/skin_lesion.jpg", "dermnet")
print(result)
```

### Batch Processing

```python
from gradio_client import Client

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
    result = client.predict(url, "dermnet", 3, api_name="/handle_prediction")
    results.append(result)

# Display
for i, result in enumerate(results):
    print(f"\nImage {i+1}:\n{result}")
```

### Get Model Information

```python
from gradio_client import Client

client = Client("thuonguyenvan/medagenn")

# Get info for each model
for model in ["dermnet", "teeth", "nail"]:
    info = client.predict(model, api_name="/get_model_info")
    print(f"\n{model.upper()} Info:\n{info}")
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
