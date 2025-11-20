---
title: Multi-Model Healthcare AI
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
license: mit
---

# 🏥 Multi-Model Healthcare AI

Multi-model AI system for healthcare image analysis supporting **skin diseases**, **dental conditions**, and **nail disorders**.

## 🩺 Supported Models

- **DermNet**: Skin disease classification (ResNet18 + ViT + CBAM) - 14 classes
- **Teeth**: Dental condition detection (MobileNetV2) - 5 classes  
- **Nail**: Nail disorder classification (MobileNetV2) - 6 classes

## ✨ Features

- ✅ **Multi-Model Support**: 3 specialized medical AI models
- ✅ **Out-of-Domain Detection**: Automatic confidence-based filtering  
- ✅ **Advanced Architectures**: ResNet18+ViT+CBAM, MobileNetV2
- ✅ **Real-time Inference**: Fast predictions with GPU/CPU support
- ✅ **User-friendly Interface**: Beautiful Gradio web interface
- ✅ **Production Ready**: Docker deployment with health checks

## 🚀 Usage

1. **Upload a medical image** (skin lesion, teeth, or nail condition)
2. **Select the appropriate AI model** for your image type
3. **Get instant predictions** with confidence scores
4. **Review results** with out-of-domain detection

## 📊 Model Specifications

### DermNet (Skin Diseases)
- **Architecture**: ResNet18 + Vision Transformer + CBAM Attention
- **Classes**: 14 skin conditions (Acne, Melanoma, Eczema, etc.)
- **Confidence Threshold**: 40%

### Teeth (Dental Conditions)  
- **Architecture**: MobileNetV2
- **Classes**: 5 dental conditions (Calculus, Caries, etc.)
- **Confidence Threshold**: 60%

### Nail (Nail Disorders)
- **Architecture**: MobileNetV2  
- **Classes**: 6 nail conditions (Melanoma, Clubbing, etc.)
- **Confidence Threshold**: 70%

## 🔧 Technical Details

- **Frameworks**: PyTorch, Gradio, Docker
- **Image Processing**: 224x224 RGB, ImageNet normalization
- **Attention Mechanism**: CBAM with spatial kernel size 3
- **Performance**: ~0.2-0.5s inference time
- **Error Handling**: Comprehensive validation and error reporting

## ⚠️ Medical Disclaimer

This AI system is for **educational and research purposes only**. It should not be used as a substitute for professional medical diagnosis or treatment. Always consult qualified healthcare professionals for medical decisions.

## 📚 More Information

- **API Documentation**: See `API_DOCS.md` for REST API usage
- **Quick Reference**: Check `QUICK_REFERENCE.md` for development
- **Source Code**: Available on [GitHub](https://github.com/thuongnvv/devfest2025-AI)

*Created for DevFest 2025 AI Challenge*
