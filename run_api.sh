#!/bin/bash

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if model file exists
if [ ! -f "best_medagen_resnet18_vits_cbam.pth" ]; then
    echo "❌ Model file not found: best_medagen_resnet18_vits_cbam.pth"
    echo "Please make sure the model file is in the current directory."
    exit 1
fi

echo "🚀 Starting DermNet AI API..."
echo "The API will be available at a public ngrok URL."
echo "Press Ctrl+C to stop the server."
echo ""

python app.py