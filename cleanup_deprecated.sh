#!/bin/bash
# Cleanup deprecated files

echo "🗑️ Removing deprecated model files..."
rm -f best_medagen_resnet18_vits_cbam.pth
rm -f best_teeth_model.pth
rm -f best_nail_model.pth
rm -f best_model_mobilenet_teeth.pth
rm -f mobilenet_nail_disease.pth

echo "🗑️ Removing old code files..."
rm -f api.py
rm -f streamlit_app.py
rm -f test_api_client.py
rm -f test_real_images.py
rm -f deploy.sh
rm -f deploy_hf.sh
rm -f start_api.sh
rm -f run_api.sh

echo "🗑️ Removing old notebooks..."
rm -f train-dermnet-small-dataset.ipynb
rm -f teeth-disease-model.ipynb
rm -f nail-disease-model.ipynb
rm -f "nail-disease-model(1).ipynb"
rm -f test_dermnet_inference.ipynb
rm -f test_teeth_inference.ipynb
rm -f test_nail_inference.ipynb

echo "🗑️ Removing old docs..."
rm -f API_DOCS.md
rm -f QUICK_REFERENCE.md

echo "✅ Cleanup complete!"
