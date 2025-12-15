#!/bin/bash

# QNA-Auth Setup Script
# This script sets up the QNA-Auth environment

set -e

echo "=========================================="
echo "QNA-Auth Setup Script"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python --version

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p dataset/samples
mkdir -p auth/device_embeddings
mkdir -p model/checkpoints

# Copy example config
echo ""
echo "Copying example configuration..."
if [ ! -f config.py ]; then
    cp config.example.py config.py
    echo "✓ config.py created (please review and adjust settings)"
else
    echo "config.py already exists"
fi

# Test installations
echo ""
echo "Testing installations..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"

echo ""
echo "=========================================="
echo "✓ Backend setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review and adjust settings in config.py"
echo "2. Start backend: python server/app.py"
echo "3. Setup frontend: cd frontend && npm install"
echo "4. Start frontend: cd frontend && npm run dev"
echo ""
