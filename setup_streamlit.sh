#!/bin/bash

echo "🔧 Setting up Streamlit Cloud environment..."

# Update package list
apt-get update

# Install system dependencies
echo "📦 Installing system dependencies..."
apt-get install -y gcc g++ build-essential python3-dev libopenblas-dev liblapack-dev

# Install Python dependencies in specific order
echo "🐍 Installing Python dependencies..."

# Upgrade pip first
pip install --upgrade pip

# Install core dependencies first
echo "📦 Installing numpy..."
pip install "numpy==2.3.2"

echo "📦 Installing pandas..."
pip install "pandas==2.3.1"

echo "📦 Installing scipy..."
pip install "scipy==1.13.1"

echo "📦 Installing statsmodels..."
pip install "statsmodels==0.15.1"

# Install pmdarima
echo "📊 Installing pmdarima..."
pip install "pmdarima==2.0.4"

# Install remaining dependencies
echo "📋 Installing remaining dependencies..."
pip install "scikit-learn==1.5.2"
pip install "xgboost==2.2.1"
pip install "openpyxl==3.1.2"
pip install "joblib==1.4.2"
pip install "tqdm==4.66.4"
pip install "plotly==5.17.0"
pip install "streamlit==1.47.1"

echo "✅ Setup completed successfully!" 
