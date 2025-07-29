#!/bin/bash

echo "🔧 Fixing plotly installation for Streamlit Cloud..."

# Update package list
apt-get update

# Install system dependencies
echo "📦 Installing system dependencies..."
apt-get install -y gcc g++ build-essential python3-dev libopenblas-dev liblapack-dev python3-pip python3-setuptools

# Install Python dependencies in specific order
echo "🐍 Installing Python dependencies..."

# Upgrade pip first
pip install --upgrade pip setuptools wheel

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

# Force install plotly
echo "📊 Force installing plotly..."
pip install "plotly==5.17.0" --force-reinstall --no-cache-dir

# Install streamlit last
echo "📱 Installing streamlit..."
pip install "streamlit==1.47.1"

# Verify plotly installation
echo "🔍 Verifying plotly installation..."
python3 -c "import plotly; print('✅ plotly imported successfully')"
python3 -c "import plotly.express; print('✅ plotly.express imported successfully')"

echo "✅ Setup completed successfully!" 
