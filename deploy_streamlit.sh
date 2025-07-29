#!/bin/bash

echo "🚀 Preparing robust deployment for Streamlit Cloud..."

# Backup original files
echo "📦 Backing up original files..."
cp streamlit_app.py streamlit_app_backup.py
cp requirements_streamlit.txt requirements_streamlit_backup.txt

# Update .streamlit/config.toml
echo "⚙️ Updating Streamlit configuration..."
cat > .streamlit/config.toml << EOF
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"

[server]
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 200
EOF

# Update runtime.txt
echo "📝 Updating runtime.txt..."
echo "python-3.13.5" > runtime.txt

# Update packages.txt
echo "📦 Updating packages.txt..."
cat > packages.txt << EOF
gcc
g++
build-essential
python3-dev
libopenblas-dev
liblapack-dev
EOF

# Update requirements_streamlit.txt with exact versions
echo "📋 Updating requirements_streamlit.txt..."
cat > requirements_streamlit.txt << EOF
streamlit==1.47.1
numpy==2.3.2
pandas==2.3.1
scikit-learn==1.5.2
xgboost==2.2.1
scipy==1.13.1
statsmodels==0.15.1
pmdarima==2.0.4
openpyxl==3.1.2
joblib==1.4.2
tqdm==4.66.4
plotly==5.17.0
EOF

# Create setup.sh
echo "🔧 Creating setup.sh..."
cat > setup.sh << 'EOF'
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
EOF

# Make setup.sh executable
chmod +x setup.sh

# Git operations
echo "📝 Committing changes..."
git add .
git commit -m "Robust deployment: Python 3.13.5 + exact versions + setup.sh"

echo "🚀 Pushing to GitHub..."
git push origin main

echo "✅ Robust deployment preparation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Go to https://share.streamlit.io"
echo "2. Connect your GitHub repository"
echo "3. Set main file to: streamlit_app.py"
echo "4. Set requirements file to: requirements_streamlit.txt"
echo "5. Deploy!"
echo ""
echo "🔧 Configuration details:"
echo "- Python version: 3.13.5 (matching Streamlit Cloud)"
echo "- Streamlit version: 1.47.1"
echo "- All dependencies: exact versions for stability"
echo "- pmdarima: 2.0.4 (latest stable)"
echo "- plotly: 5.17.0 (for visualizations)"
echo "- Build tools: gcc, g++, build-essential, python3-dev"
echo "- Math libraries: libopenblas-dev, liblapack-dev"
echo ""
echo "🔄 To restore backup files later:"
echo "cp streamlit_app_backup.py streamlit_app.py"
echo "cp requirements_streamlit_backup.txt requirements_streamlit.txt" 
