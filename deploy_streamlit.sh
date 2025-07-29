#!/bin/bash

echo "🚀 Preparing deployment for Streamlit Cloud (Fixed for Python 3.13.5)..."

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

# Update requirements_streamlit.txt
echo "📋 Updating requirements_streamlit.txt..."
cat > requirements_streamlit.txt << EOF
streamlit==1.47.1
numpy>=2.0.0
pandas>=2.0.0
scikit-learn>=1.3.0
xgboost>=2.0.0
scipy>=1.11.0
statsmodels>=0.14.0
pmdarima>=2.0.0
openpyxl>=3.1.0
joblib>=1.3.0
tqdm>=4.66.0
plotly>=5.17.0
EOF

# Git operations
echo "📝 Committing changes..."
git add .
git commit -m "Fix deployment: Python 3.13.5 + all dependencies"

echo "🚀 Pushing to GitHub..."
git push origin main

echo "✅ Deployment preparation complete!"
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
echo "- Streamlit version: 1.47.1 (latest)"
echo "- All dependencies: numpy, pandas, plotly, pmdarima, etc."
echo "- Build tools: gcc, g++, build-essential, python3-dev"
echo "- Math libraries: libopenblas-dev, liblapack-dev"
echo ""
echo "🔄 To restore backup files later:"
echo "cp streamlit_app_backup.py streamlit_app.py"
echo "cp requirements_streamlit_backup.txt requirements_streamlit.txt" 
