#!/bin/bash

echo "🧪 Testing application locally..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed"
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "❌ pip is not installed"
    exit 1
fi

echo "✅ Python and pip are available"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv_test

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv_test/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements_streamlit_simple.txt

# Test imports
echo "🔍 Testing imports..."
python3 -c "
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')
print('✅ All imports successful!')
"

# Test forecast service
echo "🔍 Testing forecast service..."
python3 -c "
try:
    from forecast_service_alt import forecast_ma6, forecast_wma, forecast_ets, forecast_arima
    print('✅ Forecast service imports successful!')
    
    # Test with sample data
    import numpy as np
    sample_data = [10, 15, 12, 18, 20, 16]
    
    ma6_result = forecast_ma6(sample_data)
    wma_result = forecast_wma(sample_data)
    ets_result = forecast_ets(sample_data)
    arima_result = forecast_arima(sample_data)
    
    print(f'MA6: {ma6_result}')
    print(f'WMA: {wma_result}')
    print(f'ETS: {ets_result}')
    print(f'ARIMA: {arima_result}')
    print('✅ All forecast functions working!')
    
except Exception as e:
    print(f'❌ Error in forecast service: {e}')
"

# Test Streamlit app
echo "🔍 Testing Streamlit app..."
python3 -c "
try:
    import streamlit_app_simple
    print('✅ Streamlit app imports successful!')
except Exception as e:
    print(f'❌ Error in Streamlit app: {e}')
"

echo ""
echo "🎉 Local testing completed!"
echo ""
echo "📋 To run Streamlit app locally:"
echo "source venv_test/bin/activate"
echo "streamlit run streamlit_app_simple.py"
echo ""
echo "📋 To clean up:"
echo "rm -rf venv_test" 
