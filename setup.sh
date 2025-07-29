#!/bin/bash

# Update package list
apt-get update

# Install system dependencies
apt-get install -y gcc g++ build-essential python3-dev

# Install Python dependencies
pip install --upgrade pip
pip install numpy>=1.26.0
pip install -r requirements_streamlit.txt 
