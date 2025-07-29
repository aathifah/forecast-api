# 📊 Forecast Parts API

Aplikasi forecasting dengan machine learning untuk prediksi permintaan part number.

## 🚀 Deployment Options

### 1. Streamlit Cloud (REKOMENDASI)
- **100% GRATIS** untuk unlimited users
- **Auto-scaling** otomatis
- **Machine Learning** libraries support
- **Beautiful UI** dengan Plotly charts

### 2. Fly.io
- **Free tier** tersedia
- **Global CDN**
- **Custom domains**

### 3. Railway
- **$5 credit** gratis per bulan
- **Fast deployment**
- **Auto-scaling**

## 📁 File Structure

```
forecast-api/
├── main.py                          # FastAPI application
├── streamlit_app.py                 # Streamlit application
├── forecast_service.py              # Forecasting logic
├── requirements.txt                 # FastAPI dependencies
├── requirements_streamlit.txt       # Streamlit dependencies
├── index.html                      # Frontend landing page
├── script.js                       # Frontend JavaScript
├── style.css                       # Frontend CSS
├── Procfile                        # Heroku deployment
├── runtime.txt                     # Python version
├── .streamlit/
│   └── config.toml                # Streamlit configuration
└── README.md                       # This file
```

## 🔧 Setup Development

### Local Development
```bash
# Clone repository
git clone https://github.com/aathifah/forecast-api.git
cd forecast-api

# Install dependencies
pip install -r requirements.txt

# Run FastAPI locally
uvicorn main:app --reload

# Run Streamlit locally
streamlit run streamlit_app.py
```

### Streamlit Deployment
1. Push ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Set path: `streamlit_app.py`
5. Deploy!

### Fly.io Deployment
```bash
# Install Fly CLI
curl -L https://fly.io/install.sh | sh

# Login
fly auth login

# Deploy
fly launch
```

## 📊 Features

### Forecasting Methods
- **MA6**: Moving Average 6 bulan
- **WMA**: Weighted Moving Average
- **ETS**: Exponential Smoothing
- **ARIMA**: Auto-regressive model
- **ML Models**: Linear Regression, Random Forest, XGBoost

### API Endpoints
- `GET /` - Landing page
- `GET /api/health` - Health check
- `POST /forecast` - Upload Excel untuk forecasting
- `POST /forecast-base64` - Forecasting dengan base64
- `POST /process-forecast` - Process dengan dashboard
- `GET /download-forecast` - Download hasil

### Streamlit Features
- 📁 **File Upload** dengan validasi
- 📊 **Interactive Dashboard** dengan Plotly
- 📈 **Real-time Charts** dan visualisasi
- 📋 **Data Preview** sebelum processing
- 🎯 **Multiple Tabs**: Forecast, Backtest, Performance
- 📥 **Excel Download** hasil forecasting
- 🎨 **Beautiful UI** dengan custom CSS

## 📋 Data Requirements

File Excel harus memiliki kolom:
- `PART_NO` (Part Number)
- `MONTH` (Bulan dalam format YYYY-MM)
- `ORIGINAL_SHIPPING_QTY` (Jumlah permintaan)

## 🔍 Usage Examples

### FastAPI Usage
```bash
# Test health check
curl https://your-app.fly.dev/api/health

# Upload file
curl -X POST -F "file=@data.xlsx" https://your-app.fly.dev/forecast
```

### Streamlit Usage
1. Upload file Excel
2. Klik "Mulai Forecasting"
3. Lihat hasil di dashboard
4. Download Excel results

## 🛠️ Technologies

- **Backend**: FastAPI, Python 3.11
- **Frontend**: HTML, CSS, JavaScript
- **ML**: scikit-learn, XGBoost, statsmodels
- **Charts**: Plotly, Chart.js
- **Deployment**: Streamlit Cloud, Fly.io, Railway

## 📈 Performance

- **Auto-scaling** untuk handle banyak users
- **Memory optimization** untuk large datasets
- **Parallel processing** untuk forecasting
- **Caching** untuk hasil komputasi

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

MIT License - feel free to use for commercial projects.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/aathifah/forecast-api/issues)
- **Documentation**: [API Docs](https://your-app.fly.dev/docs)
- **Demo**: [Streamlit App](https://your-app.streamlit.app)

---

Built with ❤️ using FastAPI & Streamlit | Machine Learning Forecasting App 
