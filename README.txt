# 🚀 Forecast API - Railway Deployment

## 📋 Overview

Forecast API adalah aplikasi FastAPI yang dapat di-deploy ke Railway untuk melakukan forecasting dengan machine learning. API ini menerima file Excel dan mengembalikan hasil forecast dalam format Excel.

## 🏗️ Struktur Project

```
forecast-api/
├── main.py                 # FastAPI application
├── forecast_service.py     # Core forecasting logic
├── requirements.txt        # Python dependencies
├── Procfile               # Railway deployment config
├── runtime.txt            # Python version
├── README_DEPLOYMENT.md   # This file
└── .gitignore            # Git ignore file
```

## 🚀 Deployment ke Railway

### 1. **Persiapkan Repository GitHub**

1. Buat repository baru di GitHub
2. Upload semua file ke repository:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Forecast API"
   git branch -M main
   git remote add origin https://github.com/username/forecast-api.git
   git push -u origin main
   ```

### 2. **Deploy ke Railway**

1. Buka [Railway.app](https://railway.app)
2. Login dengan GitHub
3. Klik "New Project"
4. Pilih "Deploy from GitHub repo"
5. Pilih repository yang sudah dibuat
6. Railway akan otomatis detect Python project dan deploy

### 3. **Environment Variables (Opsional)**

Di Railway dashboard, tambahkan environment variables jika diperlukan:
- `PORT`: Railway akan set otomatis
- `LOG_LEVEL`: `INFO` (default)

## 📡 API Endpoints

### 1. **Health Check**
```
GET /
GET /health
```

### 2. **Forecast dengan File Upload**
```
POST /forecast
Content-Type: multipart/form-data
Body: file (Excel file)
```

### 3. **Forecast dengan Base64 (Power Automate)**
```
POST /forecast-base64
Content-Type: application/json
Body: {"excel_base64": "base64_string"}
```

## 🔧 Power Automate Integration

### 1. **HTTP Request Action**

```json
{
  "method": "POST",
  "uri": "https://your-railway-app.railway.app/forecast-base64",
  "headers": {
    "Content-Type": "application/json"
  },
  "body": {
    "excel_base64": "@{base64(variables('excel_content'))}"
  }
}
```

### 2. **Response Processing**

```json
{
  "status": "success",
  "message": "Forecast completed successfully",
  "data": {
    "excel_base64": "base64_encoded_excel",
    "total_forecasts": 1000,
    "total_parts": 50,
    "forecast_months": ["2024-01", "2024-02", "2024-03", "2024-04"],
    "timestamp": "2024-01-15T10:30:00"
  }
}
```

### 3. **Excel Output**

Gunakan action "Create file" di Power Automate:
- **File Name**: `forecast_results_@{utcNow('yyyyMMdd_HHmmss')}.xlsx`
- **File Content**: `@{base64ToBinary(body('HTTP_Request')?['data']?['excel_base64'])}`

## 📊 Format Input Excel

File Excel harus memiliki kolom berikut:
- `MONTH`: Format YYYYMM (contoh: 202401)
- `PART_NO`: Nomor part
- `ORIGINAL_SHIPPING_QTY`: Quantity shipping
- `PART_NAME`: Nama part (opsional)
- `TOPAS_ORDER_TYPE`: Tipe order (opsional)
- `CREATED_DEMAND_FLAG`: Flag demand (opsional)
- `CUST_TYPE2`: Tipe customer (opsional)

## 📈 Format Output Excel

File output akan memiliki 3 sheet:

### 1. **Forecast_Results**
- `PART_NO`: Nomor part
- `MONTH`: Bulan forecast
- `BEST_MODEL`: Model terbaik
- `FORECAST_OPTIMIST`: Forecast optimis (+15%)
- `FORECAST_NEUTRAL`: Forecast netral
- `FORECAST_PESSIMIST`: Forecast pesimis (-20%)
- `ERROR_BACKTEST`: Error dari backtest

### 2. **Part_Summary**
- Ringkasan per part dengan statistik

### 3. **Error_Analysis**
- Analisis error per part

## 🧪 Testing

### 1. **Local Testing**
```bash
pip install -r requirements.txt
python main.py
```

### 2. **API Testing**
```bash
# Health check
curl https://your-railway-app.railway.app/health

# Forecast dengan file
curl -X POST -F "file=@dataset.xlsx" https://your-railway-app.railway.app/forecast
```

## 🔍 Monitoring

### 1. **Railway Dashboard**
- Monitor logs di Railway dashboard
- Check resource usage (CPU, Memory)
- View deployment status

### 2. **API Monitoring**
- Response time
- Error rates
- Success rates

## 🚨 Troubleshooting

### 1. **Deployment Issues**
- Check `requirements.txt` compatibility
- Verify Python version di `runtime.txt`
- Check Railway logs

### 2. **Runtime Issues**
- Monitor memory usage (ML models membutuhkan RAM tinggi)
- Check timeout settings
- Verify file upload limits

### 3. **Common Errors**
- `MemoryError`: Kurangi `n_jobs` di Parallel processing
- `TimeoutError`: Increase timeout di Power Automate
- `FileNotFoundError`: Check file path dan permissions

## 📝 Logs

API akan mencatat logs untuk:
- File upload
- Processing progress
- Errors dan exceptions
- Performance metrics

## 🔒 Security

- API tidak menyimpan file secara permanen
- Semua processing dilakukan di memory
- No authentication required (sesuaikan kebutuhan)

## 📞 Support

Jika ada masalah:
1. Check Railway logs
2. Verify input format
3. Test dengan file kecil terlebih dahulu
4. Contact developer jika diperlukan

---

**🎉 Selamat! Forecast API Anda sudah siap di-deploy ke Railway!**
