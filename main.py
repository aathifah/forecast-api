from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
import base64
from datetime import datetime
import logging
from pydantic import BaseModel # Import BaseModel dari pydantic
from forecast_service import process_forecast

# Setup logging
# Konfigurasi logging agar lebih detail, termasuk waktu dan nama logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Definisikan Pydantic Model untuk request body endpoint /forecast-base64
# Ini akan memastikan validasi input JSON secara otomatis
class ForecastRequest(BaseModel):
    excel_base64: str

app = FastAPI(
    title="Forecast API",
    description="API untuk forecasting dengan machine learning",
    version="1.0.0"
)

@app.get("/")
async def root():
    """
    Root endpoint untuk mengecek status API.
    """
    logger.info("Root endpoint accessed.")
    return {
        "message": "Forecast API is running!",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "forecast": "/forecast",
            "forecast_base64": "/forecast-base64", # Tambahkan ini untuk kejelasan
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """
    Endpoint untuk health check aplikasi.
    """
    logger.info("Health check endpoint accessed.")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "forecast-api"
    }

@app.post("/forecast")
async def forecast_endpoint(file: UploadFile = File(...)):
    """
    Endpoint untuk melakukan forecasting dari file Excel yang diunggah langsung (multipart/form-data).
    """
    try:
        logger.info(f"Received file upload request for: {file.filename}, content_type: {file.content_type}")
        logger.info(f"Received request with method: {request.method} to path: {request.url.path}") # Logging metode
        
        # Validasi tipe file
        if not file.filename.endswith(('.xlsx', '.xls')):
            logger.warning(f"Invalid file format received: {file.filename}")
            raise HTTPException(
                status_code=400, 
                detail="File harus berformat Excel (.xlsx atau .xls)"
            )
        
        # Baca file Excel
        # Gunakan await file.read() untuk membaca konten file asinkron
        content = await file.read()
        
        # Pastikan file tidak kosong
        if not content:
            logger.warning("Received an empty Excel file.")
            raise HTTPException(
                status_code=400,
                detail="File Excel tidak boleh kosong."
            )

        df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        
        logger.info(f"File '{file.filename}' loaded successfully. Shape: {df.shape}")
        
        # Validasi kolom yang diperlukan
        required_columns = ['MONTH', 'PART_NO', 'ORIGINAL_SHIPPING_QTY']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            raise HTTPException(
                status_code=400,
                detail=f"Kolom yang diperlukan tidak ditemukan: {missing_columns}"
            )
        
        # Proses forecasting
        logger.info("Starting forecast processing...")
        result = process_forecast(df)
        
        if result["status"] == "error":
            logger.error(f"Forecast processing failed: {result.get('message', 'Unknown error')}")
            raise HTTPException(
                status_code=500,
                detail=result["message"]
            )
        
        logger.info("Forecast processing completed successfully.")
        
        return JSONResponse(
            content=result,
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*" # Penting untuk CORS jika diakses dari domain lain
            }
        )
        
    except HTTPException:
        # Re-raise HTTPException karena sudah ditangani sebelumnya
        raise
    except pd.errors.EmptyDataError:
        logger.error("Pandas EmptyDataError: Excel file is empty or malformed.")
        raise HTTPException(
            status_code=400,
            detail="File Excel kosong atau tidak valid."
        )
    except Exception as e:
        logger.error(f"Unexpected error in /forecast endpoint: {str(e)}", exc_info=True) # exc_info=True untuk traceback
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error processing file: {str(e)}"
        )

@app.post("/forecast-base64")
async def forecast_base64_endpoint(request_body: ForecastRequest): # Menggunakan Pydantic model untuk validasi input
    """
    Endpoint untuk forecasting dengan data Excel yang dikirim sebagai base64 string dalam body JSON.
    Cocok untuk integrasi dengan Power Automate.
    """
    try:
        logger.info("Received base64 data request.")
        
        # Mengakses string base64 dari Pydantic model
        excel_base64_str = request_body.excel_base64
        
        # Logging string base64 yang diterima untuk debugging
        logger.info(f"Received base64 string length: {len(excel_base64_str)}")
        if len(excel_base64_str) > 100: # Batasi logging untuk string yang sangat panjang
            logger.info(f"Received base64 string start: {excel_base64_str[:50]}...{excel_base64_str[-50:]}")
        else:
            logger.info(f"Received base64 string: {excel_base64_str}")

        try:
            # PENTING: Meng-encode string base64 ke bytes sebelum mendekode
            # base64.b64decode() mengharapkan input bertipe bytes, bukan str
            excel_content = base64.b64decode(excel_base64_str.encode('utf-8'))
            logger.info(f"Base64 data successfully decoded. Content length: {len(excel_content)} bytes.")
        except base64.binascii.Error as e:
            # Menangkap error khusus base64 decoding (misal: incorrect padding, non-base64 characters)
            logger.error(f"Base64 decoding error (binascii.Error): {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 data: {str(e)}. Please ensure the base64 string is valid and correctly padded."
            )
        except Exception as e:
            # Menangkap error umum lainnya selama decoding
            logger.error(f"Unexpected error during base64 decoding: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=400,
                detail=f"Failed to decode base64 string: {str(e)}"
            )
        
        # Baca file Excel dari konten biner yang sudah didecode
        # Pastikan konten tidak kosong setelah decode
        if not excel_content:
            logger.warning("Decoded base64 content is empty.")
            raise HTTPException(
                status_code=400,
                detail="Konten file Excel yang didecode dari base64 kosong atau tidak valid."
            )

        with open("debug_received.xlsx", "wb") as f:
            f.write(excel_content)
        df = pd.read_excel(io.BytesIO(excel_content), engine="openpyxl")
        
        logger.info(f"File from base64 loaded successfully into DataFrame. Shape: {df.shape}")
        
        # Validasi kolom yang diperlukan
        required_columns = ['MONTH', 'PART_NO', 'ORIGINAL_SHIPPING_QTY']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing required columns in base64 Excel: {missing_columns}")
            raise HTTPException(
                status_code=400,
                detail=f"Kolom yang diperlukan tidak ditemukan: {missing_columns}"
            )
        
        # Proses forecasting
        logger.info("Starting forecast processing from base64 data...")
        result = process_forecast(df)
        
        if result["status"] == "error":
            logger.error(f"Forecast processing from base64 failed: {result.get('message', 'Unknown error')}")
            raise HTTPException(
                status_code=500,
                detail=result["message"]
            )
        
        logger.info("Forecast processing from base64 completed successfully.")
        
        return JSONResponse(
            content=result,
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except HTTPException:
        # Re-raise HTTPException karena sudah ditangani sebelumnya oleh FastAPI/Pydantic
        raise
    except pd.errors.EmptyDataError:
        logger.error("Pandas EmptyDataError: Decoded Excel file is empty or malformed.")
        raise HTTPException(
            status_code=400,
            detail="File Excel yang didecode kosong atau tidak valid."
        )
    except Exception as e:
        logger.error(f"Unexpected error in /forecast-base64 endpoint: {str(e)}", exc_info=True) # exc_info=True untuk traceback
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error processing base64 data: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    # Sesuaikan port jika Railway menggunakan environment variable PORT
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
