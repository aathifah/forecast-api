from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io
import base64
from datetime import datetime
import logging
from forecast_service import process_forecast

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Forecast API",
    description="API untuk forecasting dengan machine learning",
    version="1.0.0"
)

@app.get("/")
async def root():
    return {
        "message": "Forecast API is running!",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "forecast": "/forecast",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "forecast-api"
    }

@app.post("/forecast")
async def forecast_endpoint(file: UploadFile = File(...)):
    """
    Endpoint untuk melakukan forecasting dari file Excel
    """
    try:
        logger.info(f"Received file: {file.filename}")
        
        # Validasi file
        if not file.filename.endswith(('.xlsx', '.xls')):
            raise HTTPException(
                status_code=400, 
                detail="File harus berformat Excel (.xlsx atau .xls)"
            )
        
        # Baca file Excel
        content = await file.read()
        df = pd.read_excel(io.BytesIO(content))
        
        logger.info(f"File loaded successfully. Shape: {df.shape}")
        
        # Validasi kolom yang diperlukan
        required_columns = ['MONTH', 'PART_NO', 'ORIGINAL_SHIPPING_QTY']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Kolom yang diperlukan tidak ditemukan: {missing_columns}"
            )
        
        # Proses forecasting
        result = process_forecast(df)
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=result["message"]
            )
        
        logger.info("Forecast processing completed successfully")
        
        return JSONResponse(
            content=result,
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /forecast: {str(e)}") # Lebih spesifik untuk logging
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/forecast-base64")
async def forecast_base64_endpoint(data: dict):
    """
    Endpoint untuk forecasting dengan data base64 (untuk Power Automate)
    """
    try:
        logger.info("Received base64 data request")
        
        # Validasi input
        if "excel_base64" not in data:
            raise HTTPException(
                status_code=400,
                detail="Parameter 'excel_base64' diperlukan"
            )
        
        # Decode base64
        try:
            # --- PERUBAHAN PENTING DI SINI ---
            # Meng-encode string base64 ke bytes sebelum mendekode
            excel_content = base64.b64decode(data["excel_base64"].encode('utf-8'))
            # --- AKHIR PERUBAHAN ---

        except Exception as e:
            # Menangkap error spesifik untuk decoding base64
            logger.error(f"Base64 decoding error: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 data: {str(e)}"
            )
        
        # Baca file Excel
        df = pd.read_excel(io.BytesIO(excel_content))
        
        logger.info(f"File loaded successfully. Shape: {df.shape}")
        
        # Validasi kolom
        required_columns = ['MONTH', 'PART_NO', 'ORIGINAL_SHIPPING_QTY']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise HTTPException(
                status_code=400,
                detail=f"Kolom yang diperlukan tidak ditemukan: {missing_columns}"
            )
        
        # Proses forecasting
        result = process_forecast(df)
        
        if result["status"] == "error":
            raise HTTPException(
                status_code=500,
                detail=result["message"]
            )
        
        logger.info("Forecast processing completed successfully")
        
        return JSONResponse(
            content=result,
            status_code=200,
            headers={
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /forecast-base64: {str(e)}") # Lebih spesifik untuk logging
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
