from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Query
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
import pandas as pd
import io
import base64
from datetime import datetime
import logging
from pydantic import BaseModel
from forecast_service import process_forecast, run_real_time_forecast, run_combined_forecast
from fastapi.staticfiles import StaticFiles
import os
import uuid
import tempfile
import numpy as np

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

# Serve static files (frontend) on /static
app.mount("/static", StaticFiles(directory=".", html=True), name="static")

# Buat direktori untuk menyimpan file hasil forecast
TEMP_DIR = os.path.join(tempfile.gettempdir(), "forecast_results")
os.makedirs(TEMP_DIR, exist_ok=True)
logger.info(f"Temporary directory created: {TEMP_DIR}")

@app.get("/")
async def root():
    return FileResponse("index.html")

@app.get("/api/health")
async def health_check():
    logger.info("Health check endpoint accessed.")
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "forecast-api"
    }

@app.post("/forecast")
async def forecast_endpoint(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file upload request for: {file.filename}, content_type: {file.content_type}")
        # Validasi tipe file - hanya .xlsx yang diterima
        if not file.filename.lower().endswith(".xlsx"):
            logger.warning(f"Invalid file format received: {file.filename}")
            raise HTTPException(
                status_code=400, 
                detail="Tipe File Kamu Bukan Excel. Pastikan File Tipe Excel Saja yang Kamu Unggah!"
            )
        content = await file.read()
        if not content:
            logger.warning("Received an empty Excel file.")
            raise HTTPException(
                status_code=400,
                detail="File Excel tidak boleh kosong."
            )
        df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        logger.info(f"File '{file.filename}' loaded successfully. Shape: {df.shape}")
        required_columns = ['MONTH', 'PART_NO', 'ORIGINAL_SHIPPING_QTY']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            raise HTTPException(
                status_code=400,
                detail=f"Kolom yang diperlukan tidak ditemukan: {missing_columns}"
            )
        logger.info("Starting forecast processing...")
        result = process_forecast(df)
        # Tambahkan hasil forecast detail ke response JSON
        if result["status"] == "success":
            _, forecast_df = run_combined_forecast(df)
            real_time_forecast = run_real_time_forecast(_, forecast_df)
            result["data"]["forecast_results"] = real_time_forecast.to_dict(orient="records")
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
                "Access-Control-Allow-Origin": "*"
            }
        )
    except HTTPException:
        raise
    except pd.errors.EmptyDataError:
        logger.error("Pandas EmptyDataError: Excel file is empty or malformed.")
        raise HTTPException(
            status_code=400,
            detail="File Excel kosong atau tidak valid."
        )
    except Exception as e:
        logger.error(f"Unexpected error in /forecast endpoint: {str(e)}", exc_info=True)
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

@app.post("/forecast-raw")
async def forecast_raw(request: Request):
    """
    Endpoint untuk menerima file Excel sebagai raw body (application/octet-stream),
    cocok untuk Power Automate Web yang tidak mendukung multipart/form-data.
    """
    try:
        logger.info("Received raw body request for Excel file.")
        content = await request.body()  # baca seluruh body sebagai bytes
        logger.info(f"Raw body length: {len(content)} bytes.")

        # Pastikan file tidak kosong
        if not content:
            logger.warning("Received an empty Excel file (raw body).")
            raise HTTPException(
                status_code=400,
                detail="File Excel tidak boleh kosong."
            )

        # Simpan file untuk debug (opsional)
        with open("debug_received_raw.xlsx", "wb") as f:
            f.write(content)

        df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        logger.info(f"Raw body Excel loaded successfully. Shape: {df.shape}")

        # Validasi kolom yang diperlukan
        required_columns = ['MONTH', 'PART_NO', 'ORIGINAL_SHIPPING_QTY']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns in raw Excel: {missing_columns}")
            raise HTTPException(
                status_code=400,
                detail=f"Kolom yang diperlukan tidak ditemukan: {missing_columns}"
            )

        # Proses forecasting
        logger.info("Starting forecast processing from raw body...")
        result = process_forecast(df)
        if result["status"] == "error":
            logger.error(f"Forecast processing from raw body failed: {result.get('message', 'Unknown error')}")
            raise HTTPException(
                status_code=500,
                detail=result["message"]
            )
        logger.info("Forecast processing from raw body completed successfully.")
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
    except pd.errors.EmptyDataError:
        logger.error("Pandas EmptyDataError: Excel file (raw body) is empty or malformed.")
        raise HTTPException(
            status_code=400,
            detail="File Excel kosong atau tidak valid."
        )
    except Exception as e:
        logger.error(f"Unexpected error in /forecast-raw endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error processing raw Excel file: {str(e)}"
        )

@app.post("/process-forecast")
async def process_forecast_endpoint(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file upload request for process: {file.filename}, content_type: {file.content_type}")
        # Validasi tipe file - hanya .xlsx yang diterima
        if not file.filename.lower().endswith(".xlsx"):
            logger.warning(f"Invalid file format received: {file.filename}")
            raise HTTPException(
                status_code=400, 
                detail="Tipe File Kamu Bukan Excel. Pastikan File Tipe Excel Saja yang Kamu Unggah!"
            )
        content = await file.read()
        if not content:
            logger.warning("Received an empty Excel file.")
            raise HTTPException(
                status_code=400,
                detail="File Excel tidak boleh kosong."
            )
        df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        logger.info(f"File '{file.filename}' loaded successfully for process. Shape: {df.shape}")
        required_columns = ['MONTH', 'PART_NO', 'ORIGINAL_SHIPPING_QTY']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing required columns: {missing_columns}")
            raise HTTPException(
                status_code=400,
                detail=f"Kolom yang diperlukan tidak ditemukan: {missing_columns}"
            )
        logger.info("Starting forecast processing for process...")
        df_processed, forecast_df = run_combined_forecast(df)
        real_time_forecast = run_real_time_forecast(df_processed, forecast_df)
        # Buat file Excel hasil
        file_id = str(uuid.uuid4())
        output_path = os.path.join(TEMP_DIR, f"forecast_result_{file_id}.xlsx")
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            forecast_df.to_excel(writer, sheet_name='Backtest', index=False)
            real_time_forecast.to_excel(writer, sheet_name='RealTimeForecast', index=False)
        logger.info(f"Forecast Excel file created successfully at {output_path}.")
        # Bersihkan NaN/inf di semua dataframe sebelum dikirim ke frontend
        df_processed = df_processed.replace([np.inf, -np.inf], np.nan).fillna(0)
        forecast_df = forecast_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        real_time_forecast = real_time_forecast.replace([np.inf, -np.inf], np.nan).fillna(0)
        # Return JSON for dashboard integration
        return {
            "status": "success",
            "file_id": file_id,
            "original_df": df_processed.to_dict(orient="records"),
            "forecast_df": forecast_df.to_dict(orient="records"),
            "real_time_forecast": real_time_forecast.to_dict(orient="records")
        }
    except HTTPException:
        raise
    except pd.errors.EmptyDataError:
        logger.error("Pandas EmptyDataError: Excel file is empty or malformed.")
        raise HTTPException(
            status_code=400,
            detail="File Excel kosong atau tidak valid."
        )
    except Exception as e:
        logger.error(f"Unexpected error in /process-forecast endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error processing file: {str(e)}"
        )

@app.get("/download-forecast")
async def download_forecast_endpoint(file_id: str = Query(...)):
    try:
        output_path = os.path.join(TEMP_DIR, f"forecast_result_{file_id}.xlsx")
        if not os.path.exists(output_path):
            raise HTTPException(status_code=404, detail="File hasil tidak ditemukan atau sudah expired.")
        return FileResponse(
            output_path,
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            filename=f"forecast_result_{file_id}.xlsx"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in /download-forecast endpoint: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error processing file: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    # Sesuaikan port jika Railway menggunakan environment variable PORT
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
