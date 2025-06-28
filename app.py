from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os

# Import fungsi utama forecasting Anda
from your_forecast_module import main_forecast_pipeline

app = FastAPI()

# Pastikan folder upload dan output ada
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static files agar hasil forecast bisa diakses via URL
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

@app.post("/upload-forecast/")
async def upload_forecast(file: UploadFile = File(...)):
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        # Simpan file upload
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Jalankan forecasting pipeline
        output_file, _ = main_forecast_pipeline(file_location, output_dir=OUTPUT_DIR)
        
        # Dapatkan URL file hasil forecast
        # Ganti dengan URL Railway Anda nanti!
        railway_url = "https://your-railway-app-url.up.railway.app"
        file_url = f"{railway_url}/outputs/{os.path.basename(output_file)}"
        
        return {"forecast_file_url": file_url}
    
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
