from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import pandas as pd
from io import BytesIO

# Import fungsi run_pipeline_from_df dari module forecast kamu
from program9 import run_pipeline_from_df


app = FastAPI()

@app.post("/forecast/")
async def forecast(file: UploadFile = File(...)):
    contents = await file.read()
    df_input = pd.read_excel(BytesIO(contents))
    df_forecast = run_pipeline_from_df(df_input)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_forecast.to_excel(writer, index=False, sheet_name='Forecast')
    output.seek(0)

    headers = {
        "Content-Disposition": "attachment; filename=forecast_output.xlsx"
    }
    return StreamingResponse(output, media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', headers=headers)
