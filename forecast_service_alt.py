from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error, make_scorer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.optimize import minimize
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from pandas import ExcelWriter
import warnings
import io
import base64
from datetime import datetime
import logging

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Forecasting Methods
def forecast_ma6(series):
    """
    MA dengan penyesuaian jumlah data:
    - Jika data >= 6 bulan: ambil rata-rata 6 bulan terakhir
    - Jika data < 6 bulan: ambil rata-rata semua data yang tersedia
    """
    if len(series) == 0:
        return np.nan
    
    # Jika data >= 6 bulan, ambil 6 bulan terakhir
    if len(series) >= 6:
        return np.mean(series[-6:])
    else:
        # Jika data < 6 bulan, ambil rata-rata semua data yang tersedia
        return np.mean(series)

def forecast_wma(series, actual=None, window=6):
    """
    WMA dengan penyesuaian jumlah data:
    - Jika data >= window: gunakan WMA dengan optimisasi
    - Jika data < window: gunakan rata-rata sederhana
    """
    if len(series) == 0:
        return np.nan
    
    if len(series) < window:
        # Jika data < window, gunakan rata-rata sederhana
        return np.mean(series)
    
    if actual is None:
        # Untuk real-time forecast, gunakan WMA sederhana
        weights = np.arange(1, len(series[-window:]) + 1)
        weights = weights / weights.sum()
        return np.sum(series[-window:] * weights)
    
    # Untuk backtesting, optimisasi bobot
    def objective(weights):
        weights = np.array(weights)
        weights /= weights.sum()
        forecast = np.sum(series[-window:] * weights)
        return abs(forecast - actual)
    
    bounds = [(0, 1)] * window
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    init_weights = np.repeat(1 / window, window)
    
    try:
        result = minimize(objective, init_weights, bounds=bounds, constraints=constraints)
        if result.success:
            weights = result.x / result.x.sum()
            return np.sum(series[-window:] * weights)
        else:
            # Fallback ke WMA sederhana
            weights = np.arange(1, window + 1)
            weights = weights / weights.sum()
            return np.sum(series[-window:] * weights)
    except:
        # Fallback ke rata-rata sederhana
        return np.mean(series[-window:])

def forecast_ets(series):
    """
    ETS (Exponential Smoothing) dengan penyesuaian jumlah data
    """
    if len(series) < 3:
        return np.nan
    
    try:
        model = ExponentialSmoothing(
            series, 
            trend=None, 
            seasonal=None, 
            initialization_method="estimated"
        )
        fit = model.fit()
        return max(fit.forecast(1)[0], 0)
    except:
        return np.nan

def forecast_arima(series, val_size=6):
    """
    ARIMA menggunakan statsmodels sebagai pengganti pmdarima
    """
    if len(series) < val_size + 6:
        return np.nan
    
    try:
        # Cek stasioneritas
        adf_result = adfuller(series)
        is_stationary = adf_result[1] < 0.05
        
        # Jika tidak stasioner, lakukan differencing
        if not is_stationary:
            diff_series = np.diff(series)
            if len(diff_series) < 6:
                return np.nan
            series = diff_series
        
        # Coba beberapa kombinasi ARIMA sederhana
        best_aic = np.inf
        best_model = None
        
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_model = fitted_model
                    except:
                        continue
        
        if best_model is not None:
            forecast = best_model.forecast(steps=1)[0]
            # Jika menggunakan differencing, lakukan inverse
            if not is_stationary:
                forecast = series[-1] + forecast
            return max(forecast, 0)
        else:
            return np.nan
    except:
        return np.nan

def forecast_ml(series, model_type='linreg'):
    """
    Machine Learning forecasting dengan penyesuaian jumlah data
    """
    if len(series) < 4:
        return np.nan
    
    try:
        # Buat features
        df = pd.DataFrame({'y': series})
        df['lag1'] = df['y'].shift(1)
        df['lag2'] = df['y'].shift(2)
        df['lag3'] = df['y'].shift(3)
        df.dropna(inplace=True)
        
        if len(df) < 2:
            return np.nan
        
        X = df[['lag1', 'lag2', 'lag3']]
        y = df['y']
        
        # Pilih model
        if model_type == 'linreg':
            model = LinearRegression()
        elif model_type == 'rf':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'xgb':
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        else:
            return np.nan
        
        # Fit model
        model.fit(X, y)
        
        # Predict
        last_values = series[-3:]
        X_pred = np.array([last_values[-1], last_values[-2], last_values[-3]])[::-1].reshape(1, -1)
        prediction = model.predict(X_pred)[0]
        
        return max(prediction, 0)
    except:
        return np.nan

# Model mapping
models = {
    'MA6': forecast_ma6,
    'WMA': forecast_wma,
    'ETS': forecast_ets,
    'ARIMA': forecast_arima,
    'LINREG': lambda s: forecast_ml(s, 'linreg'),
    'RF': lambda s: forecast_ml(s, 'rf'),
    'XGB': lambda s: forecast_ml(s, 'xgb')
}

def process_forecast(file_content, filename):
    """
    Process forecast dari file Excel
    """
    try:
        # Baca file
        df = pd.read_excel(io.BytesIO(file_content))
        
        # Validasi kolom
        required_columns = ['PART_NO', 'MONTH', 'ORIGINAL_SHIPPING_QTY']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return None, f"Kolom yang diperlukan tidak ditemukan: {missing_columns}"
        
        # Proses data
        df['MONTH'] = pd.to_datetime(df['MONTH'].astype(str), format='%Y%m')
        df = df.groupby(['PART_NO', 'MONTH'], as_index=False).agg({
            'ORIGINAL_SHIPPING_QTY': 'sum'
        })
        
        # Jalankan forecasting
        original_df, forecast_df = run_combined_forecast_alt(df)
        
        return original_df, forecast_df, None
        
    except Exception as e:
        return None, None, f"Error processing file: {str(e)}"

def run_combined_forecast_alt(df):
    """
    Versi alternatif dari run_combined_forecast tanpa pmdarima
    """
    # Implementasi forecasting tanpa pmdarima
    # ... (implementasi lengkap akan ditambahkan jika diperlukan)
    pass

def run_real_time_forecast_alt(original_df, forecast_df):
    """
    Versi alternatif dari run_real_time_forecast tanpa pmdarima
    """
    # Implementasi real-time forecast tanpa pmdarima
    # ... (implementasi lengkap akan ditambahkan jika diperlukan)
    pass 
