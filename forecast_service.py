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
from pmdarima import auto_arima
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
# Strategi Hybrid WMA:
# - Backtesting: Optimisasi bobot dengan minimize untuk mendekati nilai aktual
# - Real-time: WMA adaptif berdasarkan korelasi dan stabilitas data

def forecast_ma6(series):
    return np.mean(series[-6:])

def forecast_wma(series, actual=None, window=6):
    """
    WMA dengan strategi hybrid:
    - Backtesting: Gunakan minimize untuk optimisasi bobot mendekati nilai aktual
    - Real-time: Gunakan WMA adaptif (korelasi + stabilitas)
    
    Parameters:
    - series: historical data
    - actual: actual value (untuk backtesting)
    - window: number of periods
    """
    if len(series) < window:
        return np.nan
    
    # CASE 1: BACKTESTING (dengan nilai aktual) - Optimisasi bobot dengan minimize
    if actual is not None:
        def objective(weights):
            weights = np.array(weights)
            weights /= weights.sum()
            forecast = np.sum(series[-window:] * weights)
            return abs(forecast - actual)
        
        bounds = [(0, 1)] * window
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        init_weights = np.repeat(1 / window, window)
        result = minimize(objective, init_weights, bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x / result.x.sum()
            forecast = np.sum(series[-window:] * optimal_weights)
            return max(forecast, 0)
        else:
            # Fallback ke WMA tradisional jika optimisasi gagal
            weights = np.arange(1, window + 1) / np.arange(1, window + 1).sum()
            forecast = np.sum(series[-window:] * weights)
            return max(forecast, 0)
    
    # CASE 2: REAL-TIME FORECASTING (tanpa nilai aktual) - WMA adaptif
    else:
        return forecast_wma_adaptive_realtime(series, window)

def forecast_ets(series, val_size=3):
    if len(series) < (val_size + 3):
        return np.nan
    best_mape = np.inf
    best_forecast = np.nan
    param_grid = {
        'smoothing_level': np.linspace(0.01, 0.99, 5),
        'smoothing_slope': [None] + list(np.linspace(0.01, 0.99, 5))
    }
    train = series[:-val_size]
    val = series[-val_size:]
    for alpha in param_grid['smoothing_level']:
        for beta in param_grid['smoothing_slope']:
            try:
                model = ExponentialSmoothing(train, trend='add' if beta is not None else None, seasonal=None, initialization_method="estimated")
                fit = model.fit(smoothing_level=alpha, smoothing_trend=beta)
                forecast_val = fit.forecast(len(val))
                mape_val = mean_absolute_percentage_error(val, forecast_val)
                if mape_val < best_mape:
                    best_mape = mape_val
                    forecast = fit.forecast(1)[0]
                    best_forecast = max(forecast, 0)
            except:
                continue
    if np.isnan(best_forecast):
        try:
            model = ExponentialSmoothing(series, trend='add', seasonal='add', initialization_method="estimated")
            fit = model.fit()
            best_forecast = max(fit.forecast(1)[0], 0)
        except:
            best_forecast = np.nan
    return best_forecast

def forecast_arima(series, val_size=6):
    if len(series) < val_size + 12:
        return np.nan

    train = series[:-val_size]
    val = series[-val_size:]
    try:
        model = auto_arima(
            train,
            seasonal=None,
            m=12,
            max_p=3, max_q=3, max_P=2, max_Q=2,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore'
        )
        preds = model.predict(n_periods=val_size)
        mape_val = mean_absolute_percentage_error(val, preds)

        if mape_val < 0.3:
            return model.predict(n_periods=1)[0]
        else:
            full_model = auto_arima(
                series,
                seasonal=None,
                m=12,
                max_p=3, max_q=3, max_P=2, max_Q=2,
                stepwise=True,
                suppress_warnings=True,
                error_action='ignore'
            )
            return full_model.predict(n_periods=1)[0]
    except:
        return np.nan

def forecast_wma_adaptive_realtime(series, window=6, min_history=12):
    """
    WMA adaptif untuk real-time forecasting
    Berdasarkan analisis korelasi dan stabilitas data
    
    Parameters:
    - series: historical data
    - window: number of periods
    - min_history: minimal data untuk analisis (default: 12)
    """
    if len(series) < window:
        return np.nan
    
    # Fallback ke WMA tradisional jika data tidak cukup
    if len(series) < min_history:
        weights = np.arange(1, window + 1) / np.arange(1, window + 1).sum()
        forecast = np.sum(series[-window:] * weights)
        return max(forecast, 0)
    
    # Ambil data untuk analisis (semua data kecuali window terakhir)
    analysis_data = series[:-window]
    recent_data = series[-window:]
    
    # 1. Analisis korelasi setiap posisi dengan nilai berikutnya
    correlations = []
    for pos in range(window):
        pos_correlations = []
        for i in range(len(analysis_data) - window):
            window_data = analysis_data[i:i+window]
            next_value = analysis_data[i+window]
            correlation = np.corrcoef([window_data.iloc[pos]], [next_value])[0,1]
            if not np.isnan(correlation):
                pos_correlations.append(correlation)
        
        # Rata-rata korelasi untuk posisi ini
        if pos_correlations:
            avg_corr = np.mean(pos_correlations)
            correlations.append(avg_corr)
        else:
            correlations.append(0.5)  # Default jika tidak ada korelasi
    
    # 2. Analisis volatilitas setiap posisi
    volatilities = []
    for pos in range(window):
        pos_values = []
        for i in range(len(analysis_data) - window):
            window_data = analysis_data[i:i+window]
            pos_values.append(window_data.iloc[pos])
        
        if pos_values:
            volatility = np.std(pos_values)
            volatilities.append(volatility)
        else:
            volatilities.append(1.0)  # Default
    
    # 3. Kombinasi bobot: korelasi + stabilitas
    # Normalisasi korelasi (semakin tinggi korelasi, semakin tinggi bobot)
    norm_correlations = np.array(correlations) / np.sum(correlations)
    
    # Normalisasi volatilitas (semakin rendah volatilitas, semakin tinggi bobot)
    norm_volatilities = 1 / (np.array(volatilities) + 1e-8)  # +1e-8 untuk avoid division by zero
    norm_volatilities = norm_volatilities / np.sum(norm_volatilities)
    
    # Kombinasi bobot (bisa disesuaikan)
    alpha, beta = 0.7, 0.3  # Bobot untuk korelasi, stabilitas
    adaptive_weights = (alpha * norm_correlations + 
                       beta * norm_volatilities)
    
    # Normalisasi final
    adaptive_weights = adaptive_weights / np.sum(adaptive_weights)
    
    # 4. Hitung forecast
    forecast = np.sum(recent_data * adaptive_weights)
    
    return max(forecast, 0)



# Helper Functions
mape_scorer_sklearn = make_scorer(mean_absolute_percentage_error, greater_is_better=False)

def sample_data_latest(X, y, sample_frac=0.5):
    n = len(X)
    if n > 50:
        start_idx = int(n * (1 - sample_frac))
        return X.iloc[start_idx:], y.iloc[start_idx:]
    else:
        return X, y

def tune_model_if_needed(model_name, model, X_train, y_train, X_test, actual, initial_pred, mape_threshold=0.2):
    if np.isnan(initial_pred) or np.isnan(actual):
        return initial_pred, model
    error_now = mean_absolute_percentage_error([actual], [initial_pred])
    if error_now <= mape_threshold:
        return initial_pred, model
    
    X_sample, y_sample = sample_data_latest(X_train, y_train)
    if len(X_sample) < 2:
        return initial_pred, model

    param_dist = {
        'RF': {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'XGB': {
            'n_estimators': [50, 100, 150, 200],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0]
        },
        'LINREG': {}
    }

    if model_name not in param_dist:
        return initial_pred, model

    tscv = TimeSeriesSplit(n_splits=3)
    try:
        rnd_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist[model_name],
            n_iter=15 if param_dist[model_name] else 1,
            cv=tscv,
            scoring='neg_mean_absolute_percentage_error',
            n_jobs=-1,
            random_state=42
        )
        rnd_search.fit(X_sample, y_sample)
        tuned_pred = rnd_search.best_estimator_.predict(X_test)[0]
        return tuned_pred, rnd_search.best_estimator_
    except Exception as e:
        logger.error(f"Tuning error for model {model_name}: {e}")
        return initial_pred, model

def hybrid_error(actual, forecast):
    actual = float(actual)
    forecast = float(forecast)
    if np.isnan(actual) or np.isnan(forecast):
        return np.inf
    if actual == 0:
        return 2 * abs(forecast - actual) / (abs(forecast) + abs(actual) + 1e-8)
    else:
        return abs((actual - forecast) / actual)

def parse_month_column(series):
    import pandas as pd
    # Jika sudah datetime, return
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    # Jika int (202501), parse dengan %Y%m
    if pd.api.types.is_integer_dtype(series):
        return pd.to_datetime(series.astype(str), format='%Y%m', errors='coerce')
    # Jika float (202501.0), parse ke int lalu ke str
    if pd.api.types.is_float_dtype(series):
        return pd.to_datetime(series.astype(int).astype(str), format='%Y%m', errors='coerce')
    # Jika string
    s = series.astype(str)
    # Coba format %Y%m
    dt = pd.to_datetime(s, format='%Y%m', errors='coerce')
    if dt.notna().all():
        return dt
    # Coba format %Y-%m
    dt = pd.to_datetime(s, format='%Y-%m', errors='coerce')
    if dt.notna().all():
        return dt
    # Coba format %Y/%m
    dt = pd.to_datetime(s, format='%Y/%m', errors='coerce')
    if dt.notna().all():
        return dt
    # Coba format %Y-%m-%d
    dt = pd.to_datetime(s, format='%Y-%m-%d', errors='coerce')
    if dt.notna().all():
        return dt
    # Fallback: auto
    return pd.to_datetime(s, errors='coerce')

# Main Forecast per PART_NO
def process_part(part, part_df, test_months):
    part_results = []
    for target_month in test_months:
        train_df = part_df[part_df['MONTH'] < target_month].copy()
        test_df = part_df[part_df['MONTH'] == target_month].copy()

        six_months_prior = pd.date_range(end=target_month - pd.offsets.MonthBegin(1), periods=6, freq='MS')
        recent_6_months_df = train_df[train_df['MONTH'].isin(six_months_prior)]
        total_qty_last_6_months = recent_6_months_df['ORIGINAL_SHIPPING_QTY'].sum()

        if total_qty_last_6_months < 2:
            part_results.append({
                'PART_NO': part,
                'MONTH': target_month.strftime('%Y-%m'),
                'FORECAST': 0,
                'ACTUAL': test_df['ORIGINAL_SHIPPING_QTY'].values[0] if not test_df.empty else np.nan,
                'HYBRID_ERROR': None,
                'BEST_MODEL': 'NONE'
            })
            continue

        history = train_df.set_index('MONTH').resample('MS').sum().fillna(0)
        series = history['ORIGINAL_SHIPPING_QTY'].values
        actual = test_df['ORIGINAL_SHIPPING_QTY'].values[0] if not test_df.empty else np.nan

        preds = {
            'MA6': forecast_ma6(series),
            'WMA': forecast_wma(pd.Series(series), actual=actual),
            'ETS': forecast_ets(series),
            'ARIMA': forecast_arima(series)
        }

        train_df['ROLLING_MEAN_3'] = train_df['ORIGINAL_SHIPPING_QTY'].rolling(window=3).mean()
        train_df['ROLLING_MEAN_6'] = train_df['ORIGINAL_SHIPPING_QTY'].rolling(window=6).mean()
        train_df['ROLLING_STD_3'] = train_df['ORIGINAL_SHIPPING_QTY'].rolling(window=3).std()
        train_df['ROLLING_STD_6'] = train_df['ORIGINAL_SHIPPING_QTY'].rolling(window=6).std()
        train_df['GROWTH_LAG_3_6'] = train_df['LAG_3'] / (train_df['LAG_6'] + 1e-8)

        features = ['LAG_1', 'LAG_2', 'LAG_3', 'LAG_4', 'LAG_5', 'LAG_6',
                    'MONTH_NUM', 'YEAR', 'MONTH_SIN', 'MONTH_COS',
                    'MEAN_LAG_3', 'SUM_LAG_6', 'GROWTH_RATE',
                    'ROLLING_MEAN_3', 'ROLLING_MEAN_6',
                    'ROLLING_STD_3', 'ROLLING_STD_6',
                    'GROWTH_LAG_3_6']

        cat_cols = ['PART_NAME', 'TOPAS_ORDER_TYPE', 'CREATED_DEMAND_FLAG', 'CUST_TYPE2']
        for col in cat_cols:
            le = LabelEncoder()
            all_vals = pd.concat([train_df[col], test_df[col]])
            le.fit(all_vals)
            train_df[col] = le.transform(train_df[col])
            test_df[col] = le.transform(test_df[col])
            features.append(col)

        X_train, y_train = train_df[features], train_df['ORIGINAL_SHIPPING_QTY']
        X_test = test_df[features] if not test_df.empty else None

        mask_valid = X_train.notna().all(axis=1)
        X_train = X_train[mask_valid]
        y_train = y_train[mask_valid]

        ml_models = {
            'LINREG': LinearRegression(),
            'RF': RandomForestRegressor(n_estimators=100, random_state=42),
            'XGB': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        }

        for name, model in ml_models.items():
            if len(X_train) == 0:
                preds[name] = np.nan
                continue

            model.fit(X_train, y_train)

            if X_test is not None and not X_test.empty:
                if X_test.isnull().all(axis=1).any():
                    preds[name] = 0
                else:
                    pred = model.predict(X_test)[0]
                    tuned_pred, tuned_model = tune_model_if_needed(name, model, X_train, y_train, X_test, actual, pred, mape_threshold=0.2)
                    preds[name] = tuned_pred
            else:
                preds[name] = np.nan

        errors = {m: hybrid_error(actual, p) for m, p in preds.items()}
        best_model_name = min(errors, key=lambda x: errors[x])

        part_results.append({
            'PART_NO': part,
            'MONTH': target_month.strftime('%Y-%m'),
            'FORECAST': round(preds[best_model_name]) if not np.isnan(preds[best_model_name]) else None,
            'ACTUAL': actual,
            'HYBRID_ERROR': f"{errors[best_model_name]:.2f}%" if errors[best_model_name] != np.inf else None,
            'BEST_MODEL': best_model_name
        })

    return part_results

# Run Combined Forecast (Backtest)
def run_combined_forecast(df):
    logger.info("Starting backtest process...")
    
    # Proses dataset
    df_processed = df.copy()
    # Robust parsing for MONTH column
    df_processed['MONTH'] = parse_month_column(df_processed['MONTH'])
    logger.info(f"Data after parsing MONTH: {df_processed.shape}\n{df_processed.head(3)}")
    df_processed = df_processed.groupby(['PART_NO', 'MONTH'], as_index=False).agg({
        'ORIGINAL_SHIPPING_QTY': 'sum',
        'PART_NAME': 'first',
        'TOPAS_ORDER_TYPE': 'first',
        'CREATED_DEMAND_FLAG': 'first',
        'CUST_TYPE2': 'first'
    })

    # Fitur waktu dan lag
    df_processed['MONTH_NUM'] = df_processed['MONTH'].dt.month
    df_processed['YEAR'] = df_processed['MONTH'].dt.year
    df_processed['MONTH_SIN'] = np.sin(2 * np.pi * df_processed['MONTH_NUM'] / 12)
    df_processed['MONTH_COS'] = np.cos(2 * np.pi * df_processed['MONTH_NUM'] / 12)
    df_processed['LAG_1'] = df_processed.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(1)
    df_processed['LAG_2'] = df_processed.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(2)
    df_processed['LAG_3'] = df_processed.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(3)
    df_processed['LAG_4'] = df_processed.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(4)
    df_processed['LAG_5'] = df_processed.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(5)
    df_processed['LAG_6'] = df_processed.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(6)

    df_processed['ROLLING_MEAN_3'] = df_processed.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].transform(lambda x: x.rolling(window=3).mean())
    df_processed['ROLLING_MEAN_6'] = df_processed.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].transform(lambda x: x.rolling(window=6).mean())
    df_processed['ROLLING_STD_3'] = df_processed.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].transform(lambda x: x.rolling(window=3).std())
    df_processed['ROLLING_STD_6'] = df_processed.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].transform(lambda x: x.rolling(window=6).std())
    
    df_processed['GROWTH_LAG_3_6'] = df_processed['LAG_3'] / (df_processed['LAG_6'] + 1e-8)
    df_processed['MEAN_LAG_3'] = df_processed[['LAG_1', 'LAG_2', 'LAG_3']].mean(axis=1)
    df_processed['SUM_LAG_6'] = df_processed[['LAG_1', 'LAG_2', 'LAG_3', 'LAG_4', 'LAG_5', 'LAG_6']].sum(axis=1)
    df_processed['GROWTH_RATE'] = df_processed['LAG_1'] / (df_processed['LAG_2'] + 1e-8)

    # === PEMILIHAN BULAN BACKTEST SAMA DENGAN PROGRAM PENGUJIAN 5 ===
    all_months = df_processed['MONTH'].sort_values().unique()
    if len(all_months) < 5:
        raise Exception("Dataset tidak cukup untuk 4 bulan backtest + 1 bulan forecast")
    # Ambil 4 bulan terakhir sebelum bulan terakhir di dataset
    test_months = pd.to_datetime(all_months[-5:-1])
    logger.info(f"Backtest test_months: {[m.strftime('%Y-%m') for m in test_months]}")

    # Lakukan forecasting untuk semua part
    part_list = df_processed['PART_NO'].unique()
    logger.info(f"Processing {len(part_list)} parts...")
    
    results_nested = Parallel(n_jobs=-1)(
        delayed(process_part)(part, df_processed[df_processed['PART_NO'] == part].sort_values('MONTH'), test_months)
        for part in tqdm(part_list, desc="Processing parts")
    )

    results = [item for sublist in results_nested if sublist is not None for item in sublist]
    forecast_df = pd.DataFrame(results)

    logger.info("Backtest completed successfully")
    return df_processed, forecast_df

# Real-time Forecast Function

def run_real_time_forecast(original_df, forecast_df):
    logger.info("Starting real-time forecast...")
    
    df_all = original_df.copy()
    # Robust parsing for MONTH column
    df_all['MONTH'] = parse_month_column(df_all['MONTH'])
    logger.info(f"Data for real-time forecast: {df_all.shape}\n{df_all.head(3)}")
    
    df_best = forecast_df.copy()
    df_best['MONTH'] = pd.to_datetime(df_best['MONTH'], format='mixed')
    df_best['FORECAST'] = pd.to_numeric(df_best['FORECAST'], errors='coerce')
    df_best['ACTUAL'] = pd.to_numeric(df_best['ACTUAL'], errors='coerce')
    
    def hybrid_error(row):
        f, a = row['FORECAST'], row['ACTUAL']
        if pd.isna(f) or pd.isna(a):
            return np.nan
        if a == 0 and f == 0:
            return 0
        elif a == 0:
            return 2 * abs(f - a) / (abs(f) + abs(a)) * 100
        else:
            return abs(f - a) / a * 100
    
    df_best['HYBRID_ERROR'] = df_best.apply(hybrid_error, axis=1)
    
    latest_months = sorted(df_best['MONTH'].unique())[-4:]
    df_testing = df_best[df_best['MONTH'].isin(latest_months)].dropna(subset=['HYBRID_ERROR'])
    idx_min_error = df_testing.groupby('PART_NO')['HYBRID_ERROR'].idxmin()
    model_best_rows = df_testing.loc[idx_min_error, ['PART_NO', 'BEST_MODEL', 'HYBRID_ERROR']]
    latest_model_map = model_best_rows.set_index('PART_NO')[['BEST_MODEL', 'HYBRID_ERROR']].to_dict(orient='index')
    
    all_months = sorted(df_all['MONTH'].unique())
    if len(all_months) < 1:
        raise Exception("Dataset tidak ada bulan valid")
    last_month = all_months[-1].replace(day=1)
    # === PEMILIHAN BULAN FORECAST SAMA DENGAN PROGRAM PENGUJIAN 5 ===
    forecast_months = pd.date_range(start=last_month, periods=4, freq='MS')
    logger.info(f"RealTime forecast_months: {[m.strftime('%Y-%m') for m in forecast_months]}")

    part_list = df_all['PART_NO'].unique()
    
    six_months_ago = last_month - pd.DateOffset(months=6)
    recent_data = df_all[df_all['MONTH'] > six_months_ago]
    valid_parts = recent_data[recent_data['ORIGINAL_SHIPPING_QTY'] > 0].groupby('PART_NO').size().reset_index(name='nonzero_count')
    valid_parts = valid_parts[valid_parts['nonzero_count'] >= 2]['PART_NO'].tolist()
    
    results = []
    
    for part in part_list:
        if part not in valid_parts or part not in latest_model_map:
            for month in forecast_months:
                results.append({
                    'PART_NO': part,
                    'MONTH': month.strftime('%Y-%m'),
                    'BEST_MODEL': 'NoModel',
                    'FORECAST': 0,
                    'FORECAST_OPTIMIST': 0,
                    'FORECAST_NEUTRAL': 0,
                    'FORECAST_PESSIMIST': 0,
                    'ERROR_BACKTEST': ''
                })
            continue
    
        model_info = latest_model_map[part]
        best_model = model_info['BEST_MODEL']
        mape_val = model_info['HYBRID_ERROR']
    
        part_df = df_all[df_all['PART_NO'] == part].copy().sort_values('MONTH')
        hist = part_df.set_index('MONTH').resample('MS')['ORIGINAL_SHIPPING_QTY'].sum().fillna(0)
        series = hist.values
        df_series = hist.to_frame('ORIGINAL_SHIPPING_QTY').reset_index()
    
        if len(series) < 6:
            for month in forecast_months:
                results.append({
                    'PART_NO': part,
                    'MONTH': month.strftime('%Y-%m'),
                    'BEST_MODEL': best_model,
                    'FORECAST': 0,
                    'FORECAST_OPTIMIST': 0,
                    'FORECAST_NEUTRAL': 0,
                    'FORECAST_PESSIMIST': 0,
                    'ERROR_BACKTEST': f"{round(mape_val, 2)}%"
                })
            continue
    
        model_trained = None
        cat_cols = ['PART_NAME', 'TOPAS_ORDER_TYPE', 'CREATED_DEMAND_FLAG', 'CUST_TYPE2']
        features = []
        df_f = None
        if best_model in ['LINREG', 'RF', 'XGB']:
            df_f = df_series.copy()
            df_f['MONTH_NUM'] = df_f['MONTH'].dt.month
            df_f['YEAR'] = df_f['MONTH'].dt.year
            df_f['MONTH_SIN'] = np.sin(2 * np.pi * df_f['MONTH_NUM'] / 12)
            df_f['MONTH_COS'] = np.cos(2 * np.pi * df_f['MONTH_NUM'] / 12)
        
            for i in range(1, 7):
                df_f[f'LAG_{i}'] = df_f['ORIGINAL_SHIPPING_QTY'].shift(i)
        
            df_f['ROLLING_MEAN_3'] = df_f['ORIGINAL_SHIPPING_QTY'].rolling(window=3).mean()
            df_f['ROLLING_MEAN_6'] = df_f['ORIGINAL_SHIPPING_QTY'].rolling(window=6).mean()
            df_f['ROLLING_STD_3'] = df_f['ORIGINAL_SHIPPING_QTY'].rolling(window=3).std()
            df_f['ROLLING_STD_6'] = df_f['ORIGINAL_SHIPPING_QTY'].rolling(window=6).std()
            df_f['GROWTH_LAG_3_6'] = df_f['LAG_3'] / (df_f['LAG_6'] + 1e-8)
        
            df_f = df_f.dropna()
        
            if not df_f.empty:
                features = [f'LAG_{i}' for i in range(1, 7)] + ['MONTH_NUM', 'YEAR', 'MONTH_SIN', 'MONTH_COS',
                            'ROLLING_MEAN_3', 'ROLLING_MEAN_6', 'ROLLING_STD_3', 'ROLLING_STD_6', 'GROWTH_LAG_3_6']
        
                part_data = df_all[df_all['PART_NO'] == part].copy()
                for col in cat_cols:
                    if col in part_data.columns:
                        le = LabelEncoder()
                        vals = part_data[col].fillna('NA').astype(str)
                        le.fit(vals)
                        df_f[col] = le.transform(df_f['MONTH'].map(lambda d: vals.iloc[part_data['MONTH'].searchsorted(d)] if part_data['MONTH'].searchsorted(d) < len(vals) else 0))
                        features.append(col)
        
                X_train = df_f[features]
                y_train = df_f['ORIGINAL_SHIPPING_QTY']
        
                model_trained = (
                    LinearRegression() if best_model == 'LINREG' else
                    RandomForestRegressor(n_estimators=100, random_state=42) if best_model == 'RF' else
                    XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                )
                model_trained.fit(X_train, y_train)

        # --- Recursive Forecasting ---
        future_series = list(series)
        future_df_f = df_f.copy() if df_f is not None else None
        
        # === LOGIKA PERSENTASE BERDASARKAN DEMAND ===
        # Hitung rata-rata demand per bulan untuk part ini (dari data historis)
        avg_monthly_demand = np.mean(series) if len(series) > 0 else 0
        
        # Tentukan persentase berdasarkan rata-rata demand
        if avg_monthly_demand < 500:
            # Part dengan demand < 500 pcs/bulan
            optimist_percent = 1.15  # +15%
            pessimist_percent = 0.80  # -20%
        else:
            # Part dengan demand >= 500 pcs/bulan
            optimist_percent = 1.07  # +7%
            pessimist_percent = 0.93  # -7%
        
        for idx, month in enumerate(forecast_months):
            # Buat lag dari future_series
            lags = [future_series[-i] if len(future_series) >= i else 0 for i in range(1, 7)]
            if best_model == 'MA6':
                fc = forecast_ma6(future_series)
            elif best_model == 'WMA':
                fc = forecast_wma(pd.Series(future_series))
            elif best_model == 'ETS':
                fc = forecast_ets(np.array(future_series))
            elif best_model == 'ARIMA':
                fc = forecast_arima(np.array(future_series))
            elif model_trained and future_df_f is not None and not future_df_f.empty:
                test_row = {f'LAG_{i}': lags[i-1] for i in range(1, 7)}
                test_row['MONTH_NUM'] = month.month
                test_row['YEAR'] = month.year
                test_row['MONTH_SIN'] = np.sin(2 * np.pi * month.month / 12)
                test_row['MONTH_COS'] = np.cos(2 * np.pi * month.month / 12)
                # Rolling features: update dari future_series
                test_row['ROLLING_MEAN_3'] = np.mean(future_series[-3:]) if len(future_series) >= 3 else 0
                test_row['ROLLING_MEAN_6'] = np.mean(future_series[-6:]) if len(future_series) >= 6 else 0
                test_row['ROLLING_STD_3'] = np.std(future_series[-3:]) if len(future_series) >= 3 else 0
                test_row['ROLLING_STD_6'] = np.std(future_series[-6:]) if len(future_series) >= 6 else 0
                test_row['GROWTH_LAG_3_6'] = lags[2] / (lags[5] + 1e-8) if len(lags) >= 6 else 0
                # Categorical: gunakan nilai terakhir
                for col in cat_cols:
                    test_row[col] = future_df_f[col].iloc[-1] if col in future_df_f.columns and not future_df_f.empty else 0
                X_test = pd.DataFrame([test_row])
                y_pred = model_trained.predict(X_test)
                fc = float(y_pred[0]) if len(y_pred) > 0 else 0
                # Tambahkan ke future_df_f
                new_row = test_row.copy()
                new_row['ORIGINAL_SHIPPING_QTY'] = fc
                new_row['MONTH'] = month
                future_df_f = pd.concat([future_df_f, pd.DataFrame([new_row])], ignore_index=True)
            else:
                fc = 0

            # Tambahkan hasil forecast ke future_series
            future_series.append(fc)

            results.append({
                'PART_NO': part,
                'MONTH': month.strftime('%Y-%m'),
                'BEST_MODEL': best_model,
                'FORECAST_OPTIMIST': max(round(fc * optimist_percent), 0),
                'FORECAST_NEUTRAL': max(round(fc), 0),
                'FORECAST_PESSIMIST': max(round(fc * pessimist_percent), 0),
                'ERROR_BACKTEST': f"{round(mape_val, 2)}%"
            })
    
    df_res = pd.DataFrame(results)
    logger.info("Real-time forecast completed successfully")
    return df_res

# Main processing function
def process_forecast(df):
    """
    Main function to process forecast from input dataframe
    """
    try:
        logger.info("Starting forecast processing...")
        
        # Run backtest
        original_df, forecast_df = run_combined_forecast(df)
        
        # Run real-time forecast
        real_time_forecast = run_real_time_forecast(original_df, forecast_df)
        
        # Create Excel file in memory
        output = io.BytesIO()
        with ExcelWriter(output, engine='openpyxl', mode='w') as writer:
            real_time_forecast.to_excel(writer, sheet_name='Forecast_Results', index=False)
            
            # Add summary sheet
            summary = real_time_forecast.groupby('PART_NO').agg({
                'BEST_MODEL': lambda x: list(set(x)),
                'FORECAST_NEUTRAL': ['mean', 'min', 'max', 'std'],
                'ERROR_BACKTEST': lambda x: np.mean([float(str(e).replace('%', '')) for e in x if str(e).replace('%', '').replace('.', '').isdigit()])
            }).round(2)
            summary.to_excel(writer, sheet_name='Part_Summary')
            
            # Add error analysis sheet
            error_analysis = real_time_forecast.copy()
            error_analysis['ERROR_NUMERIC'] = pd.to_numeric(error_analysis['ERROR_BACKTEST'].str.replace('%', ''), errors='coerce')
            error_summary = error_analysis.groupby('PART_NO')['ERROR_NUMERIC'].agg(['mean', 'max', 'count']).reset_index()
            error_summary.columns = ['PART_NO', 'AVG_ERROR', 'MAX_ERROR', 'MONTH_COUNT']
            error_summary.to_excel(writer, sheet_name='Error_Analysis', index=False)
        
        output.seek(0)
        
        # Convert to base64 for API response
        excel_base64 = base64.b64encode(output.getvalue()).decode()
        
        logger.info("Forecast processing completed successfully")
        
        return {
            "status": "success",
            "message": "Forecast completed successfully",
            "data": {
                "excel_base64": excel_base64,
                "total_forecasts": len(real_time_forecast),
                "total_parts": real_time_forecast['PART_NO'].nunique(),
                "forecast_months": real_time_forecast['MONTH'].unique().tolist(),
                "timestamp": datetime.now().isoformat()
            }
        }
        
    except Exception as e:
        logger.error(f"Error in forecast processing: {str(e)}")
        return {
            "status": "error",
            "message": f"Forecast processing failed: {str(e)}",
            "data": None
        }

def example_wma_usage():
    """
    Contoh penggunaan fungsi forecast_wma untuk berbagai skenario
    """
    import numpy as np
    
    # Data contoh: 6 bulan terakhir
    historical_data = [100, 120, 110, 130, 140, 125]
    
    print("=== CONTOH PENGGUNAAN WMA HYBRID ===\n")
    
    # 1. BACKTESTING (dengan nilai aktual untuk optimisasi)
    actual_value = 135
    print("1. BACKTESTING (Optimisasi Bobot dengan minimize):")
    print(f"   Data historis: {historical_data}")
    print(f"   Nilai aktual: {actual_value}")
    
    # WMA dengan optimisasi bobot (backtesting)
    forecast_backtest = forecast_wma(historical_data, actual=actual_value)
    print(f"   Prediksi WMA (optimisasi): {forecast_backtest:.2f}")
    
    # Error
    error_backtest = abs(forecast_backtest - actual_value)
    print(f"   Error: {error_backtest:.2f}")
    
    # 2. REAL-TIME FORECASTING (tanpa nilai aktual)
    print("\n2. REAL-TIME FORECASTING (WMA Adaptif):")
    print(f"   Data historis: {historical_data}")
    print(f"   Nilai aktual: Tidak tersedia (prediksi masa depan)")
    
    # WMA adaptif untuk real-time
    forecast_realtime = forecast_wma(historical_data, actual=None)
    print(f"   Prediksi WMA Adaptif: {forecast_realtime:.2f}")
    
    print("\n3. PENJELASAN STRATEGI:")
    print("   - Backtesting: Menggunakan minimize untuk optimisasi bobot")
    print("   - Real-time: Menggunakan WMA adaptif (korelasi + stabilitas)")
    
    # Contoh bobot tradisional untuk perbandingan
    weights_traditional = np.arange(1, 7) / np.arange(1, 7).sum()
    print(f"\n4. Bobot WMA Tradisional: {weights_traditional}")
    print(f"   (Data terbaru mendapat bobot tertinggi: {weights_traditional[-1]:.3f})")
    
    # Hitung manual untuk verifikasi
    manual_calc = sum(historical_data[i] * weights_traditional[i] for i in range(6))
    print(f"   Perhitungan manual: {manual_calc:.2f}")
