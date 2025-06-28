from joblib import Parallel, delayed
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.optimize import minimize
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
import warnings

warnings.filterwarnings("ignore")

# === Forecasting Methods ===
def forecast_ma6(series):
    return np.mean(series[-6:])

def forecast_wma(series, actual=None, window=6):
    if len(series) < window:
        return np.nan
    if actual is None:
        return np.mean(series[-window:])
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
        weights = result.x / result.x.sum()
        forecast = np.sum(series[-window:] * weights)
        return max(forecast, 0)
    else:
        return max(np.mean(series[-window:]), 0)

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
                fit = model.fit(smoothing_level=alpha, smoothing_slope=beta)
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
        model = auto_arima(train, seasonal=None, m=12,
                           max_p=3, max_q=3, max_P=2, max_Q=2,
                           stepwise=True, suppress_warnings=True, error_action='ignore')
        preds = model.predict(n_periods=val_size)
        mape_val = mean_absolute_percentage_error(val, preds)
        if mape_val < 0.3:
            return model.predict(n_periods=1)[0]
        else:
            full_model = auto_arima(series, seasonal=None, m=12,
                                   max_p=3, max_q=3, max_P=2, max_Q=2,
                                   stepwise=True, suppress_warnings=True, error_action='ignore')
            return full_model.predict(n_periods=1)[0]
    except:
        return np.nan

# === Helper functions ===
def hybrid_error(actual, forecast):
    actual = float(actual)
    forecast = float(forecast)
    if np.isnan(actual) or np.isnan(forecast):
        return np.inf
    if actual == 0:
        return 2 * abs(forecast - actual) / (abs(forecast) + abs(actual) + 1e-8)
    else:
        return abs((actual - forecast) / actual)

# Function untuk proses satu part (untuk backtest)
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
        errors = {m: hybrid_error(actual, p) for m, p in preds.items()}
        best_model_name = min(errors, key=errors.get)
        part_results.append({
            'PART_NO': part,
            'MONTH': target_month.strftime('%Y-%m'),
            'FORECAST': round(preds[best_model_name]) if not np.isnan(preds[best_model_name]) else None,
            'ACTUAL': actual,
            'HYBRID_ERROR': f"{errors[best_model_name]:.2f}%" if errors[best_model_name] != np.inf else None,
            'BEST_MODEL': best_model_name
        })
    return part_results

# === Fungsi utama run_combined_forecast dari dataframe ===
def run_combined_forecast_from_df(original_df):
    df = original_df.copy()
    df['MONTH'] = pd.to_datetime(df['MONTH'].astype(str), format='%Y%m')
    df = df.groupby(['PART_NO', 'MONTH'], as_index=False).agg({
        'ORIGINAL_SHIPPING_QTY': 'sum',
        'PART_NAME': 'first',
        'TOPAS_ORDER_TYPE': 'first',
        'CREATED_DEMAND_FLAG': 'first',
        'CUST_TYPE2': 'first'
    })

    # Fitur waktu dan lag (disesuaikan sesuai kebutuhan)
    df['MONTH_NUM'] = df['MONTH'].dt.month
    df['YEAR'] = df['MONTH'].dt.year
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH_NUM'] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH_NUM'] / 12)

    # Tambah lag
    for lag in range(1, 7):
        df[f'LAG_{lag}'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(lag)

    # Tambah rolling mean dan std dev
    df['ROLLING_MEAN_3'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].transform(lambda x: x.rolling(window=3).mean())
    df['ROLLING_MEAN_6'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].transform(lambda x: x.rolling(window=6).mean())
    df['ROLLING_STD_3'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].transform(lambda x: x.rolling(window=3).std())
    df['ROLLING_STD_6'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].transform(lambda x: x.rolling(window=6).std())

    df['GROWTH_LAG_3_6'] = df['LAG_3'] / (df['LAG_6'] + 1e-8)
    df['MEAN_LAG_3'] = df[['LAG_1', 'LAG_2', 'LAG_3']].mean(axis=1)
    df['SUM_LAG_6'] = df[['LAG_1', 'LAG_2', 'LAG_3', 'LAG_4', 'LAG_5', 'LAG_6']].sum(axis=1)
    df['GROWTH_RATE'] = df['LAG_1'] / (df['LAG_2'] + 1e-8)

    # Tentukan test months (4 bulan terakhir sebelum bulan ini)
    now = pd.Timestamp.now()
    current_month_start = pd.Timestamp(year=now.year, month=now.month, day=1)
    all_months = df['MONTH'].sort_values().unique()
    valid_months = all_months[all_months < current_month_start]
    test_months = pd.to_datetime(valid_months[-4:])

    part_list = df['PART_NO'].unique()
    results_nested = Parallel(n_jobs=-1)(
        delayed(process_part)(part, df[df['PART_NO'] == part].sort_values('MONTH'), test_months)
        for part in tqdm(part_list, desc="Processing parts")
    )
    results = [item for sublist in results_nested for item in sublist]
    forecast_df = pd.DataFrame(results)
    return original_df, forecast_df

# === Fungsi run real time forecast pakai hasil backtest dataframe ===
def run_real_time_forecast_no_mba_df(original_df, forecast_df):
    df_all = original_df.copy()
    df_all['MONTH'] = pd.to_datetime(df_all['MONTH'].astype(str), format='%Y%m')

    df_best = forecast_df.copy()
    df_best['MONTH'] = pd.to_datetime(df_best['MONTH'])
    df_best['FORECAST'] = pd.to_numeric(df_best['FORECAST'], errors='coerce')
    df_best['ACTUAL'] = pd.to_numeric(df_best['ACTUAL'], errors='coerce')

    latest_months = sorted(df_best['MONTH'].unique())[-4:]
    df_testing = df_best[df_best['MONTH'].isin(latest_months)].dropna(subset=['HYBRID_ERROR'])
    idx_min_error = df_testing.groupby('PART_NO')['HYBRID_ERROR'].idxmin()
    model_best_rows = df_testing.loc[idx_min_error, ['PART_NO', 'BEST_MODEL', 'HYBRID_ERROR']]
    latest_model_map = model_best_rows.set_index('PART_NO')[['BEST_MODEL', 'HYBRID_ERROR']].to_dict(orient='index')

    all_months = sorted(df_all['MONTH'].unique())
    if len(all_months) < 5:
        raise ValueError("Dataset tidak cukup untuk 4 bulan backtest + 1 bulan forecast")

    last_month = all_months[-1]
    forecast_months = pd.date_range(start=last_month, periods=4, freq='MS')

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
        if best_model in ['LINREG', 'RF', 'XGB']:
            df_f = hist.to_frame('ORIGINAL_SHIPPING_QTY').reset_index()
            df_f['MONTH_NUM'] = df_f['MONTH'].dt.month
            df_f['YEAR'] = df_f['MONTH'].dt.year
            df_f['MONTH_SIN'] = np.sin(2 * np.pi * df_f['MONTH_NUM'] / 12)
            df_f['MONTH_COS'] = np.cos(2 * np.pi * df_f['MONTH_NUM'] / 12)
            for i in range(1, 7):
                df_f[f'LAG_{i}'] = df_f['ORIGINAL_SHIPPING_QTY'].shift(i)
            df_f = df_f.dropna()
            if not df_f.empty:
                X_train = df_f[[f'LAG_{i}' for i in range(1, 7)] + ['MONTH_NUM', 'YEAR', 'MONTH_SIN', 'MONTH_COS']]
                y_train = df_f['ORIGINAL_SHIPPING_QTY']
                if best_model == 'LINREG':
                    model_trained = LinearRegression()
                elif best_model == 'RF':
                    model_trained = RandomForestRegressor(n_estimators=100, random_state=42)
                elif best_model == 'XGB':
                    model_trained = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
                model_trained.fit(X_train, y_train)

        for month in forecast_months:
            fc = 0
            if best_model == 'MA6':
                fc = forecast_ma6(series)
            elif best_model == 'WMA':
                fc = forecast_wma(series)
            elif best_model == 'ETS':
                fc = forecast_ets(series)
            elif best_model == 'ARIMA':
                fc = forecast_arima(series)
            elif model_trained:
                lags = {f'LAG_{i}': series[-i] if len(series) >= i else 0 for i in range(1, 7)}
                X_test = pd.DataFrame([{**lags, 'MONTH_NUM': month.month, 'YEAR': month.year,
                                       'MONTH_SIN': np.sin(2 * np.pi * month.month / 12),
                                       'MONTH_COS': np.cos(2 * np.pi * month.month / 12)}])
                y_pred = model_trained.predict(X_test)
                fc = float(y_pred[0]) if len(y_pred) > 0 else 0

            results.append({
                'PART_NO': part,
                'MONTH': month.strftime('%Y-%m'),
                'BEST_MODEL': best_model,
                'FORECAST_OPTIMIST': max(round(fc * 1.15), 0),
                'FORECAST_NEUTRAL': max(round(fc), 0),
                'FORECAST_PESSIMIST': max(round(fc * 0.8), 0),
                'ERROR_BACKTEST': f"{round(mape_val, 2)}%"
            })

    df_res = pd.DataFrame(results)
    return df_res

# === Fungsi utama pipeline dari dataframe input ===
def run_pipeline_from_df(df):
    original_df, forecast_df = run_combined_forecast_from_df(df)
    df_real_time_forecast = run_real_time_forecast_no_mba_df(original_df, forecast_df)
    return df_real_time_forecast

# Jika run langsung, contoh:
if __name__ == "__main__":
    df_input = pd.read_excel('dataset.xlsx')
    df_result = run_pipeline_from_df(df_input)
    df_result.to_excel('hasil_forecast.xlsx', index=False)
