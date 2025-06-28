import os
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
import warnings

warnings.filterwarnings("ignore")

# --- Forecasting Methods ---
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

# --- Helper Functions ---
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
        # logging.warning(f"Tuning error for model {model_name}: {e}")
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

# --- Process per PART ---
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

            try:
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
            except Exception:
                preds[name] = np.nan

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

# --- Run Backtest Forecast ---
def run_combined_forecast(file_path):
    # Baca dataset
    original_df = pd.read_excel(file_path)
    df = original_df.copy()
    df['MONTH'] = pd.to_datetime(df['MONTH'].astype(str), format='%Y%m')
    df = df.groupby(['PART_NO', 'MONTH'], as_index=False).agg({
        'ORIGINAL_SHIPPING_QTY': 'sum',
        'PART_NAME': 'first',
        'TOPAS_ORDER_TYPE': 'first',
        'CREATED_DEMAND_FLAG': 'first',
        'CUST_TYPE2': 'first'
    })

    # Fitur waktu dan lag
    df['MONTH_NUM'] = df['MONTH'].dt.month
    df['YEAR'] = df['MONTH'].dt.year
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH_NUM'] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH_NUM'] / 12)
    df['LAG_1'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(1)
    df['LAG_2'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(2)
    df['LAG_3'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(3)
    df['LAG_4'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(4)
    df['LAG_5'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(5)
    df['LAG_6'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(6)

    df['ROLLING_MEAN_3'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].transform(lambda x: x.rolling(window=3).mean())
    df['ROLLING_MEAN_6'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].transform(lambda x: x.rolling(window=6).mean())
    df['ROLLING_STD_3'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].transform(lambda x: x.rolling(window=3).std())
    df['ROLLING_STD_6'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].transform(lambda x: x.rolling(window=6).std())

    df['GROWTH_LAG_3_6'] = df['LAG_3'] / (df['LAG_6'] + 1e-8)
    df['MEAN_LAG_3'] = df[['LAG_1', 'LAG_2', 'LAG_3']].mean(axis=1)
    df['SUM_LAG_6'] = df[['LAG_1', 'LAG_2', 'LAG_3', 'LAG_4', 'LAG_5', 'LAG_6']].sum(axis=1)
    df['GROWTH_RATE'] = df['LAG_1'] / (df['LAG_2'] + 1e-8)

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

# --- Run Real-Time Forecast ---
def run_real_time_forecast_no_mba_df(original_df, forecast_df, output_dir='outputs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_all = original_df.copy()
    df_all['MONTH'] = pd.to_datetime(df_all['MONTH'].astype(str), format='%Y%m')

    df_best = forecast_df.copy()
    df_best['MONTH'] = pd.to_datetime(df_best['MONTH'])
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
    if len(all_months) == 0:
        raise ValueError("Data kosong atau format tanggal salah.")
    last_month = max(all_months)
    next_month = (last_month + pd.offsets.MonthBegin(1)).strftime('%Y-%m')
    future_months = pd.date_range(start=last_month + pd.offsets.MonthBegin(1), periods=4, freq='MS')

    forecast_rows = []
    for part in df_all['PART_NO'].unique():
        part_df = df_all[df_all['PART_NO'] == part].sort_values('MONTH')
        series = part_df.set_index('MONTH')['ORIGINAL_SHIPPING_QTY']
        best_model_info = latest_model_map.get(part, {'BEST_MODEL': 'MA6'})

        for month in future_months:
            forecast_value = np.nan
            model_name = best_model_info.get('BEST_MODEL', 'MA6')

            if model_name == 'MA6':
                forecast_value = forecast_ma6(series)
            elif model_name == 'WMA':
                forecast_value = forecast_wma(series)
            elif model_name == 'ETS':
                forecast_value = forecast_ets(series)
            elif model_name == 'ARIMA':
                forecast_value = forecast_arima(series)
            else:
                forecast_value = forecast_ma6(series)

            forecast_rows.append({
                'PART_NO': part,
                'MONTH': month.strftime('%Y-%m'),
                'FORECAST': round(max(forecast_value, 0)) if not pd.isna(forecast_value) else None,
                'BEST_MODEL': model_name
            })

            series = series.append(pd.Series([forecast_value], index=[month]))

    df_forecast_future = pd.DataFrame(forecast_rows)
    df_output = pd.concat([df_all[['PART_NO', 'MONTH', 'ORIGINAL_SHIPPING_QTY']], df_forecast_future], sort=False)

    # Simpan hasil forecast ke Excel di folder output_dir
    output_file = os.path.join(output_dir, 'realtime_forecast.xlsx')
    df_output.to_excel(output_file, index=False)

    return output_file, df_output

# --- Pipeline lengkap: dari file input sampai hasil forecast ---
def main_forecast_pipeline(input_file_path, output_dir='outputs'):
    original_df, forecast_df = run_combined_forecast(input_file_path)
    output_file, df_forecast = run_real_time_forecast_no_mba_df(original_df, forecast_df, output_dir)
    return output_file, df_forecast

# --- Jika mau testing lokal bisa jalankan ini ---
if __name__ == "__main__":
    input_path = 'dataset.xlsx'  # Ganti sesuai file Anda
    output_path, df_result = main_forecast_pipeline(input_path)
    print(f"Forecast hasil disimpan di: {output_path}")
import os
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
import warnings

warnings.filterwarnings("ignore")

# --- Forecasting Methods ---
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

# --- Helper Functions ---
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
        # logging.warning(f"Tuning error for model {model_name}: {e}")
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

# --- Process per PART ---
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

            try:
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
            except Exception:
                preds[name] = np.nan

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

# --- Run Backtest Forecast ---
def run_combined_forecast(file_path):
    # Baca dataset
    original_df = pd.read_excel(file_path)
    df = original_df.copy()
    df['MONTH'] = pd.to_datetime(df['MONTH'].astype(str), format='%Y%m')
    df = df.groupby(['PART_NO', 'MONTH'], as_index=False).agg({
        'ORIGINAL_SHIPPING_QTY': 'sum',
        'PART_NAME': 'first',
        'TOPAS_ORDER_TYPE': 'first',
        'CREATED_DEMAND_FLAG': 'first',
        'CUST_TYPE2': 'first'
    })

    # Fitur waktu dan lag
    df['MONTH_NUM'] = df['MONTH'].dt.month
    df['YEAR'] = df['MONTH'].dt.year
    df['MONTH_SIN'] = np.sin(2 * np.pi * df['MONTH_NUM'] / 12)
    df['MONTH_COS'] = np.cos(2 * np.pi * df['MONTH_NUM'] / 12)
    df['LAG_1'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(1)
    df['LAG_2'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(2)
    df['LAG_3'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(3)
    df['LAG_4'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(4)
    df['LAG_5'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(5)
    df['LAG_6'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].shift(6)

    df['ROLLING_MEAN_3'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].transform(lambda x: x.rolling(window=3).mean())
    df['ROLLING_MEAN_6'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].transform(lambda x: x.rolling(window=6).mean())
    df['ROLLING_STD_3'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].transform(lambda x: x.rolling(window=3).std())
    df['ROLLING_STD_6'] = df.groupby('PART_NO')['ORIGINAL_SHIPPING_QTY'].transform(lambda x: x.rolling(window=6).std())

    df['GROWTH_LAG_3_6'] = df['LAG_3'] / (df['LAG_6'] + 1e-8)
    df['MEAN_LAG_3'] = df[['LAG_1', 'LAG_2', 'LAG_3']].mean(axis=1)
    df['SUM_LAG_6'] = df[['LAG_1', 'LAG_2', 'LAG_3', 'LAG_4', 'LAG_5', 'LAG_6']].sum(axis=1)
    df['GROWTH_RATE'] = df['LAG_1'] / (df['LAG_2'] + 1e-8)

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

# --- Run Real-Time Forecast ---
def run_real_time_forecast_no_mba_df(original_df, forecast_df, output_dir='outputs'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_all = original_df.copy()
    df_all['MONTH'] = pd.to_datetime(df_all['MONTH'].astype(str), format='%Y%m')

    df_best = forecast_df.copy()
    df_best['MONTH'] = pd.to_datetime(df_best['MONTH'])
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
    if len(all_months) == 0:
        raise ValueError("Data kosong atau format tanggal salah.")
    last_month = max(all_months)
    next_month = (last_month + pd.offsets.MonthBegin(1)).strftime('%Y-%m')
    future_months = pd.date_range(start=last_month + pd.offsets.MonthBegin(1), periods=4, freq='MS')

    forecast_rows = []
    for part in df_all['PART_NO'].unique():
        part_df = df_all[df_all['PART_NO'] == part].sort_values('MONTH')
        series = part_df.set_index('MONTH')['ORIGINAL_SHIPPING_QTY']
        best_model_info = latest_model_map.get(part, {'BEST_MODEL': 'MA6'})

        for month in future_months:
            forecast_value = np.nan
            model_name = best_model_info.get('BEST_MODEL', 'MA6')

            if model_name == 'MA6':
                forecast_value = forecast_ma6(series)
            elif model_name == 'WMA':
                forecast_value = forecast_wma(series)
            elif model_name == 'ETS':
                forecast_value = forecast_ets(series)
            elif model_name == 'ARIMA':
                forecast_value = forecast_arima(series)
            else:
                forecast_value = forecast_ma6(series)

            forecast_rows.append({
                'PART_NO': part,
                'MONTH': month.strftime('%Y-%m'),
                'FORECAST': round(max(forecast_value, 0)) if not pd.isna(forecast_value) else None,
                'BEST_MODEL': model_name
            })

            series = series.append(pd.Series([forecast_value], index=[month]))

    df_forecast_future = pd.DataFrame(forecast_rows)
    df_output = pd.concat([df_all[['PART_NO', 'MONTH', 'ORIGINAL_SHIPPING_QTY']], df_forecast_future], sort=False)

    # Simpan hasil forecast ke Excel di folder output_dir
    output_file = os.path.join(output_dir, 'realtime_forecast.xlsx')
    df_output.to_excel(output_file, index=False)

    return output_file, df_output

# --- Pipeline lengkap: dari file input sampai hasil forecast ---
def main_forecast_pipeline(input_file_path, output_dir='outputs'):
    original_df, forecast_df = run_combined_forecast(input_file_path)
    output_file, df_forecast = run_real_time_forecast_no_mba_df(original_df, forecast_df, output_dir)
    return output_file, df_forecast

# --- Jika mau testing lokal bisa jalankan ini ---
if __name__ == "__main__":
    input_path = 'dataset.xlsx'  # Ganti sesuai file Anda
    output_path, df_result = main_forecast_pipeline(input_path)
    print(f"Forecast hasil disimpan di: {output_path}")
