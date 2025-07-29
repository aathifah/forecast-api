import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

# Import forecasting functions tanpa pmdarima
try:
    from forecast_service_alt import (
        forecast_ma6, forecast_wma, forecast_ets, forecast_arima,
        forecast_ml, models
    )
    FORECAST_AVAILABLE = True
except ImportError:
    st.error("❌ Modul forecast_service_alt tidak ditemukan")
    FORECAST_AVAILABLE = False

warnings.filterwarnings("ignore")

# Konfigurasi halaman
st.set_page_config(
    page_title="Forecast Parts App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS untuk styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📊 Forecast Parts Application</h1>', unsafe_allow_html=True)
st.markdown("### Upload dataset Excel dan dapatkan prediksi permintaan part number")

# Sidebar
with st.sidebar:
    st.header("📋 Instruksi")
    st.markdown("""
    1. **Siapkan file Excel** dengan kolom:
       - `PART_NO` (Part Number)
       - `MONTH` (Bulan dalam format YYYY-MM)
       - `ORIGINAL_SHIPPING_QTY` (Jumlah permintaan)
    
    2. **Upload file** di bawah ini
    
    3. **Lihat hasil** forecasting di dashboard
    """)
    
    st.header("ℹ️ Tentang Aplikasi")
    st.markdown("""
    Aplikasi ini menggunakan **Machine Learning** untuk:
    - **MA6**: Moving Average 6 bulan
    - **WMA**: Weighted Moving Average
    - **ETS**: Exponential Smoothing
    - **ARIMA**: Auto-regressive model (statsmodels)
    - **ML Models**: Linear Regression, Random Forest, XGBoost
    """)

# File upload
uploaded_file = st.file_uploader(
    "📁 Upload file Excel (.xlsx)",
    type=['xlsx'],
    help="Pilih file Excel dengan kolom PART_NO, MONTH, ORIGINAL_SHIPPING_QTY"
)

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_excel(uploaded_file)
        
        # Validasi kolom
        required_columns = ['PART_NO', 'MONTH', 'ORIGINAL_SHIPPING_QTY']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"❌ Kolom yang diperlukan tidak ditemukan: {missing_columns}")
            st.stop()
        
        # Tampilkan preview data
        st.success("✅ File berhasil diupload!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Unique Parts", df['PART_NO'].nunique())
        with col3:
            st.metric("Date Range", f"{df['MONTH'].min()} - {df['MONTH'].max()}")
        
        # Proses data
        with st.spinner("🔄 Memproses data..."):
            # Konversi MONTH ke datetime
            df['MONTH'] = pd.to_datetime(df['MONTH'].astype(str), format='%Y%m')
            
            # Group by PART_NO dan MONTH
            df_processed = df.groupby(['PART_NO', 'MONTH'], as_index=False).agg({
                'ORIGINAL_SHIPPING_QTY': 'sum'
            })
            
            # Sort by PART_NO dan MONTH
            df_processed = df_processed.sort_values(['PART_NO', 'MONTH'])
        
        st.success("✅ Data berhasil diproses!")
        
        # Tampilkan preview data yang diproses
        st.subheader("📋 Preview Data yang Diproses")
        st.dataframe(df_processed.head(10))
        
        # Forecasting
        if st.button("🚀 Jalankan Forecasting", type="primary"):
            if not FORECAST_AVAILABLE:
                st.error("❌ Modul forecasting tidak tersedia")
                st.stop()
            
            with st.spinner("🔄 Menjalankan forecasting..."):
                # Jalankan forecasting untuk setiap part
                results = []
                
                for part_no in df_processed['PART_NO'].unique():
                    part_data = df_processed[df_processed['PART_NO'] == part_no]
                    series = part_data['ORIGINAL_SHIPPING_QTY'].values
                    
                    if len(series) < 3:
                        continue
                    
                    # Jalankan semua model
                    forecasts = {}
                    for model_name, model_func in models.items():
                        try:
                            if model_name == 'ARIMA':
                                forecast = model_func(series)
                            else:
                                forecast = model_func(series)
                            forecasts[model_name] = forecast
                        except Exception as e:
                            forecasts[model_name] = np.nan
                    
                    # Hitung rata-rata forecast
                    valid_forecasts = [f for f in forecasts.values() if not np.isnan(f)]
                    avg_forecast = np.mean(valid_forecasts) if valid_forecasts else np.nan
                    
                    results.append({
                        'PART_NO': part_no,
                        'MA6': forecasts.get('MA6', np.nan),
                        'WMA': forecasts.get('WMA', np.nan),
                        'ETS': forecasts.get('ETS', np.nan),
                        'ARIMA': forecasts.get('ARIMA', np.nan),
                        'LINREG': forecasts.get('LINREG', np.nan),
                        'RF': forecasts.get('RF', np.nan),
                        'XGB': forecasts.get('XGB', np.nan),
                        'AVERAGE': avg_forecast
                    })
                
                # Buat DataFrame hasil
                forecast_df = pd.DataFrame(results)
                
                st.success(f"✅ Forecasting selesai! {len(forecast_df)} parts diproses.")
                
                # Tampilkan hasil
                st.subheader("📊 Hasil Forecasting")
                st.dataframe(forecast_df)
                
                # Download hasil
                csv = forecast_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Hasil (CSV)",
                    data=csv,
                    file_name=f"forecast_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Visualisasi
                if len(forecast_df) > 0:
                    st.subheader("📈 Visualisasi Hasil")
                    
                    # Chart per model
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=('MA6 vs WMA', 'ETS vs ARIMA', 'LINREG vs RF', 'XGB vs Average'),
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    # MA6 vs WMA
                    fig.add_trace(
                        go.Scatter(x=forecast_df['MA6'], y=forecast_df['WMA'], 
                                 mode='markers', name='MA6 vs WMA'),
                        row=1, col=1
                    )
                    
                    # ETS vs ARIMA
                    fig.add_trace(
                        go.Scatter(x=forecast_df['ETS'], y=forecast_df['ARIMA'], 
                                 mode='markers', name='ETS vs ARIMA'),
                        row=1, col=2
                    )
                    
                    # LINREG vs RF
                    fig.add_trace(
                        go.Scatter(x=forecast_df['LINREG'], y=forecast_df['RF'], 
                                 mode='markers', name='LINREG vs RF'),
                        row=2, col=1
                    )
                    
                    # XGB vs Average
                    fig.add_trace(
                        go.Scatter(x=forecast_df['XGB'], y=forecast_df['AVERAGE'], 
                                 mode='markers', name='XGB vs Average'),
                        row=2, col=2
                    )
                    
                    fig.update_layout(height=600, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Histogram distribusi forecast
                    fig_hist = go.Figure()
                    for col in ['MA6', 'WMA', 'ETS', 'ARIMA', 'LINREG', 'RF', 'XGB']:
                        valid_data = forecast_df[col].dropna()
                        if len(valid_data) > 0:
                            fig_hist.add_trace(go.Histogram(x=valid_data, name=col))
                    
                    fig_hist.update_layout(
                        title="Distribusi Forecast per Model",
                        xaxis_title="Nilai Forecast",
                        yaxis_title="Frekuensi"
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
    
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        st.exception(e) 
