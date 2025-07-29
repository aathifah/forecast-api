import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from forecast_service import process_forecast, run_combined_forecast, run_real_time_forecast

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
    - **ARIMA**: Auto-regressive model
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
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Records", len(df))
            st.metric("Unique Parts", df['PART_NO'].nunique())
        
        with col2:
            st.metric("Date Range", f"{df['MONTH'].min()} - {df['MONTH'].max()}")
            st.metric("Total Demand", f"{df['ORIGINAL_SHIPPING_QTY'].sum():,.0f}")
        
        # Preview data
        with st.expander("📋 Preview Data"):
            st.dataframe(df.head(10))
        
        # Tombol proses
        if st.button("🚀 Mulai Forecasting", type="primary"):
            with st.spinner("🔄 Memproses forecasting..."):
                try:
                    # Proses forecasting
                    result = process_forecast(df)
                    
                    if result["status"] == "success":
                        st.success("✅ Forecasting berhasil!")
                        
                        # Ambil data untuk dashboard
                        df_processed, forecast_df = run_combined_forecast(df)
                        real_time_forecast = run_real_time_forecast(df_processed, forecast_df)
                        
                        # Tampilkan hasil
                        st.header("📊 Hasil Forecasting")
                        
                        # Metrics
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Forecasts", len(real_time_forecast))
                        with col2:
                            st.metric("Total Parts", real_time_forecast['PART_NO'].nunique())
                        with col3:
                            st.metric("Forecast Months", len(real_time_forecast['MONTH'].unique()))
                        with col4:
                            avg_error = real_time_forecast['ERROR_BACKTEST'].str.replace('%', '').astype(float).mean()
                            st.metric("Avg Error", f"{avg_error:.1f}%")
                        
                        # Tabs untuk berbagai visualisasi
                        tab1, tab2, tab3, tab4 = st.tabs(["📈 Real-time Forecast", "📊 Backtest Results", "🎯 Model Performance", "📥 Download Results"])
                        
                        with tab1:
                            st.subheader("Real-time Forecast (4 Bulan Kedepan)")
                            
                            # Filter part number
                            part_filter = st.selectbox(
                                "Pilih Part Number:",
                                options=['All'] + real_time_forecast['PART_NO'].unique().tolist()
                            )
                            
                            if part_filter != 'All':
                                filtered_data = real_time_forecast[real_time_forecast['PART_NO'] == part_filter]
                            else:
                                filtered_data = real_time_forecast
                            
                            # Chart forecast
                            fig = go.Figure()
                            
                            for _, row in filtered_data.iterrows():
                                fig.add_trace(go.Bar(
                                    name=f"{row['PART_NO']} - {row['MONTH']}",
                                    x=[row['MONTH']],
                                    y=[row['FORECAST_NEUTRAL']],
                                    text=[f"{row['FORECAST_NEUTRAL']}"],
                                    textposition='auto',
                                    hovertemplate=f"<b>{row['PART_NO']}</b><br>" +
                                                f"Month: {row['MONTH']}<br>" +
                                                f"Neutral: {row['FORECAST_NEUTRAL']}<br>" +
                                                f"Optimist: {row['FORECAST_OPTIMIST']}<br>" +
                                                f"Pessimist: {row['FORECAST_PESSIMIST']}<br>" +
                                                f"Best Model: {row['BEST_MODEL']}<br>" +
                                                f"Error: {row['ERROR_BACKTEST']}<extra></extra>"
                                ))
                            
                            fig.update_layout(
                                title="Forecast by Month",
                                xaxis_title="Month",
                                yaxis_title="Forecast Quantity",
                                showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with tab2:
                            st.subheader("Backtest Results (4 Bulan Terakhir)")
                            
                            # Chart backtest
                            fig_backtest = go.Figure()
                            
                            for _, row in forecast_df.iterrows():
                                fig_backtest.add_trace(go.Scatter(
                                    name=f"{row['PART_NO']} - {row['MONTH']}",
                                    x=[row['MONTH']],
                                    y=[row['FORECAST']],
                                    mode='markers+text',
                                    text=[f"{row['FORECAST']}"],
                                    textposition='top center',
                                    hovertemplate=f"<b>{row['PART_NO']}</b><br>" +
                                                f"Month: {row['MONTH']}<br>" +
                                                f"Forecast: {row['FORECAST']}<br>" +
                                                f"Actual: {row['ACTUAL']}<br>" +
                                                f"Error: {row['HYBRID_ERROR']}<br>" +
                                                f"Best Model: {row['BEST_MODEL']}<extra></extra>"
                                ))
                            
                            fig_backtest.update_layout(
                                title="Backtest Results",
                                xaxis_title="Month",
                                yaxis_title="Quantity",
                                showlegend=False
                            )
                            st.plotly_chart(fig_backtest, use_container_width=True)
                        
                        with tab3:
                            st.subheader("Model Performance Analysis")
                            
                            # Model distribution
                            model_counts = real_time_forecast['BEST_MODEL'].value_counts()
                            fig_model = px.pie(
                                values=model_counts.values,
                                names=model_counts.index,
                                title="Distribution of Best Models"
                            )
                            st.plotly_chart(fig_model, use_container_width=True)
                            
                            # Error analysis
                            error_data = real_time_forecast.copy()
                            error_data['ERROR_NUMERIC'] = error_data['ERROR_BACKTEST'].str.replace('%', '').astype(float)
                            
                            fig_error = px.histogram(
                                error_data,
                                x='ERROR_NUMERIC',
                                title="Error Distribution",
                                labels={'ERROR_NUMERIC': 'Error (%)', 'count': 'Frequency'}
                            )
                            st.plotly_chart(fig_error, use_container_width=True)
                        
                        with tab4:
                            st.subheader("Download Results")
                            
                            # Create Excel file
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                real_time_forecast.to_excel(writer, sheet_name='Forecast_Results', index=False)
                                forecast_df.to_excel(writer, sheet_name='Backtest_Results', index=False)
                                
                                # Summary sheet
                                summary = real_time_forecast.groupby('PART_NO').agg({
                                    'BEST_MODEL': lambda x: list(set(x)),
                                    'FORECAST_NEUTRAL': ['mean', 'min', 'max', 'std'],
                                    'ERROR_BACKTEST': lambda x: np.mean([float(str(e).replace('%', '')) for e in x if str(e).replace('%', '').replace('.', '').isdigit()])
                                }).round(2)
                                summary.to_excel(writer, sheet_name='Summary')
                            
                            output.seek(0)
                            
                            # Download button
                            st.download_button(
                                label="📥 Download Excel Results",
                                data=output.getvalue(),
                                file_name=f"forecast_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                            st.info("💡 File Excel berisi 3 sheet: Forecast_Results, Backtest_Results, dan Summary")
                    
                    else:
                        st.error(f"❌ Forecasting gagal: {result.get('message', 'Unknown error')}")
                
                except Exception as e:
                    st.error(f"❌ Error saat memproses: {str(e)}")
                    st.exception(e)
    
    except Exception as e:
        st.error(f"❌ Error saat membaca file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with ❤️ using Streamlit | Machine Learning Forecasting App</p>
    <p>Support: MA6, WMA, ETS, ARIMA, Linear Regression, Random Forest, XGBoost</p>
</div>
""", unsafe_allow_html=True) 
