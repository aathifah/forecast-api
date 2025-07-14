console.log('script.js loaded');

// Inisialisasi elemen untuk download forecast workflow
const processBtn = document.getElementById('process-btn');
const fileInput = document.getElementById('download-file-input');
const statusDiv = document.getElementById('download-status');
const progressBarContainer = document.getElementById('download-progress-bar-container');
const progressBar = document.getElementById('download-progress-bar');
const progressLabel = document.getElementById('download-progress-label');

// Debug: cek apakah elemen ditemukan
if (!processBtn) {
  console.error('processBtn is null - tombol proses tidak ditemukan di HTML!');
}
if (!fileInput) {
  console.error('fileInput is null - input file tidak ditemukan di HTML!');
}
if (!statusDiv || !progressBarContainer || !progressBar || !progressLabel) {
  console.error('Ada elemen status/progress yang tidak ditemukan di HTML!');
}

// Fungsi validasi file Excel
function validateExcelFile(file) {
  // Cek ekstensi file
  const fileName = file.name.toLowerCase();
  const validExtensions = ['.xlsx'];
  
  // Cek apakah file memiliki ekstensi yang valid
  const hasValidExtension = validExtensions.some(ext => fileName.endsWith(ext));
  
  if (!hasValidExtension) {
    return {
      isValid: false,
      message: 'Tipe File Kamu Bukan Excel. Pastikan File Tipe Excel Saja yang Kamu Unggah!'
    };
  }
  
  // Cek MIME type (opsional, untuk validasi tambahan)
  const validMimeTypes = [
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'application/vnd.ms-excel'
  ];
  
  if (file.type && !validMimeTypes.includes(file.type)) {
    return {
      isValid: false,
      message: 'Tipe File Kamu Bukan Excel. Pastikan File Tipe Excel Saja yang Kamu Unggah!'
    };
  }
  
  return {
    isValid: true,
    message: ''
  };
}

// Update state tombol berdasarkan file yang dipilih
function updateProcessBtnState() {
  if (processBtn && fileInput) {
    const hasFile = fileInput.files.length > 0;
    const isValidFile = hasFile && validateExcelFile(fileInput.files[0]).isValid;
    
    processBtn.disabled = !hasFile || !isValidFile;
    
    // Reset classes
    statusDiv.className = '';
    
    // Tampilkan pesan error jika file tidak valid
    if (hasFile && !isValidFile) {
      const validation = validateExcelFile(fileInput.files[0]);
      statusDiv.textContent = validation.message;
      statusDiv.classList.add('error');
    } else if (hasFile && isValidFile) {
      statusDiv.textContent = 'File Excel valid, siap untuk diproses.';
      statusDiv.classList.add('success');
    } else {
      statusDiv.textContent = '';
    }
  }
}

// Event listener untuk file input
if (fileInput) {
  fileInput.addEventListener('change', updateProcessBtnState);
}

// ====== DASHBOARD INTERAKTIF DINAMIS ======
// Variabel global untuk data hasil forecast
let originalData = [];
let backtestData = [];
let realtimeData = [];

// Chart.js instance
let realtimeBarChart = null;
let realtimeLineChart = null;
let backtestLineChart = null;
let backtestBarChart = null;

// Dummy data (ganti dengan data dari backend nanti)
const dummyOriginal = [
  { PART_NO: '0888581844', MONTH: '2025-01', ORIGINAL_SHIPPING_QTY: 20 },
  { PART_NO: '0888581844', MONTH: '2025-02', ORIGINAL_SHIPPING_QTY: 25 },
  { PART_NO: '0888581844', MONTH: '2025-03', ORIGINAL_SHIPPING_QTY: 23 },
  { PART_NO: '0888581844', MONTH: '2025-04', ORIGINAL_SHIPPING_QTY: 22 },
  { PART_NO: '0888581844', MONTH: '2025-05', ORIGINAL_SHIPPING_QTY: 28 },
  { PART_NO: '0888581844', MONTH: '2025-06', ORIGINAL_SHIPPING_QTY: 30 },
  { PART_NO: '0888581844', MONTH: '2025-07', ORIGINAL_SHIPPING_QTY: 32 },
  { PART_NO: '0888581844', MONTH: '2025-08', ORIGINAL_SHIPPING_QTY: 33 },
  { PART_NO: '0999999999', MONTH: '2025-01', ORIGINAL_SHIPPING_QTY: 10 },
  { PART_NO: '0999999999', MONTH: '2025-02', ORIGINAL_SHIPPING_QTY: 12 },
  { PART_NO: '0999999999', MONTH: '2025-03', ORIGINAL_SHIPPING_QTY: 15 },
  { PART_NO: '0999999999', MONTH: '2025-04', ORIGINAL_SHIPPING_QTY: 18 },
  { PART_NO: '0999999999', MONTH: '2025-05', ORIGINAL_SHIPPING_QTY: 20 },
  { PART_NO: '0999999999', MONTH: '2025-06', ORIGINAL_SHIPPING_QTY: 22 },
  { PART_NO: '0999999999', MONTH: '2025-07', ORIGINAL_SHIPPING_QTY: 25 },
  { PART_NO: '0999999999', MONTH: '2025-08', ORIGINAL_SHIPPING_QTY: 28 },
];
const dummyRealtime = [
  { PART_NO: '0888581844', MONTH: '2025-05', FORECAST_OPTIMIST: 32, FORECAST_NEUTRAL: 28, FORECAST_PESSIMIST: 24 },
  { PART_NO: '0888581844', MONTH: '2025-06', FORECAST_OPTIMIST: 34, FORECAST_NEUTRAL: 30, FORECAST_PESSIMIST: 26 },
  { PART_NO: '0888581844', MONTH: '2025-07', FORECAST_OPTIMIST: 36, FORECAST_NEUTRAL: 32, FORECAST_PESSIMIST: 28 },
  { PART_NO: '0888581844', MONTH: '2025-08', FORECAST_OPTIMIST: 38, FORECAST_NEUTRAL: 33, FORECAST_PESSIMIST: 29 },
  { PART_NO: '0999999999', MONTH: '2025-05', FORECAST_OPTIMIST: 22, FORECAST_NEUTRAL: 20, FORECAST_PESSIMIST: 18 },
  { PART_NO: '0999999999', MONTH: '2025-06', FORECAST_OPTIMIST: 25, FORECAST_NEUTRAL: 22, FORECAST_PESSIMIST: 20 },
  { PART_NO: '0999999999', MONTH: '2025-07', FORECAST_OPTIMIST: 28, FORECAST_NEUTRAL: 25, FORECAST_PESSIMIST: 22 },
  { PART_NO: '0999999999', MONTH: '2025-08', FORECAST_OPTIMIST: 30, FORECAST_NEUTRAL: 28, FORECAST_PESSIMIST: 25 },
];
const dummyBacktest = [
  { PART_NO: '0888581844', MONTH: '2025-01', FORECAST: 19, ACTUAL: 20, BEST_MODEL: 'LINREG', HYBRID_ERROR: '5%' },
  { PART_NO: '0888581844', MONTH: '2025-02', FORECAST: 24, ACTUAL: 25, BEST_MODEL: 'RF', HYBRID_ERROR: '4%' },
  { PART_NO: '0888581844', MONTH: '2025-03', FORECAST: 22, ACTUAL: 23, BEST_MODEL: 'XGB', HYBRID_ERROR: '4%' },
  { PART_NO: '0888581844', MONTH: '2025-04', FORECAST: 21, ACTUAL: 22, BEST_MODEL: 'ARIMA', HYBRID_ERROR: '5%' },
  { PART_NO: '0999999999', MONTH: '2025-01', FORECAST: 9, ACTUAL: 10, BEST_MODEL: 'LINREG', HYBRID_ERROR: '10%' },
  { PART_NO: '0999999999', MONTH: '2025-02', FORECAST: 11, ACTUAL: 12, BEST_MODEL: 'RF', HYBRID_ERROR: '8%' },
  { PART_NO: '0999999999', MONTH: '2025-03', FORECAST: 14, ACTUAL: 15, BEST_MODEL: 'XGB', HYBRID_ERROR: '7%' },
  { PART_NO: '0999999999', MONTH: '2025-04', FORECAST: 17, ACTUAL: 18, BEST_MODEL: 'ARIMA', HYBRID_ERROR: '6%' },
];

// Helper: ambil 12 bulan terakhir dari array string bulan (format YYYY-MM atau YYYY-MM-DD...)
function getLast12Months(monthsArr) {
  // Sort as date, ambil 12 terakhir
  const sorted = monthsArr.slice().sort((a, b) => new Date(a) - new Date(b));
  return sorted.slice(-12);
}

// Helper: filter data by month range (YYYY-MM)
function filterByMonthRange(data, start, end) {
  if (!start && !end) return data;
  return data.filter(row => {
    const m = (row.MONTH || '').slice(0, 7); // YYYY-MM
    return (!start || m >= start) && (!end || m <= end);
  });
}

// Helper: konversi ke format YYYY-MM
function toYearMonth(str) {
  if (!str) return '';
  return str.slice(0, 7);
}

// Fungsi filter data berdasarkan partno dan bulan
function getFilteredData(data, partno, selectedMonths) {
  let filtered = data;
  if (partno) {
    filtered = filtered.filter(d => d.PART_NO && d.PART_NO.toLowerCase().includes(partno.toLowerCase()));
  }
  if (selectedMonths && selectedMonths.length > 0) {
    filtered = filtered.filter(d => selectedMonths.includes(d.MONTH));
  }
  return filtered;
}

// Fungsi render cards forecast
function renderForecastCards(filteredData) {
  const totalOptimist = filteredData.reduce((sum, d) => sum + (Number(d.FORECAST_OPTIMIST) || 0), 0);
  const totalNeutral = filteredData.reduce((sum, d) => sum + (Number(d.FORECAST_NEUTRAL) || 0), 0);
  const totalPessimist = filteredData.reduce((sum, d) => sum + (Number(d.FORECAST_PESSIMIST) || 0), 0);
  document.getElementById('card-optimist-value').textContent = totalOptimist.toLocaleString();
  document.getElementById('card-neutral-value').textContent = totalNeutral.toLocaleString();
  document.getElementById('card-pessimist-value').textContent = totalPessimist.toLocaleString();
}

function renderRealtimeDashboard() {
  const partno = document.getElementById('partno-input').value.trim();
  const selectedMonths = getSelectedMonthsRealtime();
  let data = getFilteredData(realtimeData, partno, selectedMonths);
  // Pastikan hanya 1 data per bulan (ambil data terakhir jika duplikat)
  const monthMap = {};
  data.forEach(d => { monthMap[d.MONTH] = d; });
  const uniqueMonths = Object.keys(monthMap).sort((a, b) => new Date(a) - new Date(b));
  const uniqueData = uniqueMonths.map(m => monthMap[m]);
  renderForecastCards(uniqueData);
  // Column chart
  const barLabels = uniqueData.map(d => toYearMonth(d.MONTH));
  const optimist = uniqueData.map(d => d.FORECAST_OPTIMIST);
  const neutral = uniqueData.map(d => d.FORECAST_NEUTRAL);
  const pessimist = uniqueData.map(d => d.FORECAST_PESSIMIST);
  if (realtimeBarChart) realtimeBarChart.destroy();
  const ctxBar = document.getElementById('realtime-bar-chart').getContext('2d');
  realtimeBarChart = new Chart(ctxBar, {
    type: 'bar',
    data: {
      labels: barLabels,
      datasets: [
        { label: 'Optimist', data: optimist, backgroundColor: 'rgba(54, 162, 235, 0.6)' },
        { label: 'Neutral', data: neutral, backgroundColor: 'rgba(75, 192, 192, 0.4)' },
        { label: 'Pessimist', data: pessimist, backgroundColor: 'rgba(255, 99, 132, 0.3)' }
      ]
    },
    options: { responsive: true, plugins: { legend: { position: 'top' } } }
  });
  // Line chart (8 bulan terakhir history + 4 bulan real time forecast)
  let history = originalData;
  if (partno) history = history.filter(d => d.PART_NO && d.PART_NO.toLowerCase().includes(partno.toLowerCase()));
  const allHistoryMonths = Array.from(new Set(history.map(d => d.MONTH))).sort((a, b) => new Date(a) - new Date(b));
  const last8History = allHistoryMonths.slice(-8);
  history = history.filter(d => last8History.includes(d.MONTH));
  let forecast = uniqueData;
  const forecastMonths = uniqueMonths;
  const last4Forecast = forecastMonths.slice(-4);
  forecast = forecast.filter(d => last4Forecast.includes(d.MONTH));
  const lineLabels = [...last8History, ...last4Forecast].map(toYearMonth);
  const historyMap = Object.fromEntries(history.map(d => [toYearMonth(d.MONTH), d.ORIGINAL_SHIPPING_QTY]));
  const optimistMap = Object.fromEntries(forecast.map(d => [toYearMonth(d.MONTH), d.FORECAST_OPTIMIST]));
  const neutralMap = Object.fromEntries(forecast.map(d => [toYearMonth(d.MONTH), d.FORECAST_NEUTRAL]));
  const pessimistMap = Object.fromEntries(forecast.map(d => [toYearMonth(d.MONTH), d.FORECAST_PESSIMIST]));
  const lineHistory = lineLabels.map(m => historyMap[m] || null);
  const lineOptimist = lineLabels.map(m => optimistMap[m] || null);
  const lineNeutral = lineLabels.map(m => neutralMap[m] || null);
  const linePessimist = lineLabels.map(m => pessimistMap[m] || null);
  if (realtimeLineChart) realtimeLineChart.destroy();
  const ctxLine = document.getElementById('realtime-line-chart').getContext('2d');
  realtimeLineChart = new Chart(ctxLine, {
    type: 'line',
    data: {
      labels: lineLabels,
      datasets: [
        { label: 'History', data: lineHistory, borderColor: '#aaa', backgroundColor: 'rgba(200,200,200,0.1)', tension: 0.2 },
        { label: 'Optimist', data: lineOptimist, borderColor: '#7c4dff', backgroundColor: 'rgba(124,77,255,0.1)', tension: 0.2 },
        { label: 'Neutral', data: lineNeutral, borderColor: '#00bcd4', backgroundColor: 'rgba(0,188,212,0.1)', tension: 0.2 },
        { label: 'Pessimist', data: linePessimist, borderColor: '#00e5ff', backgroundColor: 'rgba(0,229,255,0.1)', tension: 0.2 }
      ]
    },
    options: { responsive: true, plugins: { legend: { position: 'top' } } }
  });
}

function renderBacktestCards(filtered) {
  const qty = filtered.reduce((sum, d) => sum + (Number(d.FORECAST) || 0), 0);
  const errors = filtered.map(d => {
    if (typeof d.HYBRID_ERROR === 'string' && d.HYBRID_ERROR.includes('%')) {
      return parseFloat(d.HYBRID_ERROR.replace('%', ''));
    }
    return Number(d.HYBRID_ERROR) || 0;
  });
  const avgError = errors.length ? errors.reduce((a, b) => a + b, 0) / errors.length : 0;
  document.getElementById('card-backtest-qty-value').textContent = qty.toLocaleString();
  document.getElementById('card-backtest-error-value').textContent = avgError.toFixed(2) + '%';
}

function renderBacktestDashboard() {
  const partno = document.getElementById('partno-input').value.trim();
  const selectedMonths = getSelectedMonthsBacktest();
  let filtered = getFilteredData(backtestData, partno, selectedMonths);
  renderBacktestCards(filtered);
  // Bar chart: Best Model count
  const modelCounts = {};
  filtered.forEach(d => {
    if (d.BEST_MODEL) modelCounts[d.BEST_MODEL] = (modelCounts[d.BEST_MODEL] || 0) + 1;
  });
  const modelLabels = Object.keys(modelCounts);
  const modelData = Object.values(modelCounts);
  if (backtestBarChart) backtestBarChart.destroy();
  const ctxBar = document.getElementById('backtest-bar-chart').getContext('2d');
  backtestBarChart = new Chart(ctxBar, {
    type: 'bar',
    data: {
      labels: modelLabels,
      datasets: [
        { label: 'Best Model', data: modelData, backgroundColor: 'rgba(54, 162, 235, 0.6)' }
      ]
    },
    options: { responsive: true, plugins: { legend: { display: false } } }
  });
  // Line chart
  const monthsLine = Array.from(new Set(filtered.map(d => d.MONTH))).sort((a, b) => new Date(a) - new Date(b));
  const monthMap = {};
  filtered.forEach(d => { monthMap[d.MONTH] = d; });
  const forecast = monthsLine.map(m => monthMap[m] ? Number(monthMap[m].FORECAST) : null);
  const actual = monthsLine.map(m => monthMap[m] ? Number(monthMap[m].ACTUAL) : null);
  if (backtestLineChart) backtestLineChart.destroy();
  const ctxLine = document.getElementById('backtest-line-chart').getContext('2d');
  backtestLineChart = new Chart(ctxLine, {
    type: 'line',
    data: {
      labels: monthsLine.map(m => m.replace(/T.*$/, '')),
      datasets: [
        { label: 'Forecast', data: forecast, borderColor: '#2196f3', backgroundColor: 'rgba(33,150,243,0.1)', tension: 0.2 },
        { label: 'Actual', data: actual, borderColor: '#aaa', backgroundColor: 'rgba(200,200,200,0.1)', tension: 0.2 }
      ]
    },
    options: { responsive: true, plugins: { legend: { position: 'top' } } }
  });
}

// ===== FLATPICKR MONTH PICKER =====
let realtimeMonthPicker = null;
let backtestMonthPicker = null;
let selectedRealtimeMonths = [];
let selectedBacktestMonths = [];

function setupMonthPickers() {
  // Helper: konversi array bulan ke array tanggal pertama bulan (YYYY-MM-01)
  function monthsToDates(monthsArr) {
    return monthsArr.map(m => {
      if (m.length === 7) return m + '-01';
      if (m.length === 10) return m.slice(0, 7) + '-01';
      return m;
    });
  }
  // Real-time
  const realtimeMonths = Array.from(new Set(realtimeData.map(d => d.MONTH))).sort((a, b) => new Date(a) - new Date(b));
  if (realtimeMonthPicker) realtimeMonthPicker.destroy();
  realtimeMonthPicker = flatpickr('#realtime-month-picker', {
    plugins: [new monthSelectPlugin({ shorthand: true, dateFormat: 'Y-m', altFormat: 'F Y' })],
    mode: 'range',
    enable: monthsToDates(realtimeMonths),
    disableMobile: true,
    onChange: function(selectedDates, dateStrArr) {
      // Ambil bulan yang dipilih (format YYYY-MM)
      selectedRealtimeMonths = selectedDates.map(d => d.toISOString().slice(0, 7));
      renderRealtimeDashboard();
    }
  });
  // Backtest
  const backtestMonths = Array.from(new Set(backtestData.map(d => d.MONTH))).sort((a, b) => new Date(a) - new Date(b));
  if (backtestMonthPicker) backtestMonthPicker.destroy();
  backtestMonthPicker = flatpickr('#backtest-month-picker', {
    plugins: [new monthSelectPlugin({ shorthand: true, dateFormat: 'Y-m', altFormat: 'F Y' })],
    mode: 'range',
    enable: monthsToDates(backtestMonths),
    disableMobile: true,
    onChange: function(selectedDates, dateStrArr) {
      selectedBacktestMonths = selectedDates.map(d => d.toISOString().slice(0, 7));
      renderBacktestDashboard();
    }
  });
}

// Ganti getSelectedMonths agar ambil dari picker
function getSelectedMonthsRealtime() {
  if (!selectedRealtimeMonths || selectedRealtimeMonths.length === 0) {
    return Array.from(new Set(realtimeData.map(d => d.MONTH))).sort((a, b) => new Date(a) - new Date(b));
  }
  // Jika range, ambil semua bulan di antara
  if (selectedRealtimeMonths.length === 2) {
    const [start, end] = selectedRealtimeMonths;
    const all = Array.from(new Set(realtimeData.map(d => d.MONTH))).sort((a, b) => new Date(a) - new Date(b));
    return all.filter(m => m >= start && m <= end);
  }
  return selectedRealtimeMonths;
}
function getSelectedMonthsBacktest() {
  if (!selectedBacktestMonths || selectedBacktestMonths.length === 0) {
    return Array.from(new Set(backtestData.map(d => d.MONTH))).sort((a, b) => new Date(a) - new Date(b));
  }
  if (selectedBacktestMonths.length === 2) {
    const [start, end] = selectedBacktestMonths;
    const all = Array.from(new Set(backtestData.map(d => d.MONTH))).sort((a, b) => new Date(a) - new Date(b));
    return all.filter(m => m >= start && m <= end);
  }
  return selectedBacktestMonths;
}

// Setup event listeners
function setupDashboardListeners() {
  document.getElementById('partno-input').addEventListener('input', () => {
    renderRealtimeDashboard();
    renderBacktestDashboard();
  });
  // Tidak perlu event change untuk picker, sudah di-handle onChange Flatpickr
}

// Event listener input PartNo & Bulan (multiple select)
function setupDashboardListeners() {
  document.getElementById('partno-input').addEventListener('input', renderRealtimeDashboard);
  document.getElementById('realtime-month-start').addEventListener('change', renderRealtimeDashboard);
  document.getElementById('realtime-month-end').addEventListener('change', renderRealtimeDashboard);
}

// Event listener untuk DOM loaded
window.addEventListener('DOMContentLoaded', () => {
  if (processBtn) processBtn.disabled = false;
  updateProcessBtnState();
  console.log('DOMContentLoaded: script.js loaded, processBtn:', processBtn);
  // Ganti dengan data dari backend nanti
  originalData = dummyOriginal;
  backtestData = dummyBacktest;
  realtimeData = dummyRealtime;
  setupMonthPickers();
  renderRealtimeDashboard();
  renderBacktestDashboard();
  setupDashboardListeners();
});

// Event listener untuk tombol proses forecast
if (processBtn && fileInput && statusDiv && progressBarContainer && progressBar && progressLabel) {
  processBtn.disabled = false;
  processBtn.addEventListener('click', async function() {
    console.log('Tombol forecast diklik!');
    statusDiv.textContent = '';
    statusDiv.className = ''; // Reset classes
    
    if (!fileInput.files.length) {
      statusDiv.textContent = 'Pilih file Excel terlebih dahulu.';
      statusDiv.classList.add('error');
      return;
    }

    // Validasi file sebelum upload
    const validation = validateExcelFile(fileInput.files[0]);
    if (!validation.isValid) {
      statusDiv.textContent = validation.message;
      statusDiv.classList.add('error');
      return;
    }

    // Show progress bar and status
    progressBarContainer.style.display = 'block';
    progressBar.style.width = '30%';
    progressLabel.textContent = 'Uploading...';
    statusDiv.textContent = 'Uploading file...';
    statusDiv.classList.add('info');
  
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    try {
      // Simulasi progress upload
      setTimeout(() => { 
        progressBar.style.width = '60%'; 
        progressLabel.textContent = 'Processing...'; 
        statusDiv.textContent = 'Processing forecast...'; 
      }, 400);
      
      // Kirim file ke backend
      const response = await fetch('/process-forecast', {
        method: 'POST',
        body: formData
      });
      
      progressBar.style.width = '90%';
      progressLabel.textContent = 'Finalizing...';
      statusDiv.textContent = 'Finalizing...';
      
      if (!response.ok) {
        const errText = await response.text();
        statusDiv.textContent = 'Gagal memproses: ' + errText;
        statusDiv.className = '';
        statusDiv.classList.add('error');
        progressBarContainer.style.display = 'none';
        return;
      }
      
      const data = await response.json();
      if (data.status !== 'success' || !data.file_id) {
        statusDiv.textContent = 'Gagal proses: ' + (data.message || 'Unknown error');
        statusDiv.className = '';
        statusDiv.classList.add('error');
        progressBarContainer.style.display = 'none';
        return;
      }
  
      // ===== INTEGRASI DASHBOARD DENGAN DATA BACKEND =====
      if (data.original_df && data.forecast_df && data.real_time_forecast) {
        originalData = data.original_df;
        backtestData = data.forecast_df;
        realtimeData = data.real_time_forecast;
        setupMonthPickers(); // Update pickers with new data
        renderRealtimeDashboard();
        renderBacktestDashboard();
      }

      // Download file hasil otomatis
      const fileId = data.file_id;
      progressBar.style.width = '100%';
      progressLabel.textContent = 'Forecast selesai, file diunduh';
      statusDiv.textContent = 'Forecast selesai, file diunduh.';
      statusDiv.className = '';
      statusDiv.classList.add('success');
      
      const downloadUrl = `/download-forecast?file_id=${encodeURIComponent(fileId)}`;
      const blobResp = await fetch(downloadUrl);
      
      if (!blobResp.ok) {
        statusDiv.textContent = 'Gagal download file hasil: ' + (await blobResp.text());
        statusDiv.className = '';
        statusDiv.classList.add('error');
        return;
      }
  
      const blob = await blobResp.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'forecast_result.xlsx';
      document.body.appendChild(a);
      a.click();
      
      // Cleanup setelah download
      setTimeout(() => {
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
        progressBarContainer.style.display = 'none';
        progressBar.style.width = '0%';
        progressLabel.textContent = '';
      }, 1200);
  
    } catch (err) {
      console.error('Error during forecast processing:', err);
      progressBarContainer.style.display = 'none';
      statusDiv.textContent = 'Gagal proses: ' + err.message;
      statusDiv.className = '';
      statusDiv.classList.add('error');
    }
  });
} else {
  console.error('Beberapa elemen tidak ditemukan, tombol forecast tidak akan berfungsi');
}
