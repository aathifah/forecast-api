document.getElementById('upload-form').addEventListener('submit', async function(e) {
  e.preventDefault();
  const fileInput = document.getElementById('file-input');
  const loading = document.getElementById('loading');
  const resultDiv = document.getElementById('result');
  const chartCanvas = document.getElementById('forecastChart');
  const progressBarContainer = document.getElementById('progress-bar-container');
  const progressBar = document.getElementById('progress-bar');
  const progressLabel = document.getElementById('progress-label');
  resultDiv.innerHTML = '';
  chartCanvas.style.display = 'none';

  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  // Show progress bar
  progressBarContainer.style.display = 'block';
  progressBar.style.width = '40%';
  progressLabel.textContent = 'Uploading...';

  try {
    // Simulate progress (upload + processing)
    setTimeout(() => { progressBar.style.width = '70%'; progressLabel.textContent = 'Processing...'; }, 500);

    const response = await fetch('/forecast', {
      method: 'POST',
      body: formData
    });
    progressBar.style.width = '90%';
    progressLabel.textContent = 'Finalizing...';
    const data = await response.json();
    progressBar.style.width = '100%';
    progressLabel.textContent = 'Done!';
    setTimeout(() => { progressBarContainer.style.display = 'none'; }, 800);

    if (data.status !== 'success') {
      resultDiv.innerHTML = `<div class="error">${data.message}</div>`;
      return;
    }

    // Tampilkan summary
    resultDiv.innerHTML = `
      <h2>Summary</h2>
      <ul>
        <li><b>Total Forecasts:</b> ${data.data.total_forecasts}</li>
        <li><b>Total Parts:</b> ${data.data.total_parts}</li>
        <li><b>Forecast Months:</b> ${data.data.forecast_months.join(', ')}</li>
      </ul>
    `;

    // Ambil data forecast untuk chart
    const forecastResults = data.data.forecast_results || [];
    if (forecastResults.length === 0) {
      resultDiv.innerHTML += '<div>No forecast data found.</div>';
      return;
    }

    // Pilih PART_NO pertama untuk contoh chart
    const partNo = forecastResults[0].PART_NO;
    const partData = forecastResults.filter(r => r.PART_NO === partNo);
    const labels = partData.map(r => r.MONTH);
    const forecastNeutral = partData.map(r => r.FORECAST_NEUTRAL);
    const forecastOptimist = partData.map(r => r.FORECAST_OPTIMIST);
    const forecastPessimist = partData.map(r => r.FORECAST_PESSIMIST);

    chartCanvas.style.display = 'block';
    if (window.myForecastChart) window.myForecastChart.destroy();
    window.myForecastChart = new Chart(chartCanvas, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [
          {
            label: `Forecast Neutral (${partNo})`,
            data: forecastNeutral,
            backgroundColor: 'rgba(54, 162, 235, 0.6)'
          },
          {
            label: `Forecast Optimist`,
            data: forecastOptimist,
            backgroundColor: 'rgba(75, 192, 192, 0.4)'
          },
          {
            label: `Forecast Pessimist`,
            data: forecastPessimist,
            backgroundColor: 'rgba(255, 99, 132, 0.3)'
          }
        ]
      },
      options: {
        responsive: true,
        plugins: {
          legend: { position: 'top' },
          title: {
            display: true,
            text: `Forecast for PART_NO: ${partNo}`
          }
        }
      }
    });

    // Tampilkan tabel hasil
    let tableHtml = `<h3>Forecast Table for ${partNo}</h3><table class="forecast-table"><thead><tr><th>Month</th><th>Optimist</th><th>Neutral</th><th>Pessimist</th><th>Best Model</th></tr></thead><tbody>`;
    partData.forEach(row => {
      tableHtml += `<tr><td>${row.MONTH}</td><td>${row.FORECAST_OPTIMIST}</td><td>${row.FORECAST_NEUTRAL}</td><td>${row.FORECAST_PESSIMIST}</td><td>${row.BEST_MODEL}</td></tr>`;
    });
    tableHtml += '</tbody></table>';
    resultDiv.innerHTML += tableHtml;

  } catch (err) {
    progressBarContainer.style.display = 'none';
    resultDiv.innerHTML = `<div class="error">Error: ${err.message}</div>`;
  }
});

const processBtn = document.getElementById('process-btn');
const downloadBtn = document.getElementById('download-btn');
const fileInput = document.getElementById('download-file-input');
const statusDiv = document.getElementById('download-status');
const progressBarContainer = document.getElementById('download-progress-bar-container');
const progressBar = document.getElementById('download-progress-bar');
const progressLabel = document.getElementById('download-progress-label');

let fileId = null;

if (processBtn) {
  processBtn.addEventListener('click', async function() {
    statusDiv.textContent = '';
    downloadBtn.disabled = true;
    fileId = null;
    if (!fileInput.files.length) {
      statusDiv.textContent = 'Pilih file Excel terlebih dahulu.';
      return;
    }
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    // Show progress bar
    progressBarContainer.style.display = 'block';
    progressBar.style.width = '30%';
    progressLabel.textContent = 'Uploading...';
    try {
      setTimeout(() => { progressBar.style.width = '60%'; progressLabel.textContent = 'Processing...'; }, 400);
      const response = await fetch('/process-forecast', {
        method: 'POST',
        body: formData
      });
      progressBar.style.width = '90%';
      progressLabel.textContent = 'Finalizing...';
      if (!response.ok) {
        const errText = await response.text();
        statusDiv.textContent = 'Gagal memproses: ' + errText;
        progressBarContainer.style.display = 'none';
        return;
      }
      const data = await response.json();
      if (data.status !== 'success' || !data.file_id) {
        statusDiv.textContent = 'Gagal proses: ' + (data.message || 'Unknown error');
        progressBarContainer.style.display = 'none';
        return;
      }
      fileId = data.file_id;
      progressBar.style.width = '100%';
      progressLabel.textContent = 'Forecast Selesai';
      statusDiv.textContent = 'Forecast Selesai. Silakan download file hasil.';
      downloadBtn.disabled = false;
      setTimeout(() => {
        progressBarContainer.style.display = 'none';
        progressBar.style.width = '0%';
        progressLabel.textContent = '';
      }, 1200);
    } catch (err) {
      progressBarContainer.style.display = 'none';
      statusDiv.textContent = 'Gagal proses: ' + err.message;
    }
  });
}

if (downloadBtn) {
  downloadBtn.addEventListener('click', async function() {
    if (!fileId) {
      statusDiv.textContent = 'File hasil belum tersedia.';
      return;
    }
    statusDiv.textContent = 'Mengunduh file hasil...';
    try {
      const response = await fetch(`/download-forecast?file_id=${encodeURIComponent(fileId)}`);
      if (!response.ok) {
        const errText = await response.text();
        statusDiv.textContent = 'Gagal download: ' + errText;
        return;
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'forecast_result.xlsx';
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
      }, 100);
      statusDiv.textContent = 'Download berhasil!';
    } catch (err) {
      statusDiv.textContent = 'Gagal download: ' + err.message;
    }
  });
}
