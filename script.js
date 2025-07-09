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

console.log('script.js loaded');
const processBtn = document.getElementById('process-btn');
const fileInput = document.getElementById('download-file-input');
const statusDiv = document.getElementById('download-status');
const progressBarContainer = document.getElementById('download-progress-bar-container');
const progressBar = document.getElementById('download-progress-bar');
const progressLabel = document.getElementById('download-progress-label');

if (!processBtn) {
  alert('Tombol proses tidak ditemukan di HTML!');
  console.error('processBtn is null');
}
if (!fileInput) {
  alert('Input file tidak ditemukan di HTML!');
  console.error('fileInput is null');
}
if (!statusDiv || !progressBarContainer || !progressBar || !progressLabel) {
  alert('Ada elemen status/progress yang tidak ditemukan di HTML!');
  console.error('status/progress bar element is null');
}

function updateProcessBtnState() {
  if (processBtn && fileInput) {
    processBtn.disabled = !fileInput.files.length;
  }
}

if (fileInput) fileInput.addEventListener('change', updateProcessBtnState);
window.addEventListener('DOMContentLoaded', () => {
  if (processBtn) processBtn.disabled = false;
  updateProcessBtnState();
  console.log('DOMContentLoaded: script.js loaded, processBtn:', processBtn);
});

if (processBtn && fileInput && statusDiv && progressBarContainer && progressBar && progressLabel) {
  processBtn.disabled = false;
  processBtn.addEventListener('click', async function() {
    statusDiv.textContent = '';
    if (!fileInput.files.length) {
      statusDiv.textContent = 'Pilih file Excel terlebih dahulu.';
      return;
    }
    // Show progress bar and status
    progressBarContainer.style.display = 'block';
    progressBar.style.width = '30%';
    progressLabel.textContent = 'Uploading...';
    statusDiv.textContent = 'Uploading file...';
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    try {
      setTimeout(() => { progressBar.style.width = '60%'; progressLabel.textContent = 'Processing...'; statusDiv.textContent = 'Processing forecast...'; }, 400);
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
        progressBarContainer.style.display = 'none';
        return;
      }
      const data = await response.json();
      if (data.status !== 'success' || !data.file_id) {
        statusDiv.textContent = 'Gagal proses: ' + (data.message || 'Unknown error');
        progressBarContainer.style.display = 'none';
        return;
      }
      // Download file hasil otomatis
      const fileId = data.file_id;
      progressBar.style.width = '100%';
      progressLabel.textContent = 'Forecast selesai, file diunduh';
      statusDiv.textContent = 'Forecast selesai, file diunduh.';
      const downloadUrl = `/download-forecast?file_id=${encodeURIComponent(fileId)}`;
      const blobResp = await fetch(downloadUrl);
      if (!blobResp.ok) {
        statusDiv.textContent = 'Gagal download file hasil: ' + (await blobResp.text());
        return;
      }
      const blob = await blobResp.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'forecast_result.xlsx';
      document.body.appendChild(a);
      a.click();
      setTimeout(() => {
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
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
