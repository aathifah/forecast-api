document.getElementById('upload-form').addEventListener('submit', async function(e) {
  e.preventDefault();
  const fileInput = document.getElementById('file-input');
  const loading = document.getElementById('loading');
  const resultDiv = document.getElementById('result');
  const chartCanvas = document.getElementById('forecastChart');
  resultDiv.innerHTML = '';
  chartCanvas.style.display = 'none';

  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  loading.style.display = 'block';

  try {
    const response = await fetch('/forecast', {
      method: 'POST',
      body: formData
    });
    const data = await response.json();
    loading.style.display = 'none';

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
    loading.style.display = 'none';
    resultDiv.innerHTML = `<div class="error">Error: ${err.message}</div>`;
  }
});
