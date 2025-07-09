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

// Event listener untuk DOM loaded
window.addEventListener('DOMContentLoaded', () => {
  if (processBtn) processBtn.disabled = false;
  updateProcessBtnState();
  console.log('DOMContentLoaded: script.js loaded, processBtn:', processBtn);
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
