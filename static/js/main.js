// Global variables
let datasetChart = null;
let metricsChart = null;
let trainTestChart = null;
let statusInterval = null;
let retrainCheckInterval = null;

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializeCharts();
    loadStats();
    startStatusUpdates();
    
    // Check retraining status periodically
    retrainCheckInterval = setInterval(checkRetrainStatus, 1000);
});

// Tab switching
function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName + '-tab').classList.add('active');
    
    // Add active class to clicked button
    event.target.classList.add('active');
    
    // Load data for specific tabs
    if (tabName === 'visualize') {
        loadStats();
    } else if (tabName === 'status') {
        updateStatus();
    }
}

// Image upload handling
function handleImageSelect(event) {
    const file = event.target.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            document.getElementById('previewImg').src = e.target.result;
            document.getElementById('imagePreview').style.display = 'block';
            document.getElementById('uploadArea').style.display = 'none';
        };
        reader.readAsDataURL(file);
    }
}

// Predict image
async function predictImage() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select an image first');
        return;
    }
    
    const formData = new FormData();
    formData.append('image', file);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            displayPrediction(result);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        alert('Error making prediction: ' + error.message);
    }
}

// Display prediction result
function displayPrediction(result) {
    const resultDiv = document.getElementById('predictionResult');
    const confidencePercent = (result.confidence * 100).toFixed(2);
    
    resultDiv.innerHTML = `
        <h3>Prediction Result</h3>
        <div class="confidence">
            <strong>Class:</strong> ${result.class}<br>
            <strong>Confidence:</strong> ${confidencePercent}%
        </div>
        <div style="margin-top: 15px;">
            <p><strong>Probabilities:</strong></p>
            <p>Cat: ${(result.probabilities.cat * 100).toFixed(2)}%</p>
            <p>Dog: ${(result.probabilities.dog * 100).toFixed(2)}%</p>
        </div>
    `;
    
    resultDiv.classList.add('show');
}

// Bulk upload handling
function handleBulkUpload(event) {
    const files = event.files || event.target.files;
    const category = document.getElementById('categorySelect').value;
    
    if (files.length === 0) return;
    
    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
        formData.append('files[]', files[i]);
    }
    formData.append('category', category);
    
    uploadFiles(formData);
}

// Upload files to server
async function uploadFiles(formData) {
    const statusDiv = document.getElementById('uploadStatus');
    statusDiv.textContent = 'Uploading files...';
    statusDiv.classList.add('show');
    
    try {
        const response = await fetch('/api/retrain/upload', {
            method: 'POST',
            body: formData
        });
        
        const result = await response.json();
        
        if (response.ok) {
            statusDiv.textContent = `Successfully uploaded ${result.files.length} file(s) for ${result.category}`;
            statusDiv.style.background = '#e8f5e9';
            statusDiv.style.color = '#2e7d32';
        } else {
            statusDiv.textContent = 'Error: ' + result.error;
            statusDiv.style.background = '#ffebee';
            statusDiv.style.color = '#c62828';
        }
    } catch (error) {
        statusDiv.textContent = 'Error uploading files: ' + error.message;
        statusDiv.style.background = '#ffebee';
        statusDiv.style.color = '#c62828';
    }
}

// Trigger retraining
async function triggerRetraining() {
    const btn = document.getElementById('retrainBtn');
    const progressDiv = document.getElementById('retrainProgress');
    
    btn.disabled = true;
    btn.textContent = 'Retraining...';
    progressDiv.style.display = 'block';
    
    try {
        const response = await fetch('/api/retrain/trigger', {
            method: 'POST'
        });
        
        const result = await response.json();
        
        if (!response.ok) {
            alert('Error: ' + result.error);
            btn.disabled = false;
            btn.textContent = 'Start Retraining';
            progressDiv.style.display = 'none';
        }
    } catch (error) {
        alert('Error triggering retraining: ' + error.message);
        btn.disabled = false;
        btn.textContent = 'Start Retraining';
        progressDiv.style.display = 'none';
    }
}

// Check retraining status
async function checkRetrainStatus() {
    try {
        const response = await fetch('/api/retrain/status');
        const status = await response.json();
        
        if (status.in_progress) {
            const progressFill = document.getElementById('progressFill');
            const retrainMessage = document.getElementById('retrainMessage');
            
            progressFill.style.width = status.progress + '%';
            progressFill.textContent = status.progress + '%';
            retrainMessage.textContent = status.message;
            
            const btn = document.getElementById('retrainBtn');
            btn.disabled = true;
            btn.textContent = 'Retraining...';
        } else {
            const btn = document.getElementById('retrainBtn');
            const progressDiv = document.getElementById('retrainProgress');
            
            if (status.message && status.message.includes('completed')) {
                btn.disabled = false;
                btn.textContent = 'Start Retraining';
                setTimeout(() => {
                    progressDiv.style.display = 'none';
                    loadStats();
                    updateStatus();
                }, 2000);
            } else if (status.message && status.message.includes('Error')) {
                btn.disabled = false;
                btn.textContent = 'Start Retraining';
            }
        }
    } catch (error) {
        console.error('Error checking retrain status:', error);
    }
}

// Load statistics and update charts
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const stats = await response.json();
        
        updateDatasetChart(stats.dataset);
        updateMetricsChart(stats.model_metrics);
        updateTrainTestChart(stats.dataset);
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

// Initialize charts
function initializeCharts() {
    // Dataset distribution chart
    const datasetCtx = document.getElementById('datasetChart').getContext('2d');
    datasetChart = new Chart(datasetCtx, {
        type: 'doughnut',
        data: {
            labels: ['Cats', 'Dogs'],
            datasets: [{
                data: [0, 0],
                backgroundColor: ['#667eea', '#764ba2']
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true
        }
    });
    
    // Metrics chart
    const metricsCtx = document.getElementById('metricsChart').getContext('2d');
    metricsChart = new Chart(metricsCtx, {
        type: 'bar',
        data: {
            labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
            datasets: [{
                label: 'Score',
                data: [0, 0, 0, 0],
                backgroundColor: '#667eea'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1
                }
            }
        }
    });
    
    // Train vs Test chart
    const trainTestCtx = document.getElementById('trainTestChart').getContext('2d');
    trainTestChart = new Chart(trainTestCtx, {
        type: 'bar',
        data: {
            labels: ['Training', 'Test'],
            datasets: [{
                label: 'Cats',
                data: [0, 0],
                backgroundColor: '#667eea'
            }, {
                label: 'Dogs',
                data: [0, 0],
                backgroundColor: '#764ba2'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true
        }
    });
}

// Update dataset chart
function updateDatasetChart(dataset) {
    const totalCats = dataset.train_cats + dataset.test_cats + dataset.retrain_cats;
    const totalDogs = dataset.train_dogs + dataset.test_dogs + dataset.retrain_dogs;
    
    datasetChart.data.datasets[0].data = [totalCats, totalDogs];
    datasetChart.update();
}

// Update metrics chart
function updateMetricsChart(metrics) {
    if (metrics && Object.keys(metrics).length > 0) {
        metricsChart.data.datasets[0].data = [
            metrics.accuracy || 0,
            metrics.precision || 0,
            metrics.recall || 0,
            metrics.f1_score || 0
        ];
        metricsChart.update();
    }
}

// Update train vs test chart
function updateTrainTestChart(dataset) {
    trainTestChart.data.datasets[0].data = [
        dataset.train_cats + dataset.retrain_cats,
        dataset.test_cats
    ];
    trainTestChart.data.datasets[1].data = [
        dataset.train_dogs + dataset.retrain_dogs,
        dataset.test_dogs
    ];
    trainTestChart.update();
}

// Update status
async function updateStatus() {
    try {
        const response = await fetch('/api/model/status');
        const status = await response.json();
        
        document.getElementById('modelStatus').textContent = status.status.toUpperCase();
        document.getElementById('uptime').textContent = formatUptime(status.uptime_seconds);
        document.getElementById('totalPredictions').textContent = status.total_predictions;
        document.getElementById('lastTraining').textContent = 
            status.last_training ? new Date(status.last_training).toLocaleString() : 'Never';
        
        if (status.metadata && status.metadata.metrics) {
            const metrics = status.metadata.metrics;
            document.getElementById('metricAccuracy').textContent = 
                metrics.accuracy ? (metrics.accuracy * 100).toFixed(2) + '%' : '-';
            document.getElementById('metricPrecision').textContent = 
                metrics.precision ? (metrics.precision * 100).toFixed(2) + '%' : '-';
            document.getElementById('metricRecall').textContent = 
                metrics.recall ? (metrics.recall * 100).toFixed(2) + '%' : '-';
            document.getElementById('metricF1').textContent = 
                metrics.f1_score ? (metrics.f1_score * 100).toFixed(2) + '%' : '-';
            document.getElementById('metricLoss').textContent = 
                metrics.test_loss ? metrics.test_loss.toFixed(4) : '-';
        }
    } catch (error) {
        console.error('Error updating status:', error);
    }
}

// Format uptime
function formatUptime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    return `${hours}h ${minutes}m ${secs}s`;
}

// Start status updates
function startStatusUpdates() {
    updateStatus();
    statusInterval = setInterval(updateStatus, 5000); // Update every 5 seconds
}

