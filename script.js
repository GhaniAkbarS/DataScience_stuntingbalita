// Global variables
let genderChart, ageChart;

// Initialize dashboard
document.addEventListener('DOMContentLoaded', function() {
    loadGenderChart();
    loadAgeChart();
    checkAPIStatus();
});

// Check if Flask API is available
async function checkAPIStatus() {
    try {
        const response = await fetch('/health');
        if (response.ok) {
            const data = await response.json();
            console.log('✅ Flask API Status:', data);
            
            // Show API status in UI
            if (data.model && data.model.includes('Ready')) {
                showAPIStatus('🤖 AI Model Ready - Prediksi menggunakan Machine Learning');
            } else {
                showAPIStatus('📊 WHO Standards Ready - Prediksi menggunakan Z-score WHO');
            }
        }
    } catch (error) {
        console.log('⚠️ Flask API not available, using WHO Z-score only');
        showAPIStatus('📊 Offline Mode - Prediksi menggunakan Z-score WHO');
    }
}

// Show API status in UI
function showAPIStatus(message) {
    const statusElement = document.getElementById('statusText');
    if (statusElement) {
        statusElement.textContent = message;
    }
    console.log(message);
}

// Navigation functions - UPDATED to support analysis section
function showSection(sectionId) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Show selected section
    document.getElementById(sectionId).classList.add('active');
    
    // Update active nav link
    document.querySelectorAll('.nav-links a').forEach(link => {
        link.classList.remove('active');
    });
    
    // Find and activate the correct nav link
    const navLinks = document.querySelectorAll('.nav-links a');
    navLinks.forEach(link => {
        if (link.textContent.toLowerCase().includes(sectionId.toLowerCase()) || 
            (sectionId === 'dashboard' && link.textContent === 'Dashboard') ||
            (sectionId === 'analysis' && link.textContent === 'Analisis Data') ||
            (sectionId === 'prediction' && link.textContent === 'Prediksi Stunting') ||
            (sectionId === 'about' && link.textContent === 'Tentang')) {
            link.classList.add('active');
        }
    });
    
    // Reset analysis section when navigating away
    if (sectionId !== 'analysis') {
        resetAnalysisSection();
    }
}

// Reset analysis section to welcome state
function resetAnalysisSection() {
    const analysisWelcome = document.getElementById('analysisWelcome');
    const analysisChart = document.getElementById('analysisChart');
    const analysisLoading = document.getElementById('analysisLoading');
    
    if (analysisWelcome) analysisWelcome.style.display = 'block';
    if (analysisChart) analysisChart.style.display = 'none';
    if (analysisLoading) analysisLoading.style.display = 'none';
}

// Load gender chart with sample data
function loadGenderChart() {
    const ctx = document.getElementById('genderChart').getContext('2d');
    genderChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Laki-laki', 'Perempuan'],
            datasets: [{
                label: 'Normal',
                data: [38200, 37400],
                backgroundColor: '#28a745',
                borderColor: '#28a745',
                borderWidth: 1
            }, {
                label: 'Stunted',
                data: [12800, 11700],
                backgroundColor: '#ffc107',
                borderColor: '#ffc107',
                borderWidth: 1
            }, {
                label: 'Severely Stunted',
                data: [7500, 7400],
                backgroundColor: '#dc3545',
                borderColor: '#dc3545',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// Load age chart with sample data
function loadAgeChart() {
    const ctx = document.getElementById('ageChart').getContext('2d');
    ageChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['0-6 bulan', '7-12 bulan', '13-24 bulan', '25-36 bulan', '37-48 bulan', '49-60 bulan'],
            datasets: [{
                label: 'Normal',
                data: [15200, 14800, 13500, 12100, 10800, 9200],
                backgroundColor: '#28a745',
                borderColor: '#28a745',
                borderWidth: 1
            }, {
                label: 'Stunted',
                data: [2800, 3200, 4100, 4800, 5200, 4400],
                backgroundColor: '#ffc107',
                borderColor: '#ffc107',
                borderWidth: 1
            }, {
                label: 'Severely Stunted',
                data: [1500, 1800, 2400, 2900, 3200, 2600],
                backgroundColor: '#dc3545',
                borderColor: '#dc3545',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
}

// WHO Z-score standards (simplified) - fallback jika API tidak tersedia
const whoStandards = {
    male: {
        0: {median: 49.9, sd: 1.9},
        6: {median: 67.6, sd: 2.6},
        12: {median: 75.7, sd: 2.9},
        18: {median: 82.3, sd: 3.2},
        24: {median: 87.1, sd: 3.4},
        36: {median: 96.1, sd: 3.7},
        48: {median: 103.3, sd: 4.0},
        60: {median: 109.9, sd: 4.2}
    },
    female: {
        0: {median: 49.1, sd: 1.9},
        6: {median: 65.7, sd: 2.4},
        12: {median: 74.0, sd: 2.8},
        18: {median: 80.7, sd: 3.1},
        24: {median: 85.7, sd: 3.3},
        36: {median: 94.2, sd: 3.6},
        48: {median: 101.6, sd: 3.9},
        60: {median: 108.4, sd: 4.1}
    }
};

// Main prediction function - tries API first, then fallback to WHO Z-score
async function predictStunting() {
    const age = parseInt(document.getElementById('age').value);
    const gender = document.getElementById('gender').value;
    const height = parseFloat(document.getElementById('height').value);
    
    if (!age || !gender || !height) {
        alert('Mohon lengkapi semua data!');
        return;
    }

    // Validate input
    if (age < 0 || age > 60) {
        alert('Usia harus antara 0-60 bulan');
        return;
    }
    
    if (height < 40 || height > 120) {
        alert('Tinggi badan harus antara 40-120 cm');
        return;
    }

    // Show loading state
    showLoadingState(true);
    
    // Try API prediction first
    try {
        const result = await predictWithAPI(age, gender, height);
        if (result.success) {
            showPredictionResult(
                result.result, 
                null, // API doesn't return Z-score
                getClassFromStatus(result.result), 
                getRecommendationsFromStatus(result.result),
                `AI Model (Confidence: ${result.confidence}%)`
            );
            return;
        }
    } catch (error) {
        console.log('API prediction failed, using WHO Z-score:', error);
    }
    
    // Fallback to WHO Z-score calculation
    const whoResult = predictWithWHO(age, gender, height);
    showPredictionResult(
        whoResult.status,
        whoResult.zScore,
        whoResult.className,
        whoResult.recommendations,
        'WHO Z-score Standards'
    );
    
    showLoadingState(false);
}

// Predict using Flask API
async function predictWithAPI(age, gender, height) {
    const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            age: age,
            height: height,
            gender: gender
        })
    });
    
    const data = await response.json();
    return data;
}

// Predict using WHO Z-score (fallback)
function predictWithWHO(age, gender, height) {
    // Find closest age reference
    const ageRef = findClosestAge(age);
    const standard = whoStandards[gender][ageRef];
    
    // Calculate Z-score
    const zScore = (height - standard.median) / standard.sd;
    
    let status, className, recommendations;
    
    if (zScore >= -2) {
        status = 'Normal';
        className = 'success';
        recommendations = [
            'Pertahankan pola makan sehat dan seimbang',
            'Lanjutkan pemberian ASI eksklusif (jika masih menyusui)',
            'Berikan MPASI yang bergizi (untuk usia >6 bulan)',
            'Rutin kontrol ke posyandu untuk monitoring pertumbuhan',
            'Pastikan kebersihan dan sanitasi lingkungan'
        ];
    } else if (zScore >= -3) {
        status = 'Stunted (Pendek)';
        className = 'warning';
        recommendations = [
            'Konsultasi dengan tenaga kesehatan untuk evaluasi mendalam',
            'Perbaiki pola makan dengan makanan bergizi tinggi',
            'Berikan suplemen sesuai anjuran dokter',
            'Tingkatkan frekuensi monitoring pertumbuhan',
            'Evaluasi faktor risiko dalam keluarga'
        ];
    } else {
        status = 'Severely Stunted (Sangat Pendek)';
        className = 'danger';
        recommendations = [
            'SEGERA konsultasi dengan dokter spesialis anak',
            'Diperlukan intervensi gizi intensif',
            'Evaluasi komprehensif untuk mencari penyebab',
            'Monitoring ketat pertumbuhan dan perkembangan',
            'Dukungan keluarga dan lingkungan yang optimal'
        ];
    }
    
    return { status, zScore, className, recommendations };
}

// Helper functions
function getClassFromStatus(status) {
    if (status === 'Normal' || status === 'Gizi Baik') return 'success';
    if (status === 'Stunted' || status === 'Stunting') return 'warning';
    return 'danger';
}

function getRecommendationsFromStatus(status) {
    if (status === 'Normal' || status === 'Gizi Baik') {
        return [
            'Pertahankan pola makan sehat dan seimbang',
            'Lanjutkan pemberian ASI eksklusif (jika masih menyusui)',
            'Berikan MPASI yang bergizi (untuk usia >6 bulan)',
            'Rutin kontrol ke posyandu untuk monitoring pertumbuhan',
            'Pastikan kebersihan dan sanitasi lingkungan'
        ];
    } else if (status === 'Stunted' || status === 'Stunting') {
        return [
            'Konsultasi dengan tenaga kesehatan untuk evaluasi mendalam',
            'Perbaiki pola makan dengan makanan bergizi tinggi',
            'Berikan suplemen sesuai anjuran dokter',
            'Tingkatkan frekuensi monitoring pertumbuhan',
            'Evaluasi faktor risiko dalam keluarga'
        ];
    } else {
        return [
            'SEGERA konsultasi dengan dokter spesialis anak',
            'Diperlukan intervensi gizi intensif',
            'Evaluasi komprehensif untuk mencari penyebab',
            'Monitoring ketat pertumbuhan dan perkembangan',
            'Dukungan keluarga dan lingkungan yang optimal'
        ];
    }
}

function findClosestAge(age) {
    const ages = Object.keys(whoStandards.male).map(Number);
    return ages.reduce((prev, curr) => 
        Math.abs(curr - age) < Math.abs(prev - age) ? curr : prev
    );
}

function showLoadingState(isLoading) {
    const btn = document.querySelector('.predict-btn');
    if (btn) {
        if (isLoading) {
            btn.textContent = '⏳ Memproses Prediksi...';
            btn.disabled = true;
            btn.style.opacity = '0.7';
        } else {
            btn.textContent = '🔮 Prediksi Status Stunting';
            btn.disabled = false;
            btn.style.opacity = '1';
        }
    }
}

function showPredictionResult(status, zScore, className, recommendations, method) {
    const resultDiv = document.getElementById('predictionResult');
    const titleElement = document.getElementById('resultTitle');
    const descriptionElement = document.getElementById('resultDescription');
    const recommendationsList = document.getElementById('recommendationList');
    
    if (!resultDiv || !titleElement || !descriptionElement || !recommendationsList) {
        console.error('Prediction result elements not found');
        return;
    }
    
    titleElement.textContent = `Status Prediksi: ${status}`;
    
    // Show method and Z-score if available
    let description = `Metode: ${method}`;
    if (zScore !== null) {
        description += ` | Z-score: ${zScore.toFixed(2)}`;
    }
    descriptionElement.textContent = description;
    
    // Clear previous recommendations
    recommendationsList.innerHTML = '';
    
    // Add new recommendations
    recommendations.forEach(rec => {
        const li = document.createElement('li');
        li.textContent = rec;
        recommendationsList.appendChild(li);
    });
    
    // Set class and show result
    resultDiv.className = `prediction-result ${className}`;
    resultDiv.style.display = 'block';
    
    // Scroll to result
    resultDiv.scrollIntoView({ behavior: 'smooth' });
    
    showLoadingState(false);
}

// Smooth scrolling for navigation
document.querySelectorAll('.nav-links a').forEach(link => {
    link.addEventListener('click', function(e) {
        e.preventDefault();
        // Get the section to show from onclick attribute or determine from text
        const text = this.textContent.toLowerCase();
        let sectionId = 'dashboard';
        if (text.includes('analisis')) sectionId = 'analysis';
        else if (text.includes('prediksi')) sectionId = 'prediction';
        else if (text.includes('tentang')) sectionId = 'about';
        
        showSection(sectionId);
    });
});

// Additional utility functions
function updateStatistics(data) {
    // Function to update dashboard statistics
    const elements = {
        'totalChildren': data.total || '121,000',
        'stuntingPrevalence': data.prevalence || '24.4%',
        'normalCount': data.normal || '75,600',
        'stuntedCount': data.stunted || '29,500'
    };
    
    Object.entries(elements).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
        }
    });
}

function exportData() {
    // Function to export dashboard data
    const data = {
        timestamp: new Date().toISOString(),
        statistics: {
            total: document.getElementById('totalChildren')?.textContent || '',
            prevalence: document.getElementById('stuntingPrevalence')?.textContent || '',
            normal: document.getElementById('normalCount')?.textContent || '',
            stunted: document.getElementById('stuntedCount')?.textContent || ''
        }
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `stunting-dashboard-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Initialize tooltips or additional features
function initializeTooltips() {
    // Add tooltips to stat cards
    document.querySelectorAll('.stat-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            // Add hover effects or show additional info
            this.style.transform = 'translateY(-5px)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
}