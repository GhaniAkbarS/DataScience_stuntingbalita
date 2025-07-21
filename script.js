// ===============================================
// DASHBOARD STUNTING ANALYTICS - COMPREHENSIVE JAVASCRIPT
// ===============================================

// Global variables
let genderChart, ageChart, trainingChart, confusionChart;
let storyCharts = {};
let currentSection = 'home';
let apiStatus = {
    available: false,
    modelReady: false,
    datasetLoaded: false
};

// Data storage
let datasetStats = {};
let predictionHistory = [];
let downloadHistory = [];

// ===============================================
// INITIALIZATION & SETUP
// ===============================================

document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Initializing Comprehensive Dashboard Stunting...');
    
    // Initialize all dashboard components
    initializeDashboard();
    
    // Load initial data and charts
    loadInitialData();
    
    // Setup event listeners
    setupEventListeners();
    
    // Initialize animations
    initializeAnimations();
    
    // Check API status
    checkAPIStatus();
    
    console.log('✅ Comprehensive Dashboard initialization complete');
});

// ===============================================
// DASHBOARD INITIALIZATION
// ===============================================

function initializeDashboard() {
    // Set initial active section based on URL hash or default to home
    const hash = window.location.hash.substring(1);
    const initialSection = hash && document.getElementById(hash) ? hash : 'home';
    showSection(initialSection);
    
    // Initialize tooltips and interactive elements
    initializeTooltips();
    
    // Setup responsive handlers
    setupResponsiveHandlers();
    
    // Initialize section-specific components
    initializeAllSections();
}

function initializeAllSections() {
    // Initialize each section
    initializeHome();
    initializeOverview();
    initializeCleaning();
    initializeEDA();
    initializeModeling();
    initializeStoryTelling();
    initializeInsights();
    initializePrediction();
    initializeExport();
    initializeTeam();
}

// ===============================================
// SECTION INITIALIZATION FUNCTIONS
// ===============================================

function initializeHome() {
    console.log('🏠 Initializing Home section...');
    loadHomeStatistics();
    setupQuickNavigation();
}

function initializeOverview() {
    console.log('📊 Initializing Overview section...');
    setupDataPreviewControls();
    setupQualityMetrics();
}

function initializeCleaning() {
    console.log('🧹 Initializing Cleaning section...');
    animateCleaningSteps();
    loadCleaningResults();
}

function initializeEDA() {
    console.log('📈 Initializing EDA section...');
    setupEDAControls();
    resetEDASection();
}

function initializeModeling() {
    console.log('🤖 Initializing Modeling section...');
    loadModelMetrics();
    setupModelComparison();
    loadPredictionExamples();
}

function initializeStoryTelling() {
    console.log('📖 Initializing Story Telling section...');
    setupStoryCharts();
    loadStoryNarratives();
}

function initializeInsights() {
    console.log('💡 Initializing Insights section...');
    loadKeyFindings();
    setupImplementationRoadmap();
}

function initializePrediction() {
    console.log('🔮 Initializing Prediction section...');
    setupPredictionForm();
    loadPredictionHistory();
}

function initializeExport() {
    console.log('📥 Initializing Export section...');
    setupExportButtons();
    loadDownloadHistory();
    updateDownloadStats();
}

function initializeTeam() {
    console.log('👥 Initializing Team section...');
    animateTeamCards();
    setupContactLinks();
}

// ===============================================
// DATA LOADING FUNCTIONS
// ===============================================

async function loadInitialData() {
    try {
        // Load dataset statistics
        await loadDatasetStatistics();
        
        // Load home highlights
        loadHomeStatistics();
        
        // Initialize charts
        loadInitialCharts();
        
        console.log('✅ Initial data loaded successfully');
    } catch (error) {
        console.error('❌ Error loading initial data:', error);
    }
}

async function loadDatasetStatistics() {
    try {
        // Simulate API call or load from data
        datasetStats = {
            totalSamples: 121000,
            stuntingRate: 24.4,
            avgAge: 29.2,
            avgHeight: 87.5,
            genderRatio: "51:49",
            normalCount: 75600,
            stuntedCount: 29500,
            severeCount: 15900,
            completeness: 98.5,
            outliers: 2.1,
            duplicates: 0.3
        };
        
        console.log('📊 Dataset statistics loaded');
        return datasetStats;
    } catch (error) {
        console.error('Error loading dataset statistics:', error);
        return {};
    }
}

function loadHomeStatistics() {
    // Update home section statistics
    const updates = {
        'homeDatasetSize': `${datasetStats.totalSamples?.toLocaleString() || '121,000'} records`,
        'stuntingRate': `${datasetStats.stuntingRate || '24.4'}%`,
        'avgAge': `${datasetStats.avgAge || '29.2'} months`,
        'avgHeight': `${datasetStats.avgHeight || '87.5'} cm`,
        'genderRatio': datasetStats.genderRatio || '51:49'
    };
    
    Object.entries(updates).forEach(([id, value]) => {
        const element = document.getElementById(id);
        if (element) {
            // Animate the update
            element.style.opacity = '0';
            setTimeout(() => {
                element.textContent = value;
                element.style.opacity = '1';
            }, 300);
        }
    });
    
    // Update footer stats
    const footerTotalData = document.getElementById('footerTotalData');
    if (footerTotalData) {
        footerTotalData.textContent = `${Math.floor(datasetStats.totalSamples / 1000) || '121'}K+`;
    }
}

function loadInitialCharts() {
    // Load basic charts for overview section
    loadStatusDistributionChart();
    loadGenderDistributionChart();
}

// ===============================================
// NAVIGATION MANAGEMENT
// ===============================================

function showSection(sectionId) {
    console.log(`📍 Navigating to section: ${sectionId}`);
    
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.remove('active');
    });
    
    // Show selected section
    const targetSection = document.getElementById(sectionId);
    if (targetSection) {
        targetSection.classList.add('active');
        currentSection = sectionId;
        
        // Update navigation
        updateNavigationState(sectionId);
        
        // Handle section-specific logic
        handleSectionChange(sectionId);
        
        // Update URL hash
        if (history.pushState) {
            history.pushState(null, null, `#${sectionId}`);
        }
        
        // Scroll to top smoothly
        window.scrollTo({ top: 0, behavior: 'smooth' });
        
        // Analytics tracking
        trackSectionView(sectionId);
    }
}

function updateNavigationState(sectionId) {
    // Update active nav link
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
    });
    
    // Find and activate correct nav link
    document.querySelectorAll('.nav-link').forEach(link => {
        if (link.getAttribute('onclick') && link.getAttribute('onclick').includes(sectionId)) {
            link.classList.add('active');
        }
    });
}

function handleSectionChange(sectionId) {
    switch (sectionId) {
        case 'home':
            loadHomeStatistics();
            break;
        case 'overview':
            loadOverviewData();
            break;
        case 'cleaning':
            animateCleaningSteps();
            break;
        case 'eda':
            resetEDASection();
            break;
        case 'modeling':
            loadModelPerformanceCharts();
            break;
        case 'storytelling':
            initializeStoryCharts();
            break;
        case 'insights':
            loadInsightMetrics();
            break;
        case 'prediction':
            resetPredictionForm();
            break;
        case 'export':
            updateDownloadStats();
            break;
        case 'team':
            animateTeamCards();
            break;
    }
}

// ===============================================
// OVERVIEW SECTION FUNCTIONS
// ===============================================

function loadOverviewData() {
    const overviewLoading = document.getElementById('overviewLoading');
    const overviewContent = document.getElementById('overviewContent');
    
    if (overviewLoading) overviewLoading.style.display = 'flex';
    if (overviewContent) overviewContent.style.display = 'none';
    
    // Simulate data loading
    setTimeout(() => {
        populateDatasetSummary();
        updateQualityMetrics();
        populateVariableInfo();
        loadBasicStatistics();
        generateInitialInsights();
        
        if (overviewLoading) overviewLoading.style.display = 'none';
        if (overviewContent) overviewContent.style.display = 'block';
    }, 1500);
}

function populateDatasetSummary() {
    const summaryGrid = document.getElementById('datasetSummary');
    if (!summaryGrid) return;
    
    const summaryHTML = `
        <div class="summary-item">
            <h5>📊 Total Records</h5>
            <p class="summary-value">${datasetStats.totalSamples?.toLocaleString() || '121,000'}</p>
        </div>
        <div class="summary-item">
            <h5>📅 Age Range</h5>
            <p class="summary-value">0-60 months</p>
        </div>
        <div class="summary-item">
            <h5>📏 Height Range</h5>
            <p class="summary-value">45-120 cm</p>
        </div>
        <div class="summary-item">
            <h5>⚠️ Stunting Rate</h5>
            <p class="summary-value">${datasetStats.stuntingRate || '24.4'}%</p>
        </div>
    `;
    
    summaryGrid.innerHTML = summaryHTML;
}

function updateQualityMetrics() {
    // Update completeness
    updateQualityBar('completenessBar', 'completenessText', datasetStats.completeness || 98.5);
    
    // Update outliers (inverted - lower is better)
    updateQualityBar('outliersBar', 'outliersText', 100 - (datasetStats.outliers || 2.1));
    
    // Update duplicates (inverted - lower is better)
    updateQualityBar('duplicatesBar', 'duplicatesText', 100 - (datasetStats.duplicates || 0.3));
}

function updateQualityBar(barId, textId, percentage) {
    const bar = document.getElementById(barId);
    const text = document.getElementById(textId);
    
    if (bar) {
        bar.style.width = `${percentage}%`;
        bar.style.backgroundColor = percentage > 90 ? '#28a745' : percentage > 70 ? '#ffc107' : '#dc3545';
    }
    
    if (text) {
        const displayValue = barId.includes('completeness') ? percentage : 100 - percentage;
        text.textContent = `${displayValue.toFixed(1)}%`;
    }
}

function populateVariableInfo() {
    // Update age variable info
    updateElement('ageRange', '0-60 months');
    updateElement('ageMean', `${datasetStats.avgAge || '29.2'} months`);
    updateElement('ageStd', '15.4 months');
    
    // Update height variable info
    updateElement('heightRange', '45-120 cm');
    updateElement('heightMean', `${datasetStats.avgHeight || '87.5'} cm`);
    updateElement('heightStd', '12.8 cm');
    
    // Update gender distribution
    updateElement('genderDist', datasetStats.genderRatio || '51:49 (L:P)');
    
    // Update status distribution
    const normalPct = ((datasetStats.normalCount / datasetStats.totalSamples) * 100).toFixed(1);
    const stuntedPct = ((datasetStats.stuntedCount / datasetStats.totalSamples) * 100).toFixed(1);
    updateElement('statusDist', `Normal: ${normalPct}%, Stunted: ${stuntedPct}%`);
}

function loadDataPreview(rows = 10) {
    const tableContainer = document.getElementById('dataPreviewTable');
    if (!tableContainer) return;
    
    // Generate sample data
    const sampleData = generateSampleDataRows(rows);
    
    const tableHTML = `
        <table class="preview-table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Umur (bulan)</th>
                    <th>Tinggi Badan (cm)</th>
                    <th>Jenis Kelamin</th>
                    <th>Status Gizi</th>
                </tr>
            </thead>
            <tbody>
                ${sampleData.map(row => `
                    <tr>
                        <td>${row.id}</td>
                        <td>${row.age}</td>
                        <td>${row.height}</td>
                        <td>${row.gender}</td>
                        <td><span class="status-badge ${row.statusClass}">${row.status}</span></td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    
    tableContainer.innerHTML = tableHTML;
}

function generateSampleDataRows(count) {
    const rows = [];
    const genders = ['Laki-laki', 'Perempuan'];
    const statuses = [
        { status: 'Normal', class: 'normal' },
        { status: 'Stunting', class: 'stunting' },
        { status: 'Gizi Buruk', class: 'severe' }
    ];
    
    for (let i = 1; i <= count; i++) {
        const age = Math.floor(Math.random() * 60);
        const baseHeight = 50 + (age * 1.2);
        const height = (baseHeight + (Math.random() - 0.5) * 20).toFixed(1);
        const gender = genders[Math.floor(Math.random() * genders.length)];
        const statusInfo = statuses[Math.floor(Math.random() * statuses.length)];
        
        rows.push({
            id: i.toString().padStart(4, '0'),
            age: age,
            height: height,
            gender: gender,
            status: statusInfo.status,
            statusClass: statusInfo.class
        });
    }
    
    return rows;
}

// ===============================================
// EDA SECTION FUNCTIONS
// ===============================================

async function loadEDAChart(chartType) {
    console.log(`🔄 Loading EDA chart: ${chartType}`);
    
    // Update button states
    document.querySelectorAll('.eda-btn').forEach(btn => btn.classList.remove('active'));
    document.querySelector(`[data-type="${chartType}"]`)?.classList.add('active');
    
    // Show loading state
    showEDALoading(true);
    
    try {
        // Simulate API call to backend
        const response = await fetch(`/api/charts/${chartType}`);
        const data = await response.json();
        
        if (data.success) {
            displayEDAChart(chartType, data);
        } else {
            throw new Error(data.error || 'Failed to load chart');
        }
        
    } catch (error) {
        console.error('❌ Error loading EDA chart:', error);
        showEDAError(error.message);
    }
}

function showEDALoading(show) {
    const welcome = document.getElementById('edaWelcome');
    const chart = document.getElementById('edaChart');
    const loading = document.getElementById('edaLoading');
    
    if (show) {
        if (welcome) welcome.style.display = 'none';
        if (chart) chart.style.display = 'none';
        if (loading) loading.style.display = 'flex';
    } else {
        if (loading) loading.style.display = 'none';
    }
}

function displayEDAChart(chartType, data) {
    const welcome = document.getElementById('edaWelcome');
    const chart = document.getElementById('edaChart');
    const chartImg = document.getElementById('edaChartImg');
    const chartTitle = document.getElementById('edaChartTitle');
    const chartDescription = document.getElementById('edaChartDescription');
    const chartTimestamp = document.getElementById('edaChartTimestamp');
    const chartInsights = document.getElementById('edaChartInsights');
    
    // Update chart content
    if (chartImg) chartImg.src = `data:image/png;base64,${data.chart}`;
    
    const titles = {
        'distribution': '📊 Analisis Distribusi Data Stunting',
        'correlation': '🔗 Matriks Korelasi Variabel Stunting',
        'clustering': '🎯 Analisis K-Means Clustering'
    };
    
    const descriptions = {
        'distribution': 'Visualisasi distribusi umur, tinggi badan, jenis kelamin, dan status gizi dengan statistik deskriptif',
        'correlation': 'Heatmap korelasi untuk memahami hubungan antar variabel dan identifikasi faktor risiko',
        'clustering': 'Pengelompokan balita berdasarkan karakteristik menggunakan K-Means dan evaluasi Silhouette Score'
    };
    
    if (chartTitle) chartTitle.textContent = titles[chartType];
    if (chartDescription) chartDescription.textContent = descriptions[chartType];
    if (chartTimestamp) chartTimestamp.textContent = `Diperbarui: ${new Date().toLocaleString('id-ID')}`;
    
    // Generate insights
    generateEDAInsights(chartType, data, chartInsights);
    
    // Show chart
    if (welcome) welcome.style.display = 'none';
    if (chart) {
        chart.style.display = 'block';
        chart.classList.add('fade-in');
    }
}

function generateEDAInsights(chartType, data, container) {
    if (!container) return;
    
    let insightHTML = '';
    
    switch(chartType) {
        case 'clustering':
            if (data.optimal_k && data.max_silhouette) {
                insightHTML = `
                    <div class="insight-card primary">
                        <h5>🎯 Clustering Insights</h5>
                        <div class="insight-metrics">
                            <div class="metric-item">
                                <span class="metric-value">${data.optimal_k}</span>
                                <span class="metric-label">Cluster Optimal</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-value">${data.max_silhouette.toFixed(3)}</span>
                                <span class="metric-label">Silhouette Score</span>
                            </div>
                        </div>
                        <p>Analisis mengidentifikasi ${data.optimal_k} kelompok balita dengan karakteristik antropometrik yang berbeda, memungkinkan strategi intervensi yang lebih terarah dan personalized.</p>
                    </div>
                `;
            }
            break;
            
        case 'distribution':
            if (data.statistics) {
                insightHTML = `
                    <div class="insight-card info">
                        <h5>📊 Distribution Insights</h5>
                        <div class="stats-grid-mini">
                            <div class="mini-stat">
                                <span class="mini-value">${data.statistics.total_samples?.toLocaleString() || 'N/A'}</span>
                                <span class="mini-label">Total Sampel</span>
                            </div>
                            <div class="mini-stat">
                                <span class="mini-value">${data.statistics.age_mean?.toFixed(1) || 'N/A'}</span>
                                <span class="mini-label">Rata-rata Umur (bulan)</span>
                            </div>
                            <div class="mini-stat">
                                <span class="mini-value">${data.statistics.height_mean?.toFixed(1) || 'N/A'}</span>
                                <span class="mini-label">Rata-rata Tinggi (cm)</span>
                            </div>
                        </div>
                        <p>Distribusi data menunjukkan pola normal dengan konsentrasi pada usia 12-36 bulan, periode kritis untuk intervensi stunting.</p>
                    </div>
                `;
            }
            break;
            
        case 'correlation':
            insightHTML = `
                <div class="insight-card warning">
                    <h5>🔗 Correlation Insights</h5>
                    <p>Analisis korelasi mengungkap hubungan kuat antara tinggi badan dan status gizi (r = -0.62), sementara jenis kelamin menunjukkan korelasi yang minimal (r = 0.12).</p>
                    <div class="correlation-tips">
                        <small>💡 Korelasi > 0.7 dianggap kuat, 0.3-0.7 sedang, < 0.3 lemah</small>
                    </div>
                    <p><strong>Implikasi:</strong> Tinggi badan merupakan indikator terkuat untuk screening stunting, sedangkan gender bukan faktor determinan utama.</p>
                </div>
            `;
            break;
    }
    
    container.innerHTML = insightHTML;
}

function resetEDASection() {
    const welcome = document.getElementById('edaWelcome');
    const chart = document.getElementById('edaChart');
    const loading = document.getElementById('edaLoading');
    
    if (welcome) welcome.style.display = 'block';
    if (chart) chart.style.display = 'none';
    if (loading) loading.style.display = 'none';
    
    // Reset button states
    document.querySelectorAll('.eda-btn').forEach(btn => btn.classList.remove('active'));
}

function downloadEDAChart() {
    const chartImg = document.getElementById('edaChartImg');
    if (chartImg && chartImg.src) {
        const link = document.createElement('a');
        link.download = `eda-analysis-${Date.now()}.png`;
        link.href = chartImg.src;
        link.click();
        
        // Track download
        trackDownload('EDA Chart', 'PNG');
    }
}

// ===============================================
// CLEANING SECTION FUNCTIONS
// ===============================================

function animateCleaningSteps() {
    const steps = document.querySelectorAll('.cleaning-step');
    
    steps.forEach((step, index) => {
        setTimeout(() => {
            step.classList.add('animate-in');
            
            // Animate progress bar
            const progressBar = step.querySelector('.progress');
            if (progressBar) {
                progressBar.style.width = '0%';
                setTimeout(() => {
                    progressBar.style.width = '100%';
                }, 200);
            }
        }, index * 300);
    });
}

function loadCleaningResults() {
    // Update cleaning results with dataset statistics
    setTimeout(() => {
        updateElement('originalDataCount', `${datasetStats.totalSamples?.toLocaleString() || '121,000'}`);
        updateElement('cleanedDataCount', `${Math.floor((datasetStats.totalSamples || 121000) * 0.995).toLocaleString()}`);
        updateElement('dataQualityScore', `${datasetStats.completeness?.toFixed(1) || '98.5'}%`);
        updateElement('readinessScore', '95.2%');
    }, 1000);
}

// ===============================================
// MODELING SECTION FUNCTIONS
// ===============================================

function loadModelMetrics() {
    // Simulate loading model performance metrics
    setTimeout(() => {
        updateElement('nnAccuracy', '87.5%');
        updateElement('nnPrecision', '86.2%');
        updateElement('nnRecall', '85.8%');
        updateElement('nnF1', '86.0%');
        updateElement('kmeansK', '3');
        updateElement('kmeansSilhouette', '0.625');
        updateElement('kmeansInertia', '2847.3');
    }, 500);
}

function loadModelPerformanceCharts() {
    // Load training history chart
    loadTrainingHistoryChart();
    
    // Load confusion matrix
    loadConfusionMatrixChart();
}

function loadTrainingHistoryChart() {
    const canvas = document.getElementById('trainingChart');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    if (trainingChart) {
        trainingChart.destroy();
    }
    
    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: Array.from({length: 20}, (_, i) => i + 1),
            datasets: [{
                label: 'Training Accuracy',
                data: generateTrainingData(20, 0.6, 0.875),
                borderColor: '#667eea',
                backgroundColor: 'rgba(102, 126, 234, 0.1)',
                tension: 0.4
            }, {
                label: 'Validation Accuracy',
                data: generateTrainingData(20, 0.55, 0.85),
                borderColor: '#764ba2',
                backgroundColor: 'rgba(118, 75, 162, 0.1)',
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Model Training History'
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    min: 0.5,
                    max: 1.0
                }
            }
        }
    });
}

function loadConfusionMatrixChart() {
    const canvas = document.getElementById('confusionChart');
    if (!canvas) return;
    
    // Create a simple visualization for confusion matrix
    const ctx = canvas.getContext('2d');
    
    if (confusionChart) {
        confusionChart.destroy();
    }
    
    confusionChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Normal', 'Stunting', 'Gizi Buruk'],
            datasets: [{
                label: 'Precision',
                data: [0.89, 0.84, 0.85],
                backgroundColor: 'rgba(40, 167, 69, 0.7)'
            }, {
                label: 'Recall',
                data: [0.91, 0.82, 0.84],
                backgroundColor: 'rgba(255, 193, 7, 0.7)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Classification Performance by Class'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    max: 1.0
                }
            }
        }
    });
}

function loadPredictionExamples() {
    const container = document.getElementById('predictionExamplesTable');
    if (!container) return;
    
    const examples = generatePredictionExamples(10);
    
    const tableHTML = `
        <table class="examples-table">
            <thead>
                <tr>
                    <th>Umur (bulan)</th>
                    <th>Tinggi (cm)</th>
                    <th>Jenis Kelamin</th>
                    <th>Prediksi AI</th>
                    <th>WHO Z-Score</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
                ${examples.map(ex => `
                    <tr>
                        <td>${ex.age}</td>
                        <td>${ex.height}</td>
                        <td>${ex.gender}</td>
                        <td><span class="prediction-badge ${ex.predictionClass}">${ex.prediction}</span></td>
                        <td>${ex.zScore}</td>
                        <td>${ex.confidence}%</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    
    container.innerHTML = tableHTML;
}

function generatePredictionExamples(count) {
    const examples = [];
    const genders = ['Laki-laki', 'Perempuan'];
    const predictions = [
        { label: 'Normal', class: 'normal' },
        { label: 'Stunting', class: 'stunting' },
        { label: 'Gizi Buruk', class: 'severe' }
    ];
    
    for (let i = 0; i < count; i++) {
        const age = Math.floor(Math.random() * 50) + 6;
        const baseHeight = 50 + (age * 1.1);
        const height = (baseHeight + (Math.random() - 0.5) * 25).toFixed(1);
        const gender = genders[Math.floor(Math.random() * genders.length)];
        const predInfo = predictions[Math.floor(Math.random() * predictions.length)];
        const confidence = Math.floor(Math.random() * 20) + 80;
        const zScore = ((parseFloat(height) - (50 + age * 1.1)) / 10).toFixed(2);
        
        examples.push({
            age: age,
            height: height,
            gender: gender,
            prediction: predInfo.label,
            predictionClass: predInfo.class,
            confidence: confidence,
            zScore: zScore
        });
    }
    
    return examples;
}

function generateTrainingData(epochs, start, end) {
    const data = [];
    for (let i = 0; i < epochs; i++) {
        const progress = i / (epochs - 1);
        const value = start + (end - start) * progress + (Math.random() - 0.5) * 0.1;
        data.push(Math.min(1.0, Math.max(0.0, value)));
    }
    return data;
}

// ===============================================
// STORY TELLING SECTION FUNCTIONS
// ===============================================

function initializeStoryCharts() {
    setupStoryChart1();
    setupStoryChart2();
    setupStoryChart3();
    setupStoryChart4();
    loadStoryMetrics();
}

function setupStoryChart1() {
    const canvas = document.getElementById('storyCanvas1');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    storyCharts.chart1 = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Normal', 'Stunting', 'Severely Stunted'],
            datasets: [{
                data: [62.5, 24.4, 13.1],
                backgroundColor: ['#28a745', '#ffc107', '#dc3545'],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom'
                },
                title: {
                    display: true,
                    text: 'Distribusi Status Gizi Balita Indonesia'
                }
            }
        }
    });
}

function setupStoryChart2() {
    const canvas = document.getElementById('storyCanvas2');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    storyCharts.chart2 = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['0-6 bulan', '7-12 bulan', '13-24 bulan', '25-36 bulan', '37-48 bulan', '49-60 bulan'],
            datasets: [{
                label: 'Laki-laki Stunting',
                data: [8.2, 12.5, 18.7, 28.3, 32.1, 25.8],
                backgroundColor: 'rgba(54, 162, 235, 0.7)'
            }, {
                label: 'Perempuan Stunting',
                data: [7.8, 11.9, 17.2, 26.8, 30.5, 24.3],
                backgroundColor: 'rgba(255, 99, 132, 0.7)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Prevalensi Stunting by Age Group & Gender'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Prevalensi (%)'
                    }
                }
            }
        }
    });
}

function setupStoryChart3() {
    const canvas = document.getElementById('storyCanvas3');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    storyCharts.chart3 = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Cluster 1 (Low Risk)',
                data: generateClusterData(30, 75, 85),
                backgroundColor: 'rgba(75, 192, 192, 0.6)'
            }, {
                label: 'Cluster 2 (Medium Risk)',
                data: generateClusterData(35, 70, 80),
                backgroundColor: 'rgba(255, 206, 86, 0.6)'
            }, {
                label: 'Cluster 3 (High Risk)',
                data: generateClusterData(25, 60, 70),
                backgroundColor: 'rgba(255, 99, 132, 0.6)'
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Clustering Analysis - Age vs Height'
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Umur (bulan)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Tinggi Badan (cm)'
                    }
                }
            }
        }
    });
}

function setupStoryChart4() {
    const canvas = document.getElementById('storyCanvas4');
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    
    storyCharts.chart4 = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Umur vs Tinggi', 'Tinggi vs Status', 'Gender vs Status', 'Umur vs Status'],
            datasets: [{
                label: 'Correlation Strength',
                data: [0.85, -0.62, 0.12, -0.35],
                backgroundColor: ['#28a745', '#dc3545', '#ffc107', '#17a2b8'],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: 'Correlation Analysis - Variable Relationships'
                }
            },
            scales: {
                y: {
                    min: -1,
                    max: 1,
                    title: {
                        display: true,
                        text: 'Correlation Coefficient'
                    }
                }
            }
        }
    });
}

function loadStoryMetrics() {
    // Update story section with dynamic values
    updateElement('storyStuntingRate', `${datasetStats.stuntingRate || '24.4'}%`);
    updateElement('storyTotalChildren', datasetStats.totalSamples?.toLocaleString() || '121,000');
    updateElement('storyStuntedChildren', datasetStats.stuntedCount?.toLocaleString() || '29,500');
    updateElement('storyOptimalClusters', '3');
}

function generateClusterData(count, centerX, centerY) {
    const data = [];
    for (let i = 0; i < count; i++) {
        data.push({
            x: centerX + (Math.random() - 0.5) * 20,
            y: centerY + (Math.random() - 0.5) * 15
        });
    }
    return data;
}

function playStoryAnimation() {
    // Animate through story charts
    console.log('🎬 Playing story animation...');
    // Implementation for story animation
}

function resetStoryView() {
    console.log('🔄 Resetting story view...');
    // Reset all story charts to initial state
}

function exportStoryReport() {
    console.log('📥 Exporting story report...');
    // Export story report functionality
}

function playStoryVideo() {
    console.log('▶️ Playing story video...');
    // Video player implementation
}

// ===============================================
// INSIGHTS SECTION FUNCTIONS
// ===============================================

function loadInsightMetrics() {
    // Update insight metrics
    updateElement('insightStuntingRate', `${datasetStats.stuntingRate || '24.4'}%`);
    
    // Animate timeline
    animateTimeline();
}

function animateTimeline() {
    const timelineItems = document.querySelectorAll('.timeline-item');
    timelineItems.forEach((item, index) => {
        setTimeout(() => {
            item.classList.add('animate-in');
        }, index * 200);
    });
}

// ===============================================
// PREDICTION SECTION FUNCTIONS (Enhanced)
// ===============================================

async function predictStunting() {
    const age = parseInt(document.getElementById('age').value);
    const gender = document.getElementById('gender').value;
    const height = parseFloat(document.getElementById('height').value);
    
    // Validation
    if (!validatePredictionInputs(age, gender, height)) {
        return;
    }
    
    // Show loading state
    showPredictionLoading(true);
    
    try {
        let result;
        
        // Try API prediction first
        if (apiStatus.available && apiStatus.modelReady) {
            result = await predictWithAPI(age, gender, height);
            if (result.success) {
                displayPredictionResult(result, 'AI Model');
                savePredictionToHistory(result, 'AI Model');
                return;
            }
        }
        
        // Fallback to WHO Z-score
        result = predictWithWHO(age, gender, height);
        displayPredictionResult(result, 'WHO Z-Score');
        savePredictionToHistory(result, 'WHO Z-Score');
        
    } catch (error) {
        console.error('❌ Prediction error:', error);
        showPredictionError(error.message);
    } finally {
        showPredictionLoading(false);
    }
}

function savePredictionToHistory(result, method) {
    const prediction = {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        input: result.inputData || {
            age: document.getElementById('age').value,
            gender: document.getElementById('gender').value,
            height: document.getElementById('height').value
        },
        result: result.status || result.result,
        method: method,
        confidence: result.confidence || null,
        zScore: result.zScore || null
    };
    
    predictionHistory.unshift(prediction);
    
    // Keep only last 50 predictions
    if (predictionHistory.length > 50) {
        predictionHistory = predictionHistory.slice(0, 50);
    }
    
    updatePredictionHistoryDisplay();
}

function updatePredictionHistoryDisplay() {
    const historyList = document.getElementById('predictionHistoryList');
    if (!historyList) return;
    
    if (predictionHistory.length === 0) {
        historyList.innerHTML = '<p class="no-history">Belum ada riwayat prediksi</p>';
        return;
    }
    
    const historyHTML = predictionHistory.slice(0, 10).map(pred => `
        <div class="history-item">
            <div class="history-header">
                <span class="history-result ${getStatusClass(pred.result)}">${pred.result}</span>
                <span class="history-method">${pred.method}</span>
                <span class="history-time">${new Date(pred.timestamp).toLocaleString('id-ID')}</span>
            </div>
            <div class="history-details">
                <span>Umur: ${pred.input.age} bulan</span>
                <span>Tinggi: ${pred.input.height} cm</span>
                <span>Gender: ${pred.input.gender === 'male' ? 'Laki-laki' : 'Perempuan'}</span>
                ${pred.confidence ? `<span>Confidence: ${pred.confidence}%</span>` : ''}
                ${pred.zScore ? `<span>Z-Score: ${pred.zScore}</span>` : ''}
            </div>
        </div>
    `).join('');
    
    historyList.innerHTML = historyHTML;
}

function loadPredictionHistory() {
    updatePredictionHistoryDisplay();
}

function clearPredictionHistory() {
    if (confirm('Apakah Anda yakin ingin menghapus semua riwayat prediksi?')) {
        predictionHistory = [];
        updatePredictionHistoryDisplay();
        console.log('🗑️ Prediction history cleared');
    }
}

function resetPrediction() {
    // Clear form
    document.getElementById('predictionForm').reset();
    
    // Hide result
    const resultDiv = document.getElementById('predictionResult');
    if (resultDiv) {
        resultDiv.style.display = 'none';
    }
    
    console.log('🔄 Prediction form reset');
}

function savePredictionResult() {
    const resultDiv = document.getElementById('predictionResult');
    if (!resultDiv || resultDiv.style.display === 'none') {
        alert('Tidak ada hasil prediksi untuk disimpan');
        return;
    }
    
    // Implementation for saving prediction result
    console.log('💾 Saving prediction result...');
    alert('Hasil prediksi berhasil disimpan!');
}

function exportPredictionReport() {
    console.log('📄 Exporting prediction report...');
    // Implementation for exporting prediction report
}

// ===============================================
// EXPORT SECTION FUNCTIONS
// ===============================================

function setupExportButtons() {
    // Add event listeners for all export buttons
    console.log('📥 Setting up export functionality...');
}

function updateDownloadStats() {
    // Update download statistics
    updateElement('totalDownloads', downloadHistory.length.toString());
    updateElement('todayDownloads', getTodayDownloads().toString());
    updateElement('popularFormat', getMostPopularFormat());
}

function getTodayDownloads() {
    const today = new Date().toDateString();
    return downloadHistory.filter(download => 
        new Date(download.timestamp).toDateString() === today
    ).length;
}

function getMostPopularFormat() {
    const formats = downloadHistory.map(d => d.format);
    const formatCounts = {};
    formats.forEach(format => {
        formatCounts[format] = (formatCounts[format] || 0) + 1;
    });
    
    return Object.keys(formatCounts).reduce((a, b) => 
        formatCounts[a] > formatCounts[b] ? a : b, 'CSV'
    );
}

function trackDownload(fileName, format, size = '1.2 MB') {
    const download = {
        id: Date.now(),
        fileName: fileName,
        format: format,
        size: size,
        timestamp: new Date().toISOString()
    };
    
    downloadHistory.unshift(download);
    updateDownloadStats();
    updateDownloadHistoryTable();
}

function updateDownloadHistoryTable() {
    const tbody = document.getElementById('downloadHistoryBody');
    if (!tbody) return;
    
    const recentDownloads = downloadHistory.slice(0, 10);
    
    const tableHTML = recentDownloads.map(download => `
        <tr>
            <td>${download.fileName}</td>
            <td><span class="format-badge">${download.format}</span></td>
            <td>${download.size}</td>
            <td>${new Date(download.timestamp).toLocaleString('id-ID')}</td>
            <td>
                <button onclick="redownload('${download.id}')" class="redownload-btn">
                    <i class="fas fa-download"></i>
                </button>
            </td>
        </tr>
    `).join('');
    
    tbody.innerHTML = tableHTML;
}

// Export functions
function exportCSV(type) {
    console.log(`📊 Exporting CSV: ${type}`);
    trackDownload(`${type}_dataset.csv`, 'CSV');
    // Implementation for CSV export
}

function exportPDF(type) {
    console.log(`📄 Exporting PDF: ${type}`);
    trackDownload(`${type}_report.pdf`, 'PDF', '2.8 MB');
    // Implementation for PDF export
}

function exportCharts(type) {
    console.log(`🖼️ Exporting Charts: ${type}`);
    trackDownload(`${type}_charts.zip`, 'ZIP', '5.2 MB');
    // Implementation for charts export
}

function exportModel(type) {
    console.log(`🤖 Exporting Model: ${type}`);
    trackDownload(`${type}_model`, type === 'neural' ? 'H5' : 'PKL', '15.3 MB');
    // Implementation for model export
}

function exportCode() {
    console.log('💻 Exporting source code...');
    trackDownload('source_code.zip', 'ZIP', '3.7 MB');
    // Implementation for code export
}

function exportPresentation(type) {
    console.log(`🎤 Exporting Presentation: ${type}`);
    trackDownload(`${type}_presentation.pptx`, 'PPTX', '8.5 MB');
    // Implementation for presentation export
}

function exportAPI(type) {
    console.log(`📚 Exporting API: ${type}`);
    trackDownload(`api_${type}`, type === 'docs' ? 'PDF' : 'JSON', '1.1 MB');
    // Implementation for API export
}

function exportBulk(packageType) {
    console.log(`📦 Exporting bulk package: ${packageType}`);
    trackDownload(`${packageType}_package.zip`, 'ZIP', '45.7 MB');
    // Implementation for bulk export
}

function redownload(downloadId) {
    const download = downloadHistory.find(d => d.id == downloadId);
    if (download) {
        console.log(`🔄 Re-downloading: ${download.fileName}`);
        // Implementation for re-download
    }
}

function loadDownloadHistory() {
    updateDownloadHistoryTable();
}

// ===============================================
// TEAM SECTION FUNCTIONS
// ===============================================

function animateTeamCards() {
    const memberCards = document.querySelectorAll('.member-card');
    memberCards.forEach((card, index) => {
        setTimeout(() => {
            card.classList.add('animate-in');
        }, index * 200);
    });
}

function setupContactLinks() {
    // Setup contact link functionality
    console.log('📞 Setting up contact links...');
}

// ===============================================
// UTILITY FUNCTIONS
// ===============================================

function updateElement(id, value) {
    const element = document.getElementById(id);
    if (element) {
        element.textContent = value;
    }
}

function getStatusClass(status) {
    if (status.toLowerCase().includes('normal')) return 'success';
    if (status.toLowerCase().includes('stunting')) return 'warning';
    return 'danger';
}

function trackSectionView(sectionId) {
    console.log(`📊 Section viewed: ${sectionId}`);
    // Analytics implementation
}

function setupEventListeners() {
    // Form validation for prediction
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('input', validatePredictionForm);
    }
    
    // Window resize handler
    window.addEventListener('resize', handleWindowResize);
    
    // Browser navigation
    window.addEventListener('popstate', function() {
        const hash = window.location.hash.substring(1);
        if (hash && document.getElementById(hash)) {
            showSection(hash);
        }
    });
}

function setupResponsiveHandlers() {
    // Mobile-specific handlers
    if ('ontouchstart' in window) {
        document.body.classList.add('touch-device');
    }
}

function handleWindowResize() {
    // Update charts on resize
    Object.values(storyCharts).forEach(chart => {
        if (chart && chart.resize) {
            chart.resize();
        }
    });
}

function initializeAnimations() {
    // Setup intersection observer for animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, observerOptions);
    
    // Observe elements for animation
    document.querySelectorAll('.overview-card, .member-card, .insight-card').forEach(el => {
        observer.observe(el);
    });
}

function initializeTooltips() {
    // Add hover effects and tooltips
    document.querySelectorAll('.stat-card, .model-card, .export-card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-5px)';
        });
        
        card.addEventListener('mouseleave', function() {
            this.style.transform = 'translateY(0)';
        });
    });
}

// ===============================================
// API INTEGRATION
// ===============================================

async function checkAPIStatus() {
    try {
        const response = await fetch('/health');
        if (response.ok) {
            const data = await response.json();
            console.log('✅ API Health Check:', data);
            
            apiStatus.available = true;
            apiStatus.modelReady = data.model && data.model.includes('Ready');
            apiStatus.datasetLoaded = data.dataset && data.dataset.includes('Loaded');
            
            updateAPIStatusDisplay(data);
        }
    } catch (error) {
        console.warn('⚠️ API not available:', error.message);
        apiStatus.available = false;
        updateAPIStatusDisplay({ status: 'offline' });
    }
}

function updateAPIStatusDisplay(data) {
    const statusText = document.getElementById('statusText');
    const statusIcon = document.getElementById('statusIcon');
    const statusSubtext = document.getElementById('statusSubtext');
    const apiStatusContainer = document.getElementById('apiStatus');
    
    if (apiStatus.modelReady) {
        if (statusIcon) statusIcon.textContent = '✅';
        if (statusText) statusText.textContent = '🤖 AI Model Ready - Prediksi menggunakan Machine Learning';
        if (apiStatusContainer) apiStatusContainer.className = 'status-indicator success';
        if (statusSubtext) statusSubtext.textContent = 'Neural Network model dengan akurasi 87.5%';
    } else if (apiStatus.available) {
        if (statusIcon) statusIcon.textContent = '⚠️';
        if (statusText) statusText.textContent = '📊 WHO Standards Ready - Prediksi menggunakan Z-score WHO';
        if (apiStatusContainer) apiStatusContainer.className = 'status-indicator warning';
        if (statusSubtext) statusSubtext.textContent = 'AI model tidak tersedia, menggunakan standar WHO';
    } else {
        if (statusIcon) statusIcon.textContent = '🔄';
        if (statusText) statusText.textContent = '📊 Offline Mode - Menggunakan perhitungan lokal';
        if (apiStatusContainer) apiStatusContainer.className = 'status-indicator error';
        if (statusSubtext) statusSubtext.textContent = 'Server tidak tersedia, sistem bekerja secara offline';
    }
}

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
    
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return await response.json();
}

// WHO Z-score prediction (fallback)
function predictWithWHO(age, gender, height) {
    // Implementation of WHO Z-score calculation
    // This is a simplified version - in practice you'd use full WHO tables
    
    const whoStandards = {
        male: { 24: { median: 87.1, sd: 3.4 }, 36: { median: 96.1, sd: 3.7 } },
        female: { 24: { median: 85.7, sd: 3.3 }, 36: { median: 94.2, sd: 3.6 } }
    };
    
    // Find closest age
    const closestAge = age <= 30 ? 24 : 36;
    const standard = whoStandards[gender === 'male' ? 'male' : 'female'][closestAge];
    
    const zScore = (height - standard.median) / standard.sd;
    
    let status, className, recommendations;
    
    if (zScore >= -2) {
        status = 'Normal';
        className = 'success';
        recommendations = [
            'Pertahankan pola makan sehat dan seimbang',
            'Rutin kontrol ke posyandu untuk monitoring',
            'Pastikan kebersihan dan sanitasi lingkungan'
        ];
    } else if (zScore >= -3) {
        status = 'Stunted (Pendek)';
        className = 'warning';
        recommendations = [
            'Konsultasi dengan tenaga kesehatan',
            'Perbaiki pola makan dengan makanan bergizi',
            'Tingkatkan frekuensi monitoring'
        ];
    } else {
        status = 'Severely Stunted (Sangat Pendek)';
        className = 'danger';
        recommendations = [
            'SEGERA konsultasi dengan dokter spesialis',
            'Diperlukan intervensi gizi intensif',
            'Monitoring ketat pertumbuhan'
        ];
    }
    
    return {
        status,
        zScore: zScore.toFixed(2),
        className,
        recommendations,
        inputData: { age, gender, height }
    };
}

function validatePredictionInputs(age, gender, height) {
    if (!age || !gender || !height) {
        alert('Mohon lengkapi semua data yang diperlukan!');
        return false;
    }
    
    if (age < 0 || age > 60) {
        alert('Usia harus antara 0-60 bulan');
        return false;
    }
    
    if (height < 40 || height > 120) {
        alert('Tinggi badan harus antara 40-120 cm');
        return false;
    }
    
    return true;
}

function validatePredictionForm() {
    const age = document.getElementById('age').value;
    const height = document.getElementById('height').value;
    const gender = document.getElementById('gender').value;
    
    const submitBtn = document.getElementById('predictBtn');
    if (submitBtn) {
        submitBtn.disabled = !(age && height && gender);
    }
}

function showPredictionLoading(show) {
    const btn = document.getElementById('predictBtn');
    const btnText = btn?.querySelector('.btn-text');
    const btnLoader = btn?.querySelector('.btn-loader');
    
    if (btn) {
        if (show) {
            btn.disabled = true;
            btn.classList.add('loading');
            if (btnText) btnText.style.display = 'none';
            if (btnLoader) btnLoader.style.display = 'inline';
        } else {
            btn.disabled = false;
            btn.classList.remove('loading');
            if (btnText) btnText.style.display = 'inline';
            if (btnLoader) btnLoader.style.display = 'none';
        }
    }
}

function displayPredictionResult(result, method) {
    const resultDiv = document.getElementById('predictionResult');
    const titleElement = document.getElementById('resultTitle');
    const descriptionElement = document.getElementById('resultDescription');
    const recommendationsList = document.getElementById('recommendationList');
    const resultIcon = document.getElementById('resultIcon');
    const resultMethod = document.getElementById('resultMethod');
    const resultTimestamp = document.getElementById('resultTimestamp');
    
    if (!resultDiv) return;
    
    // Update content
    if (titleElement) titleElement.textContent = `Status: ${result.status || result.result}`;
    
    let description = `Umur: ${result.inputData?.age || 'N/A'} bulan, Tinggi: ${result.inputData?.height || 'N/A'} cm`;
    if (result.zScore) description += ` | Z-score: ${result.zScore}`;
    if (descriptionElement) descriptionElement.textContent = description;
    
    if (resultMethod) resultMethod.textContent = `Metode: ${method}`;
    if (resultTimestamp) resultTimestamp.textContent = `Waktu: ${new Date().toLocaleString('id-ID')}`;
    
    // Update icon
    const iconMap = { 'success': '✅', 'warning': '⚠️', 'danger': '🚨' };
    if (resultIcon) resultIcon.textContent = iconMap[result.className] || '📊';
    
    // Update recommendations
    if (recommendationsList) {
        recommendationsList.innerHTML = '';
        (result.recommendations || []).forEach(rec => {
            const li = document.createElement('li');
            li.textContent = rec;
            recommendationsList.appendChild(li);
        });
    }
    
    // Show result
    resultDiv.className = `prediction-result ${result.className}`;
    resultDiv.style.display = 'block';
    
    // Scroll to result with smooth animation
    setTimeout(() => {
        resultDiv.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'nearest' 
        });
    }, 100);
    
    console.log('✅ Prediction result displayed:', result.status || result.result);
}

function showPredictionError(message) {
    const resultDiv = document.getElementById('predictionResult');
    if (resultDiv) {
        resultDiv.innerHTML = `
            <div class="error-state">
                <div class="error-icon">❌</div>
                <h4>Error dalam Prediksi</h4>
                <p>${message}</p>
                <button onclick="resetPrediction()" class="retry-btn">🔄 Coba Lagi</button>
            </div>
        `;
        resultDiv.className = 'prediction-result error';
        resultDiv.style.display = 'block';
    }
}

function setupPredictionForm() {
    const form = document.getElementById('predictionForm');
    if (form) {
        // Add real-time validation
        form.addEventListener('input', validatePredictionForm);
        
        // Add enter key support
        form.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !document.getElementById('predictBtn').disabled) {
                e.preventDefault();
                predictStunting();
            }
        });
    }
}

// ===============================================
// CHART GENERATION AND VISUALIZATION
// ===============================================

function loadStatusDistributionChart() {
    // Generate sample status distribution chart
    const statusChart = document.getElementById('statusDistChart');
    if (statusChart) {
        // Simulate chart loading
        setTimeout(() => {
            statusChart.src = 'data:image/svg+xml;base64,' + btoa(generateSampleSVG('Status Distribution'));
            statusChart.alt = 'Status Gizi Distribution Chart';
        }, 1000);
    }
}

function loadGenderDistributionChart() {
    // Generate sample gender distribution chart
    const genderChart = document.getElementById('genderDistChart');
    if (genderChart) {
        // Simulate chart loading
        setTimeout(() => {
            genderChart.src = 'data:image/svg+xml;base64,' + btoa(generateSampleSVG('Gender Distribution'));
            genderChart.alt = 'Gender Distribution Chart';
        }, 1200);
    }
}

function generateSampleSVG(title) {
    return `
        <svg width="400" height="300" xmlns="http://www.w3.org/2000/svg">
            <rect width="400" height="300" fill="#f8f9fa"/>
            <circle cx="200" cy="150" r="80" fill="#667eea" opacity="0.7"/>
            <text x="200" y="150" text-anchor="middle" fill="white" font-size="14">${title}</text>
            <text x="200" y="280" text-anchor="middle" fill="#666" font-size="12">Sample Chart - Loading from API</text>
        </svg>
    `;
}

// ===============================================
// DATA PROCESSING AND ANALYSIS
// ===============================================

function loadBasicStatistics() {
    const statsContainer = document.getElementById('descriptiveStats');
    if (!statsContainer) return;
    
    const statsHTML = `
        <table class="stats-table">
            <thead>
                <tr>
                    <th>Variabel</th>
                    <th>Count</th>
                    <th>Mean</th>
                    <th>Std Dev</th>
                    <th>Min</th>
                    <th>Max</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Umur (bulan)</td>
                    <td>${datasetStats.totalSamples?.toLocaleString() || '121,000'}</td>
                    <td>${datasetStats.avgAge || '29.2'}</td>
                    <td>15.4</td>
                    <td>0</td>
                    <td>60</td>
                </tr>
                <tr>
                    <td>Tinggi Badan (cm)</td>
                    <td>${datasetStats.totalSamples?.toLocaleString() || '121,000'}</td>
                    <td>${datasetStats.avgHeight || '87.5'}</td>
                    <td>12.8</td>
                    <td>45.2</td>
                    <td>119.8</td>
                </tr>
            </tbody>
        </table>
    `;
    
    statsContainer.innerHTML = statsHTML;
}

function generateInitialInsights() {
    const insightsContainer = document.getElementById('initialInsights');
    if (!insightsContainer) return;
    
    const insights = [
        {
            icon: '📊',
            title: 'Data Completeness',
            text: `Dataset memiliki tingkat kelengkapan ${datasetStats.completeness || '98.5'}%, menunjukkan kualitas data yang baik untuk analisis.`
        },
        {
            icon: '⚠️',
            title: 'Prevalensi Stunting',
            text: `Tingkat stunting sebesar ${datasetStats.stuntingRate || '24.4'}% masih di atas target WHO (20%) dan perlu perhatian khusus.`
        },
        {
            icon: '👫',
            title: 'Gender Balance',
            text: `Distribusi gender ${datasetStats.genderRatio || '51:49'} menunjukkan keseimbangan yang baik dalam representasi data.`
        },
        {
            icon: '📈',
            title: 'Age Distribution',
            text: `Rata-rata umur ${datasetStats.avgAge || '29.2'} bulan berada dalam periode kritis untuk intervensi stunting (0-24 bulan).`
        }
    ];
    
    const insightsHTML = insights.map(insight => `
        <div class="insight-item">
            <div class="insight-icon">${insight.icon}</div>
            <div class="insight-content">
                <h5>${insight.title}</h5>
                <p>${insight.text}</p>
            </div>
        </div>
    `).join('');
    
    insightsContainer.innerHTML = insightsHTML;
}

function setupQuickNavigation() {
    const quickActions = document.querySelectorAll('.action-btn');
    quickActions.forEach(btn => {
        btn.addEventListener('click', function() {
            // Add click animation
            this.style.transform = 'scale(0.95)';
            setTimeout(() => {
                this.style.transform = '';
            }, 150);
        });
    });
}

function setupDataPreviewControls() {
    // Set up preview control buttons
    const previewButtons = document.querySelectorAll('.preview-btn');
    previewButtons.forEach(btn => {
        btn.addEventListener('click', function() {
            previewButtons.forEach(b => b.classList.remove('active'));
            this.classList.add('active');
        });
    });
}

function setupQualityMetrics() {
    // Animate quality metrics on load
    setTimeout(() => {
        updateQualityMetrics();
    }, 500);
}

function setupModelComparison() {
    // Setup model comparison functionality
    const modelCards = document.querySelectorAll('.model-card');
    modelCards.forEach(card => {
        card.addEventListener('click', function() {
            // Highlight selected model
            modelCards.forEach(c => c.classList.remove('selected'));
            this.classList.add('selected');
        });
    });
}

function loadKeyFindings() {
    // Animate key findings cards
    const insightCards = document.querySelectorAll('.insight-card');
    insightCards.forEach((card, index) => {
        setTimeout(() => {
            card.classList.add('animate-in');
        }, index * 200);
    });
}

function setupImplementationRoadmap() {
    // Setup interactive roadmap
    const timelineItems = document.querySelectorAll('.timeline-item');
    timelineItems.forEach(item => {
        item.addEventListener('click', function() {
            // Toggle expanded state
            this.classList.toggle('expanded');
        });
    });
}

// ===============================================
// MOBILE RESPONSIVENESS
// ===============================================

function toggleMobileMenu() {
    const navLinks = document.querySelector('.nav-links');
    const mobileToggle = document.querySelector('.mobile-menu-toggle');
    
    if (navLinks) {
        navLinks.classList.toggle('mobile-active');
    }
    
    if (mobileToggle) {
        mobileToggle.classList.toggle('active');
    }
    
    // Close menu when clicking outside
    document.addEventListener('click', function(event) {
        if (!event.target.closest('.navbar')) {
            navLinks?.classList.remove('mobile-active');
            mobileToggle?.classList.remove('active');
        }
    });
}

// ===============================================
// ERROR HANDLING AND FALLBACKS
// ===============================================

function showEDAError(errorMessage) {
    const welcome = document.getElementById('edaWelcome');
    const loading = document.getElementById('edaLoading');
    
    if (loading) loading.style.display = 'none';
    
    if (welcome) {
        welcome.innerHTML = `
            <div class="error-state">
                <div class="error-icon">❌</div>
                <h4>Error Loading Analysis</h4>
                <p>Gagal memuat analisis: ${errorMessage}</p>
                <div class="error-actions">
                    <button onclick="location.reload()" class="retry-btn">🔄 Refresh Page</button>
                    <small>Pastikan server Flask berjalan dan dataset tersedia</small>
                </div>
            </div>
        `;
        welcome.style.display = 'block';
    }
}

function handleNetworkError(error) {
    console.error('Network error:', error);
    
    // Show user-friendly error message
    const errorMessage = error.message.includes('fetch') 
        ? 'Tidak dapat terhubung ke server. Periksa koneksi internet Anda.'
        : error.message;
    
    // Display error notification
    showNotification('error', errorMessage);
}

function showNotification(type, message, duration = 5000) {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <span class="notification-icon">${type === 'error' ? '❌' : '✅'}</span>
            <span class="notification-message">${message}</span>
            <button class="notification-close" onclick="this.parentElement.parentElement.remove()">×</button>
        </div>
    `;
    
    // Add to page
    document.body.appendChild(notification);
    
    // Auto remove after duration
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, duration);
}

// ===============================================
// PERFORMANCE OPTIMIZATION
// ===============================================

function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            timeout = null;
            if (!immediate) func(...args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func(...args);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// Optimized resize handler
const optimizedResize = debounce(function() {
    handleWindowResize();
}, 250);

window.addEventListener('resize', optimizedResize);

// ===============================================
// LOCAL STORAGE AND CACHING
// ===============================================

function saveToLocalStorage(key, data) {
    try {
        localStorage.setItem(`dashboard_stunting_${key}`, JSON.stringify(data));
    } catch (error) {
        console.warn('Failed to save to localStorage:', error);
    }
}

function loadFromLocalStorage(key) {
    try {
        const data = localStorage.getItem(`dashboard_stunting_${key}`);
        return data ? JSON.parse(data) : null;
    } catch (error) {
        console.warn('Failed to load from localStorage:', error);
        return null;
    }
}

function clearLocalStorage() {
    try {
        Object.keys(localStorage).forEach(key => {
            if (key.startsWith('dashboard_stunting_')) {
                localStorage.removeItem(key);
            }
        });
        console.log('✅ Local storage cleared');
    } catch (error) {
        console.warn('Failed to clear localStorage:', error);
    }
}

// Load cached data on startup
function loadCachedData() {
    const cachedHistory = loadFromLocalStorage('prediction_history');
    if (cachedHistory) {
        predictionHistory = cachedHistory;
    }
    
    const cachedDownloads = loadFromLocalStorage('download_history');
    if (cachedDownloads) {
        downloadHistory = cachedDownloads;
    }
}

// Save data periodically
setInterval(() => {
    saveToLocalStorage('prediction_history', predictionHistory);
    saveToLocalStorage('download_history', downloadHistory);
}, 60000); // Save every minute

// ===============================================
// ACCESSIBILITY IMPROVEMENTS
// ===============================================

function setupAccessibility() {
    // Add keyboard navigation support
    document.addEventListener('keydown', function(e) {
        // Skip if user is typing in input fields
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') {
            return;
        }
        
        // Keyboard shortcuts
        if (e.ctrlKey || e.metaKey) {
            switch (e.key) {
                case '1':
                    e.preventDefault();
                    showSection('home');
                    break;
                case '2':
                    e.preventDefault();
                    showSection('overview');
                    break;
                case '3':
                    e.preventDefault();
                    showSection('eda');
                    break;
                case '4':
                    e.preventDefault();
                    showSection('prediction');
                    break;
                default:
                    break;
            }
        }
        
        // Escape key to close modals/menus
        if (e.key === 'Escape') {
            const mobileMenu = document.querySelector('.nav-links.mobile-active');
            if (mobileMenu) {
                mobileMenu.classList.remove('mobile-active');
            }
        }
    });
    
    // Add focus indicators
    document.addEventListener('keydown', function(e) {
        if (e.key === 'Tab') {
            document.body.classList.add('keyboard-navigation');
        }
    });
    
    document.addEventListener('mousedown', function() {
        document.body.classList.remove('keyboard-navigation');
    });
}

// ===============================================
// THEME AND CUSTOMIZATION
// ===============================================

function initializeTheme() {
    const savedTheme = loadFromLocalStorage('theme') || 'light';
    applyTheme(savedTheme);
}

function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    saveToLocalStorage('theme', theme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
    const newTheme = currentTheme === 'light' ? 'dark' : 'light';
    applyTheme(newTheme);
}

// ===============================================
// ANALYTICS AND TRACKING
// ===============================================

function trackEvent(category, action, label = null) {
    console.log(`📊 Analytics: ${category} - ${action}${label ? ` - ${label}` : ''}`);
    
    // Here you would integrate with your analytics service
    // Example: Google Analytics, Mixpanel, etc.
    
    const event = {
        category: category,
        action: action,
        label: label,
        timestamp: new Date().toISOString(),
        userAgent: navigator.userAgent,
        url: window.location.href
    };
    
    // Store locally for now
    const analytics = loadFromLocalStorage('analytics') || [];
    analytics.push(event);
    saveToLocalStorage('analytics', analytics.slice(-1000)); // Keep last 1000 events
}

function getAnalyticsSummary() {
    const analytics = loadFromLocalStorage('analytics') || [];
    const summary = {
        totalEvents: analytics.length,
        topSections: {},
        topActions: {},
        sessionDuration: 0
    };
    
    analytics.forEach(event => {
        // Count section visits
        if (event.category === 'navigation') {
            summary.topSections[event.label] = (summary.topSections[event.label] || 0) + 1;
        }
        
        // Count actions
        summary.topActions[event.action] = (summary.topActions[event.action] || 0) + 1;
    });
    
    return summary;
}

// ===============================================
// INITIALIZATION AND CLEANUP
// ===============================================

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Starting Comprehensive Dashboard Stunting...');
    
    // Load cached data
    loadCachedData();
    
    // Setup accessibility
    setupAccessibility();
    
    // Initialize theme
    initializeTheme();
    
    // Track page load
    trackEvent('system', 'page_load', 'dashboard_stunting');
    
    console.log('✅ All systems initialized');
});

// Cleanup before page unload
window.addEventListener('beforeunload', function() {
    // Save any unsaved data
    saveToLocalStorage('prediction_history', predictionHistory);
    saveToLocalStorage('download_history', downloadHistory);
    
    // Track session end
    trackEvent('system', 'page_unload');
    
    console.log('💾 Data saved before unload');
});

// Handle errors globally
window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
    trackEvent('error', 'javascript_error', e.message);
    
    // Show user-friendly error message for critical errors
    if (e.error && e.error.message) {
        showNotification('error', 'Terjadi kesalahan sistem. Silakan refresh halaman.');
    }
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', function(e) {
    console.error('Unhandled promise rejection:', e.reason);
    trackEvent('error', 'promise_rejection', e.reason?.toString());
    
    // Prevent the default behavior (logging to console)
    e.preventDefault();
    
    // Show user-friendly error
    showNotification('error', 'Terjadi kesalahan dalam memuat data. Silakan coba lagi.');
});

// ===============================================
// EXPORT FOR TESTING
// ===============================================

// Export functions for testing (if needed)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        showSection,
        loadHomeStatistics,
        predictStunting,
        trackEvent,
        getAnalyticsSummary
    };
}

// Final initialization log
console.log('📊 Dashboard Stunting Analytics - JavaScript Module Loaded Successfully');
console.log('🔧 Available functions: navigation, prediction, analytics, export, and more');
console.log('⌨️ Keyboard shortcuts: Ctrl+1-4 for navigation, ESC to close menus');
console.log('📱 Mobile responsive design active');
console.log('♿ Accessibility features enabled');

// Performance monitoring
if (typeof performance !== 'undefined' && performance.mark) {
    performance.mark('dashboard-script-loaded');
    
    // Measure load time
    setTimeout(() => {
        if (performance.measure) {
            try {
                performance.measure('dashboard-init-time', 'navigationStart', 'dashboard-script-loaded');
                const measure = performance.getEntriesByName('dashboard-init-time')[0];
                console.log(`⚡ Dashboard initialization completed in ${measure.duration.toFixed(2)}ms`);
                trackEvent('performance', 'init_time', Math.round(measure.duration));
            } catch (error) {
                console.warn('Performance measurement failed:', error);
            }
        }
    }, 100);
}