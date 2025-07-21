// Sample data generation sesuai dengan notebook
const generateSampleData = () => {
    const data = [];
    const genders = ['laki-laki', 'perempuan'];
    const statuses = ['normal', 'stunted', 'severely stunted', 'tinggi'];
    // Berdasarkan data dari notebook: normal ~63.7%, stunted ~25%, severely stunted ~8%, tinggi ~3.3%
    const statusWeights = [0.637, 0.25, 0.08, 0.033];
    
    for (let i = 0; i < 1000; i++) {
        const age = Math.floor(Math.random() * 60);
        const gender = genders[Math.floor(Math.random() * genders.length)];
        
        // Generate height berdasarkan age dengan korelasi seperti di notebook
        let baseHeight = 40 + (age * 0.4) + (Math.random() * 10 - 5);
        
        // Choose status berdasarkan weights
        let statusIndex = 0;
        const rand = Math.random();
        let cumulative = 0;
        for (let j = 0; j < statusWeights.length; j++) {
            cumulative += statusWeights[j];
            if (rand <= cumulative) {
                statusIndex = j;
                break;
            }
        }
        
        const status = statuses[statusIndex];
        
        // Adjust height berdasarkan status
        if (status === 'stunted') baseHeight *= 0.85;
        else if (status === 'severely stunted') baseHeight *= 0.75;
        else if (status === 'tinggi') baseHeight *= 1.15;
        
        data.push({
            umur: age,
            jenisKelamin: gender,
            tinggiBadan: Math.max(30, baseHeight),
            statusGizi: status,
            cluster: Math.floor(Math.random() * 3)
        });
    }
    return data;
};

const sampleData = generateSampleData();

// Navigation functions
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
        if ((sectionId === 'overview' && link.textContent.includes('Data Overview')) ||
            (sectionId === 'eda' && link.textContent.includes('EDA'))) {
            link.classList.add('active');
        }
    });
}

// Visualization functions
function showVisualization(type) {
    // Hide all visualization containers
    document.querySelectorAll('#edaDisplay > div').forEach(div => {
        div.style.display = 'none';
    });
    
    // Show selected visualization
    switch(type) {
        case 'scatter':
            document.getElementById('scatterContainer').style.display = 'block';
            createScatterPlot();
            break;
        case 'correlation':
            document.getElementById('correlationContainer').style.display = 'block';
            createCorrelationHeatmap();
            break;
        case 'clustering':
            document.getElementById('clusteringContainer').style.display = 'block';
            createClusterPlot();
            break;
        case 'interactive':
            document.getElementById('interactiveContainer').style.display = 'block';
            updateScatterPlot();
            break;
    }
}

// Initialize dashboard saat page load
document.addEventListener('DOMContentLoaded', function() {
    loadGenderChart();
    loadStatusChart();
});

// Gender Chart - sesuai dengan notebook
function loadGenderChart() {
    const ctx = document.getElementById('genderChart').getContext('2d');
    const genderCounts = sampleData.reduce((acc, item) => {
        acc[item.jenisKelamin] = (acc[item.jenisKelamin] || 0) + 1;
        return acc;
    }, {});

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Laki-laki', 'Perempuan'],
            datasets: [{
                data: [genderCounts['laki-laki'] || 0, genderCounts['perempuan'] || 0],
                backgroundColor: ['#667eea', '#f093fb'],
                borderColor: ['#5a67d8', '#e879f9'],
                borderWidth: 2,
                borderRadius: 10
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: { display: false }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(0,0,0,0.1)' }
                },
                x: {
                    grid: { display: false }
                }
            }
        }
    });
}

// Status Chart - sesuai dengan notebook
function loadStatusChart() {
    const ctx = document.getElementById('statusChart').getContext('2d');
    const statusCounts = sampleData.reduce((acc, item) => {
        acc[item.statusGizi] = (acc[item.statusGizi] || 0) + 1;
        return acc;
    }, {});

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Normal', 'Stunted', 'Severely Stunted', 'Tinggi'],
            datasets: [{
                data: [
                    statusCounts['normal'] || 0,
                    statusCounts['stunted'] || 0,
                    statusCounts['severely stunted'] || 0,
                    statusCounts['tinggi'] || 0
                ],
                backgroundColor: ['#48bb78', '#f6ad55', '#fc8181', '#68d391'],
                borderColor: ['#38a169', '#ed8936', '#e53e3e', '#48bb78'],
                borderWidth: 3
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { padding: 20 }
                }
            }
        }
    });
}

// Correlation Heatmap - sesuai dengan notebook
function createCorrelationHeatmap() {
    const heatmapData = [
        { label: '', value: '', isHeader: true },
        { label: 'Umur', value: '', isHeader: true },
        { label: 'Tinggi Badan', value: '', isHeader: true },
        { label: 'Status Gizi', value: '', isHeader: true },
        
        { label: 'Umur', value: '', isHeader: true },
        { label: '1.00', value: 1.00 },
        { label: '0.87', value: 0.87 },
        { label: '0.23', value: 0.23 },
        
        { label: 'Tinggi Badan', value: '', isHeader: true },
        { label: '0.87', value: 0.87 },
        { label: '1.00', value: 1.00 },
        { label: '0.31', value: 0.31 },
        
        { label: 'Status Gizi', value: '', isHeader: true },
        { label: '0.23', value: 0.23 },
        { label: '0.31', value: 0.31 },
        { label: '1.00', value: 1.00 }
    ];

    const heatmapContainer = document.getElementById('correlationHeatmap');
    heatmapContainer.innerHTML = '';

    heatmapData.forEach(cell => {
        const cellDiv = document.createElement('div');
        cellDiv.className = 'heatmap-cell';
        cellDiv.textContent = cell.label;
        
        if (cell.isHeader) {
            cellDiv.classList.add('heatmap-header');
        } else if (cell.value !== undefined) {
            const intensity = Math.abs(cell.value);
            const color = cell.value > 0 ? 
                `rgba(66, 153, 225, ${intensity})` : 
                `rgba(245, 101, 101, ${intensity})`;
            cellDiv.style.backgroundColor = color;
        }
        
        heatmapContainer.appendChild(cellDiv);
    });
}

// Scatter Plot - sesuai dengan notebook
function createScatterPlot() {
    const svg = d3.select('#scatterPlot');
    svg.selectAll('*').remove();

    const margin = {top: 20, right: 80, bottom: 60, left: 80};
    const width = parseInt(svg.style('width')) - margin.left - margin.right;
    const height = 500 - margin.top - margin.bottom;

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear()
        .domain([0, d3.max(sampleData, d => d.umur)])
        .range([0, width]);

    const yScale = d3.scaleLinear()
        .domain([30, d3.max(sampleData, d => d.tinggiBadan)])
        .range([height, 0]);

    const colorScale = d => d.jenisKelamin === 'laki-laki' ? '#ff6b6b' : '#4ecdc4';

    // Add axes
    g.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(xScale))
        .append('text')
        .attr('x', width / 2)
        .attr('y', 40)
        .attr('fill', '#333')
        .style('text-anchor', 'middle')
        .style('font-weight', 'bold')
        .text('Umur (bulan)');

    g.append('g')
        .call(d3.axisLeft(yScale))
        .append('text')
        .attr('transform', 'rotate(-90)')
        .attr('y', -50)
        .attr('x', -height / 2)
        .attr('fill', '#333')
        .style('text-anchor', 'middle')
        .style('font-weight', 'bold')
        .text('Tinggi Badan (cm)');

    // Add points
    g.selectAll('.dot')
        .data(sampleData)
        .enter().append('circle')
        .attr('class', 'dot')
        .attr('cx', d => xScale(d.umur))
        .attr('cy', d => yScale(d.tinggiBadan))
        .attr('r', 4)
        .style('fill', colorScale)
        .style('opacity', 0.7)
        .on('mouseover', function(event, d) {
            d3.select(this).attr('r', 8).style('opacity', 1);
            
            // Tooltip
            const tooltip = d3.select('body').append('div')
                .attr('class', 'tooltip')
                .style('position', 'absolute')
                .style('background', 'rgba(0,0,0,0.8)')
                .style('color', 'white')
                .style('padding', '10px')
                .style('border-radius', '5px')
                .style('pointer-events', 'none')
                .style('z-index', '1000')
                .html(`
                    <strong>Umur:</strong> ${d.umur} bulan<br>
                    <strong>Tinggi:</strong> ${d.tinggiBadan.toFixed(1)} cm<br>
                    <strong>Jenis Kelamin:</strong> ${d.jenisKelamin}<br>
                    <strong>Status Gizi:</strong> ${d.statusGizi}
                `)
                .style('left', (event.pageX + 10) + 'px')
                .style('top', (event.pageY - 10) + 'px');
        })
        .on('mouseout', function() {
            d3.select(this).attr('r', 4).style('opacity', 0.7);
            d3.selectAll('.tooltip').remove();
        });
}

// Interactive Scatter Plot dengan filter
function updateScatterPlot() {
    const ageFilter = document.getElementById('ageFilter')?.value || 60;
    const genderFilter = document.getElementById('genderFilter')?.value || 'all';
    const statusFilter = document.getElementById('statusFilter')?.value || 'all';
    
    const ageValueEl = document.getElementById('ageValue');
    if (ageValueEl) {
        ageValueEl.textContent = `0 - ${ageFilter} bulan`;
    }

    const filteredData = sampleData.filter(d => {
        return d.umur <= ageFilter &&
               (genderFilter === 'all' || d.jenisKelamin === genderFilter) &&
               (statusFilter === 'all' || d.statusGizi === statusFilter);
    });

    const svg = d3.select('#filteredScatterPlot');
    svg.selectAll('*').remove();

    const margin = {top: 20, right: 80, bottom: 60, left: 80};
    const width = parseInt(svg.style('width')) - margin.left - margin.right;
    const height = 500 - margin.top - margin.bottom;

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear()
        .domain([0, d3.max(filteredData, d => d.umur) || 60])
        .range([0, width]);

    const yScale = d3.scaleLinear()
        .domain([30, d3.max(filteredData, d => d.tinggiBadan) || 120])
        .range([height, 0]);

    const colorScale = d => d.jenisKelamin === 'laki-laki' ? '#ff6b6b' : '#4ecdc4';

    // Add axes
    g.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(xScale))
        .append('text')
        .attr('x', width / 2)
        .attr('y', 40)
        .attr('fill', '#333')
        .style('text-anchor', 'middle')
        .style('font-weight', 'bold')
        .text('Umur (bulan)');

    g.append('g')
        .call(d3.axisLeft(yScale))
        .append('text')
        .attr('transform', 'rotate(-90)')
        .attr('y', -50)
        .attr('x', -height / 2)
        .attr('fill', '#333')
        .style('text-anchor', 'middle')
        .style('font-weight', 'bold')
        .text('Tinggi Badan (cm)');

    // Add points
    g.selectAll('.dot')
        .data(filteredData)
        .enter().append('circle')
        .attr('class', 'dot')
        .attr('cx', d => xScale(d.umur))
        .attr('cy', d => yScale(d.tinggiBadan))
        .attr('r', 4)
        .style('fill', colorScale)
        .style('opacity', 0.7)
        .on('mouseover', function(event, d) {
            d3.select(this).attr('r', 8).style('opacity', 1);
        })
        .on('mouseout', function() {
            d3.select(this).attr('r', 4).style('opacity', 0.7);
        });
}

// Cluster Plot - sesuai dengan notebook
function createClusterPlot() {
    const svg = d3.select('#clusterPlot');
    svg.selectAll('*').remove();

    const margin = {top: 20, right: 80, bottom: 60, left: 80};
    const width = parseInt(svg.style('width')) - margin.left - margin.right;
    const height = 500 - margin.top - margin.bottom;

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`);

    const xScale = d3.scaleLinear()
        .domain([0, d3.max(sampleData, d => d.umur)])
        .range([0, width]);

    const yScale = d3.scaleLinear()
        .domain([30, d3.max(sampleData, d => d.tinggiBadan)])
        .range([height, 0]);

    const clusterColors = ['#ff9999', '#66b3ff', '#99ff99'];

    // Add axes
    g.append('g')
        .attr('transform', `translate(0,${height})`)
        .call(d3.axisBottom(xScale))
        .append('text')
        .attr('x', width / 2)
        .attr('y', 40)
        .attr('fill', '#333')
        .style('text-anchor', 'middle')
        .style('font-weight', 'bold')
        .text('Umur (bulan)');

    g.append('g')
        .call(d3.axisLeft(yScale))
        .append('text')
        .attr('transform', 'rotate(-90)')
        .attr('y', -50)
        .attr('x', -height / 2)
        .attr('fill', '#333')
        .style('text-anchor', 'middle')
        .style('font-weight', 'bold')
        .text('Tinggi Badan (cm)');

    // Add cluster points
    sampleData.forEach(d => {
        if (d.jenisKelamin === 'laki-laki') {
            g.append('circle')
                .attr('cx', xScale(d.umur))
                .attr('cy', yScale(d.tinggiBadan))
                .attr('r', 4)
                .style('fill', clusterColors[d.cluster])
                .style('opacity', 0.7);
        } else {
            // Draw cross for women - sesuai dengan notebook
            const size = 4;
            g.append('g')
                .attr('transform', `translate(${xScale(d.umur)},${yScale(d.tinggiBadan)})`)
                .selectAll('.cross')
                .data([1])
                .enter()
                .append('g')
                .html(`
                    <line x1="-${size}" y1="0" x2="${size}" y2="0" stroke="${clusterColors[d.cluster]}" stroke-width="2" opacity="0.7"/>
                    <line x1="0" y1="-${size}" x2="0" y2="${size}" stroke="${clusterColors[d.cluster]}" stroke-width="2" opacity="0.7"/>
                `);
        }
    });
}

// Update age filter display
document.addEventListener('DOMContentLoaded', function() {
    const ageFilterEl = document.getElementById('ageFilter');
    if (ageFilterEl) {
        ageFilterEl.addEventListener('input', function() {
            const ageValueEl = document.getElementById('ageValue');
            if (ageValueEl) {
                ageValueEl.textContent = `0 - ${this.value} bulan`;
            }
        });
    }
});

// Additional utility functions for future enhancement
function exportData() {
    // Function to export dashboard data
    const data = {
        timestamp: new Date().toISOString(),
        total_data: sampleData.length,
        gender_distribution: sampleData.reduce((acc, item) => {
            acc[item.jenisKelamin] = (acc[item.jenisKelamin] || 0) + 1;
            return acc;
        }, {}),
        status_distribution: sampleData.reduce((acc, item) => {
            acc[item.statusGizi] = (acc[item.statusGizi] || 0) + 1;
            return acc;
        }, {})
    };
    
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `stunting-analysis-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Console log for debugging
console.log('🚀 Dashboard Stunting Analytics loaded successfully!');
console.log('📊 Sample data generated:', sampleData.length, 'records');
console.log('📈 Ready for data visualization and analysis');