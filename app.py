#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dashboard Stunting - Enhanced Flask Application
===============================================

Sistem monitoring dan prediksi status stunting balita berbasis AI
dengan visualisasi data science yang komprehensif.

Features:
- AI Prediction using TensorFlow/Keras
- WHO Z-score fallback system
- Advanced data visualization
- K-Means clustering analysis
- Correlation and distribution analysis
- RESTful API endpoints
- Comprehensive error handling

Author: Dashboard Stunting Team
Version: 2.0.0
"""

import os
import sys
import io
import base64
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dashboard_stunting.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

print("🔄 Starting Dashboard Stunting Application...")
logger.info("Initializing Dashboard Stunting Application")

# Set matplotlib backend before importing
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server deployment
import matplotlib.pyplot as plt
import seaborn as sns

# Flask imports
from flask import Flask, render_template, request, send_from_directory, jsonify, send_file
from flask import abort, make_response
from werkzeug.exceptions import BadRequest, InternalServerError

# Data science imports with error handling
try:
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, silhouette_samples
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    print("✅ Data Science libraries imported successfully")
    logger.info("Data science libraries loaded successfully")
    DATA_SCIENCE_AVAILABLE = True
except ImportError as e:
    print(f"❌ Install required libraries: pip install numpy pandas scikit-learn matplotlib seaborn")
    print(f"Error: {e}")
    logger.error(f"Failed to import data science libraries: {e}")
    DATA_SCIENCE_AVAILABLE = False

# ML model imports
try:
    import joblib
    print("✅ Joblib imported successfully")
    JOBLIB_AVAILABLE = True
except ImportError:
    print("❌ Install Joblib: pip install joblib")
    logger.warning("Joblib not available")
    JOBLIB_AVAILABLE = False

try:
    from tensorflow.keras.models import load_model
    print("✅ TensorFlow imported successfully")
    logger.info("TensorFlow available for AI predictions")
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available - AI predictions disabled")
    logger.warning("TensorFlow not available, falling back to WHO standards")
    TENSORFLOW_AVAILABLE = False

# ===============================================
# APPLICATION CONFIGURATION
# ===============================================

class Config:
    """Application configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-dashboard-stunting-2025'
    DEBUG = os.environ.get('FLASK_ENV') == 'development'
    HOST = os.environ.get('FLASK_HOST') or '127.0.0.1'
    PORT = int(os.environ.get('FLASK_PORT') or 5000)
    
    # File paths
    MODEL_PATH = "model_status_gizi (1).h5"
    SCALER_PATH = "scaler_status_gizi.pkl"
    DATASET_PATH = "data_balita.csv"
    
    # Chart settings
    CHART_DPI = 150
    CHART_STYLE = 'seaborn-v0_8'
    CHART_COLOR_PALETTE = 'husl'
    
    # API settings
    MAX_PREDICTIONS_PER_MINUTE = 60
    CACHE_TIMEOUT = 300  # 5 minutes

# Create Flask app
app = Flask(__name__, template_folder='.')
app.config.from_object(Config)

# ===============================================
# GLOBAL VARIABLES
# ===============================================

# AI Models
model = None
scaler = None

# Dataset
df_global = None
dataset_stats = {}

# Application state
app_state = {
    'model_loaded': False,
    'scaler_loaded': False,
    'dataset_loaded': False,
    'charts_generated': 0,
    'predictions_made': 0,
    'last_health_check': None
}

# ===============================================
# UTILITY FUNCTIONS
# ===============================================

def safe_float(value, default=0.0):
    """Safely convert value to float"""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert value to integer"""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default

def validate_prediction_input(age, gender, height):
    """Validate prediction input parameters"""
    errors = []
    
    if not isinstance(age, (int, float)) or age < 0 or age > 60:
        errors.append("Usia harus antara 0-60 bulan")
    
    if gender not in ['male', 'female', 'Laki-laki', 'Perempuan']:
        errors.append("Jenis kelamin tidak valid")
    
    if not isinstance(height, (int, float)) or height < 30 or height > 120:
        errors.append("Tinggi badan harus antara 30-120 cm")
    
    return errors

def plot_to_base64():
    """Convert matplotlib plot to base64 string"""
    try:
        img = io.BytesIO()
        plt.savefig(img, format='png', bbox_inches='tight', 
                   dpi=Config.CHART_DPI, facecolor='white', edgecolor='none')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()
        return plot_url
    except Exception as e:
        logger.error(f"Error converting plot to base64: {e}")
        plt.close()
        raise

def setup_plot_style():
    """Setup consistent plot styling"""
    try:
        plt.style.use('default')
        sns.set_palette(Config.CHART_COLOR_PALETTE)
        
        # Set default font sizes
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 12,
            'axes.labelsize': 10,
            'xtick.labelsize': 9,
            'ytick.labelsize': 9,
            'legend.fontsize': 9,
            'figure.titlesize': 14,
            'font.family': 'sans-serif'
        })
    except Exception as e:
        logger.warning(f"Could not setup plot style: {e}")

# ===============================================
# MODEL LOADING FUNCTIONS
# ===============================================

def load_ai_models():
    """Load AI models with comprehensive error handling"""
    global model, scaler
    
    if not TENSORFLOW_AVAILABLE or not JOBLIB_AVAILABLE:
        logger.warning("Required libraries not available for AI model loading")
        return False
        
    try:
        logger.info("🤖 Loading AI models...")
        
        # Load TensorFlow model
        if os.path.exists(Config.MODEL_PATH):
            model = load_model(Config.MODEL_PATH)
            app_state['model_loaded'] = True
            logger.info(f"✅ Model loaded successfully from {Config.MODEL_PATH}")
        else:
            logger.error(f"❌ Model file not found: {Config.MODEL_PATH}")
            return False
            
        # Load scaler
        if os.path.exists(Config.SCALER_PATH):
            scaler = joblib.load(Config.SCALER_PATH)
            app_state['scaler_loaded'] = True
            logger.info(f"✅ Scaler loaded successfully from {Config.SCALER_PATH}")
        else:
            logger.error(f"❌ Scaler file not found: {Config.SCALER_PATH}")
            return False
            
        # Validate model
        if model is not None and scaler is not None:
            test_input = np.array([[24, 85, 0]])  # Test data
            test_scaled = scaler.transform(test_input)
            test_pred = model.predict(test_scaled, verbose=0)
            logger.info(f"✅ Model validation successful. Test prediction shape: {test_pred.shape}")
            return True
        else:
            logger.error("❌ Model or scaler is None after loading")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error loading AI models: {e}")
        model = None
        scaler = None
        app_state['model_loaded'] = False
        app_state['scaler_loaded'] = False
        return False

def load_dataset():
    """Load and validate dataset"""
    global df_global, dataset_stats
    
    if not DATA_SCIENCE_AVAILABLE:
        logger.warning("Data science libraries not available")
        return False
    
    try:
        if os.path.exists(Config.DATASET_PATH):
            df_global = pd.read_csv(Config.DATASET_PATH)
            logger.info(f"✅ Dataset loaded from {Config.DATASET_PATH}: {df_global.shape}")
        else:
            logger.warning(f"⚠️ {Config.DATASET_PATH} not found, generating sample data...")
            df_global = generate_sample_dataset()
            
        # Validate and clean dataset
        df_global = clean_dataset(df_global)
        
        # Calculate dataset statistics
        dataset_stats = calculate_dataset_stats(df_global)
        app_state['dataset_loaded'] = True
        
        logger.info(f"✅ Dataset processed successfully: {len(df_global)} records")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error loading dataset: {e}")
        return False

def generate_sample_dataset(n_samples=1000):
    """Generate sample dataset for demonstration"""
    logger.info(f"Generating sample dataset with {n_samples} records")
    
    np.random.seed(42)
    
    # Generate realistic data
    ages = np.random.randint(0, 60, n_samples)
    
    # Height based on age with some variation
    base_heights = 50 + (ages * 1.2) + np.random.normal(0, 5, n_samples)
    heights = np.clip(base_heights, 45, 120)
    
    genders = np.random.choice(['Laki-laki', 'Perempuan'], n_samples)
    
    # Status based on height z-score simulation
    status_prob = np.random.random(n_samples)
    status = np.where(status_prob < 0.15, 'Gizi Buruk',
                     np.where(status_prob < 0.35, 'Stunting', 'Normal'))
    
    df = pd.DataFrame({
        'Umur (bulan)': ages,
        'Tinggi Badan (cm)': heights,
        'Jenis Kelamin': genders,
        'Status Gizi': status
    })
    
    return df

def clean_dataset(df):
    """Clean and validate dataset"""
    logger.info("Cleaning dataset...")
    
    # Remove invalid records
    initial_count = len(df)
    
    # Clean age data
    df = df[(df['Umur (bulan)'] >= 0) & (df['Umur (bulan)'] <= 60)]
    
    # Clean height data
    df = df[(df['Tinggi Badan (cm)'] >= 30) & (df['Tinggi Badan (cm)'] <= 120)]
    
    # Clean gender data
    valid_genders = ['Laki-laki', 'Perempuan', 'Male', 'Female', 'male', 'female']
    df = df[df['Jenis Kelamin'].isin(valid_genders)]
    
    # Standardize gender values
    df['Jenis Kelamin'] = df['Jenis Kelamin'].replace({
        'Male': 'Laki-laki', 'male': 'Laki-laki',
        'Female': 'Perempuan', 'female': 'Perempuan'
    })
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    final_count = len(df)
    logger.info(f"Dataset cleaned: {initial_count} -> {final_count} records")
    
    return df

def calculate_dataset_stats(df):
    """Calculate comprehensive dataset statistics"""
    try:
        stats = {
            'total_samples': len(df),
            'age_mean': df['Umur (bulan)'].mean(),
            'age_std': df['Umur (bulan)'].std(),
            'age_min': df['Umur (bulan)'].min(),
            'age_max': df['Umur (bulan)'].max(),
            'height_mean': df['Tinggi Badan (cm)'].mean(),
            'height_std': df['Tinggi Badan (cm)'].std(),
            'height_min': df['Tinggi Badan (cm)'].min(),
            'height_max': df['Tinggi Badan (cm)'].max(),
            'gender_distribution': df['Jenis Kelamin'].value_counts().to_dict(),
            'status_distribution': df['Status Gizi'].value_counts().to_dict(),
            'last_updated': datetime.now().isoformat()
        }
        
        # Calculate prevalence
        total = stats['total_samples']
        stunting_count = stats['status_distribution'].get('Stunting', 0)
        gizi_buruk_count = stats['status_distribution'].get('Gizi Buruk', 0)
        
        stats['stunting_prevalence'] = ((stunting_count + gizi_buruk_count) / total * 100) if total > 0 else 0
        
        return stats
        
    except Exception as e:
        logger.error(f"Error calculating dataset stats: {e}")
        return {}

# ===============================================
# STATIC FILE ROUTES
# ===============================================

@app.route('/css/<path:filename>')
def css_files(filename):
    """Serve CSS files"""
    return send_from_directory('css', filename)

@app.route('/js/<path:filename>')
def js_files(filename):
    """Serve JavaScript files"""
    return send_from_directory('js', filename)

@app.route('/script.js')
def script_js():
    """Serve main JavaScript file"""
    return send_from_directory('.', 'script.js')

@app.route('/styles.css')
def styles_css():
    """Serve main CSS file"""
    return send_from_directory('.', 'styles.css')

@app.route('/favicon.ico')
def favicon():
    """Serve favicon"""
    return send_from_directory('.', 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# ===============================================
# MAIN ROUTES
# ===============================================

@app.route('/')
def home():
    """Main dashboard route"""
    logger.info("🏠 Home page accessed")
    
    # Look for HTML files in order of preference
    possible_files = ['dashboard.html', 'index.html', 'main.html', 'stunting.html']
    
    for filename in possible_files:
        if os.path.exists(filename):
            logger.info(f"✅ Serving HTML file: {filename}")
            try:
                return render_template(filename)
            except Exception as e:
                logger.error(f"Error rendering {filename}: {e}")
                continue
    
    # If no HTML file found, return error page
    logger.error("❌ No HTML template files found")
    return create_error_response(
        "Dashboard Files Not Found",
        "Could not locate dashboard HTML files. Please ensure dashboard.html exists.",
        500
    )

def create_error_response(title, message, status_code):
    """Create a formatted error response"""
    html_content = f"""
    <!DOCTYPE html>
    <html lang="id">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title} - Dashboard Stunting</title>
        <style>
            body {{
                font-family: 'Segoe UI', Arial, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            .error-container {{
                background: white;
                border-radius: 15px;
                padding: 40px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                text-align: center;
                max-width: 500px;
                width: 100%;
            }}
            .error-icon {{
                font-size: 4rem;
                margin-bottom: 20px;
            }}
            .error-title {{
                color: #dc3545;
                font-size: 1.5rem;
                font-weight: 600;
                margin-bottom: 15px;
            }}
            .error-message {{
                color: #666;
                line-height: 1.6;
                margin-bottom: 25px;
            }}
            .back-button {{
                background: linear-gradient(135deg, #667eea, #764ba2);
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 25px;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
                transition: transform 0.3s ease;
            }}
            .back-button:hover {{
                transform: translateY(-2px);
            }}
        </style>
    </head>
    <body>
        <div class="error-container">
            <div class="error-icon">❌</div>
            <h1 class="error-title">{title}</h1>
            <p class="error-message">{message}</p>
            <a href="/" class="back-button" onclick="location.reload()">🔄 Coba Lagi</a>
        </div>
    </body>
    </html>
    """
    
    response = make_response(html_content, status_code)
    response.headers['Content-Type'] = 'text/html; charset=utf-8'
    return response

# ===============================================
# API ENDPOINTS
# ===============================================

@app.route('/health')
def health_check():
    """Comprehensive health check endpoint"""
    try:
        app_state['last_health_check'] = datetime.now().isoformat()
        
        health_data = {
            'status': 'running',
            'timestamp': app_state['last_health_check'],
            'version': '2.0.0',
            'model': 'Ready' if app_state['model_loaded'] else 'Not loaded',
            'scaler': 'Ready' if app_state['scaler_loaded'] else 'Not loaded', 
            'dataset': 'Loaded' if app_state['dataset_loaded'] else 'Not loaded',
            'charts_available': DATA_SCIENCE_AVAILABLE,
            'ai_predictions': TENSORFLOW_AVAILABLE and app_state['model_loaded'],
            'statistics': dataset_stats if app_state['dataset_loaded'] else {},
            'performance': {
                'charts_generated': app_state['charts_generated'],
                'predictions_made': app_state['predictions_made']
            }
        }
        
        logger.info("Health check performed successfully")
        return jsonify(health_data)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """AI prediction endpoint with comprehensive error handling"""
    try:
        # Validate request
        if not request.is_json:
            raise BadRequest("Request must be JSON")
        
        data = request.get_json()
        if not data:
            raise BadRequest("No data provided")
        
        # Extract and validate parameters
        age = safe_float(data.get('age', 0))
        height = safe_float(data.get('height', 0))
        gender_str = data.get('gender', '').strip()
        
        # Validate inputs
        validation_errors = validate_prediction_input(age, gender_str, height)
        if validation_errors:
            return jsonify({
                'success': False,
                'error': 'Validation failed',
                'details': validation_errors
            }), 400
        
        # Check if AI model is available
        if not (app_state['model_loaded'] and app_state['scaler_loaded']):
            return jsonify({
                'success': False,
                'error': 'AI model not available',
                'message': 'Model belum dimuat. Gunakan perhitungan WHO Z-score.',
                'fallback_available': True
            }), 503
        
        # Convert gender to numeric
        gender = 0 if gender_str.lower() in ['male', 'laki-laki'] else 1
        
        # Make prediction
        input_data = np.array([[age, height, gender]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        
        # Map prediction to label
        label_map = {0: 'Normal', 1: 'Stunting', 2: 'Gizi Buruk'}
        result = label_map.get(predicted_class, 'Unknown')
        
        # Update statistics
        app_state['predictions_made'] += 1
        
        response_data = {
            'success': True,
            'result': result,
            'confidence': round(confidence, 1),
            'predicted_class': int(predicted_class),
            'prediction_probabilities': [float(p) for p in prediction[0]],
            'input_data': {
                'age': age,
                'height': height, 
                'gender': gender_str
            },
            'timestamp': datetime.now().isoformat(),
            'method': 'AI Neural Network'
        }
        
        logger.info(f"AI prediction successful: {result} (confidence: {confidence:.1f}%)")
        return jsonify(response_data)
        
    except BadRequest as e:
        logger.warning(f"Bad request in prediction: {e}")
        return jsonify({'success': False, 'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Terjadi kesalahan dalam prediksi. Silakan coba lagi.',
            'timestamp': datetime.now().isoformat()
        }), 500

# ===============================================
# CHART GENERATION ENDPOINTS
# ===============================================

@app.route('/api/charts/clustering')
def generate_clustering_chart():
    """Generate comprehensive clustering analysis"""
    if not DATA_SCIENCE_AVAILABLE or df_global is None:
        return jsonify({'error': 'Data science libraries or dataset not available'}), 500
    
    try:
        logger.info("Generating clustering analysis chart")
        setup_plot_style()
        
        # Prepare data
        df_clean = df_global.dropna()
        df_clean = df_clean[
            (df_clean['Umur (bulan)'] >= 0) & 
            (df_clean['Umur (bulan)'] <= 60) &
            (df_clean['Tinggi Badan (cm)'] >= 30) & 
            (df_clean['Tinggi Badan (cm)'] <= 120)
        ].copy()
        
        if len(df_clean) < 10:
            raise ValueError("Insufficient data for clustering analysis")
        
        # Encode categorical variables
        df_encoded = df_clean.copy()
        df_encoded['Jenis Kelamin'] = df_encoded['Jenis Kelamin'].map({
            'Laki-laki': 0, 'Perempuan': 1
        })
        
        # Prepare features
        features = ['Umur (bulan)', 'Tinggi Badan (cm)', 'Jenis Kelamin']
        X = df_encoded[features].values
        
        # Scale features
        scaler_local = StandardScaler()
        X_scaled = scaler_local.fit_transform(X)
        
        # Determine optimal number of clusters
        range_k = range(2, min(8, len(df_clean)//10))
        inertia = []
        sil_scores = []
        
        for k in range_k:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertia.append(kmeans.inertia_)
            sil_scores.append(silhouette_score(X_scaled, labels))
        
        # Find optimal k
        optimal_k = range_k[np.argmax(sil_scores)]
        max_silhouette = max(sil_scores)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Analisis Clustering K-Means - Dataset Stunting', fontsize=16, fontweight='bold')
        
        # 1. Elbow Method
        axes[0, 0].plot(range_k, inertia, 'o-', linewidth=2, markersize=8, color='#1f77b4')
        axes[0, 0].set_xlabel("Jumlah Cluster (k)")
        axes[0, 0].set_ylabel("Inertia (WCSS)")
        axes[0, 0].set_title("Elbow Method")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Optimal k={optimal_k}')
        axes[0, 0].legend()
        
        # 2. Silhouette Score
        axes[0, 1].plot(range_k, sil_scores, 'o-', color='orange', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel("Jumlah Cluster (k)")
        axes[0, 1].set_ylabel("Silhouette Score")
        axes[0, 1].set_title("Silhouette Score vs k")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axvline(x=optimal_k, color='red', linestyle='--', alpha=0.7, label=f'Max: {max_silhouette:.3f}')
        axes[0, 1].legend()
        
        # 3. Cluster visualization (PCA)
        if X_scaled.shape[1] > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            kmeans_optimal = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
            cluster_labels = kmeans_optimal.fit_predict(X_scaled)
            
            scatter = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, 
                                       cmap='viridis', alpha=0.6, s=50)
            axes[1, 0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
            axes[1, 0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
            axes[1, 0].set_title(f"Cluster Visualization (k={optimal_k})")
            plt.colorbar(scatter, ax=axes[1, 0])
        
        # 4. Cluster characteristics
        kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        final_labels = kmeans_final.fit_predict(X_scaled)
        
        # Calculate cluster centers in original scale
        centers_scaled = kmeans_final.cluster_centers_
        centers_original = scaler_local.inverse_transform(centers_scaled)
        
        cluster_data = []
        for i in range(optimal_k):
            cluster_mask = final_labels == i
            cluster_data.append([
                f"Cluster {i+1}",
                centers_original[i, 0],  # Age
                centers_original[i, 1],  # Height
                np.sum(cluster_mask)     # Count
            ])
        
        # Create table
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=cluster_data,
                                colLabels=['Cluster', 'Rata-rata Umur', 'Rata-rata Tinggi', 'Jumlah'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title("Karakteristik Cluster")
        
        plt.tight_layout()
        
        # Convert to base64
        chart_url = plot_to_base64()
        app_state['charts_generated'] += 1
        
        logger.info(f"Clustering chart generated successfully. Optimal k: {optimal_k}")
        
        return jsonify({
            'success': True,
            'chart': chart_url,
            'optimal_k': int(optimal_k),
            'max_silhouette': float(max_silhouette),
            'cluster_centers': centers_original.tolist(),
            'cluster_sizes': [int(np.sum(final_labels == i)) for i in range(optimal_k)],
            'analysis': {
                'k_values': list(range_k),
                'inertia': inertia,
                'silhouette_scores': sil_scores,
                'total_samples': len(df_clean),
                'features_used': features
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error generating clustering chart: {e}")
        return jsonify({'error': f'Failed to generate clustering chart: {str(e)}'}), 500

@app.route('/api/charts/distribution')
def generate_distribution_chart():
    """Generate comprehensive data distribution analysis"""
    if not DATA_SCIENCE_AVAILABLE or df_global is None:
        return jsonify({'error': 'Data science libraries or dataset not available'}), 500
    
    try:
        logger.info("Generating distribution analysis chart")
        setup_plot_style()
        
        # Create comprehensive distribution plots
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        fig.suptitle('Analisis Distribusi Data Stunting', fontsize=16, fontweight='bold')
        
        # 1. Age distribution with statistics
        ages = df_global['Umur (bulan)'].dropna()
        axes[0, 0].hist(ages, bins=20, alpha=0.7, color='skyblue', edgecolor='navy', density=True)
        axes[0, 0].axvline(ages.mean(), color='red', linestyle='--', label=f'Mean: {ages.mean():.1f}')
        axes[0, 0].axvline(ages.median(), color='green', linestyle='--', label=f'Median: {ages.median():.1f}')
        axes[0, 0].set_xlabel('Umur (bulan)')
        axes[0, 0].set_ylabel('Densitas')
        axes[0, 0].set_title('Distribusi Umur Balita')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Height distribution with statistics
        heights = df_global['Tinggi Badan (cm)'].dropna()
        axes[0, 1].hist(heights, bins=20, alpha=0.7, color='lightgreen', edgecolor='darkgreen', density=True)
        axes[0, 1].axvline(heights.mean(), color='red', linestyle='--', label=f'Mean: {heights.mean():.1f}')
        axes[0, 1].axvline(heights.median(), color='green', linestyle='--', label=f'Median: {heights.median():.1f}')
        axes[0, 1].set_xlabel('Tinggi Badan (cm)')
        axes[0, 1].set_ylabel('Densitas')
        axes[0, 1].set_title('Distribusi Tinggi Badan')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Age vs Height scatter
        for gender in df_global['Jenis Kelamin'].unique():
            if pd.notna(gender):
                mask = df_global['Jenis Kelamin'] == gender
                axes[0, 2].scatter(df_global[mask]['Umur (bulan)'], 
                                 df_global[mask]['Tinggi Badan (cm)'], 
                                 alpha=0.6, label=gender, s=30)
        axes[0, 2].set_xlabel('Umur (bulan)')
        axes[0, 2].set_ylabel('Tinggi Badan (cm)')
        axes[0, 2].set_title('Hubungan Umur vs Tinggi Badan')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Gender distribution
        gender_counts = df_global['Jenis Kelamin'].value_counts()
        colors = ['lightcoral', 'lightskyblue']
        wedges, texts, autotexts = axes[1, 0].pie(gender_counts.values, labels=gender_counts.index, 
                                                 autopct='%1.1f%%', startangle=90, colors=colors)
        axes[1, 0].set_title('Distribusi Jenis Kelamin')
        
        # 5. Status Gizi distribution
        status_counts = df_global['Status Gizi'].value_counts()
        colors_status = ['green', 'orange', 'red'][:len(status_counts)]
        bars = axes[1, 1].bar(status_counts.index, status_counts.values, 
                             color=colors_status, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Status Gizi')
        axes[1, 1].set_ylabel('Jumlah')
        axes[1, 1].set_title('Distribusi Status Gizi')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{int(height)}', ha='center', va='bottom')
        
        # 6. Box plot of height by status
        status_data = []
        status_labels = []
        for status in df_global['Status Gizi'].unique():
            if pd.notna(status):
                data = df_global[df_global['Status Gizi'] == status]['Tinggi Badan (cm)'].dropna()
                if len(data) > 0:
                    status_data.append(data)
                    status_labels.append(status)
        
        if status_data:
            box_plot = axes[1, 2].boxplot(status_data, labels=status_labels, patch_artist=True)
            for patch, color in zip(box_plot['boxes'], colors_status):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
        axes[1, 2].set_xlabel('Status Gizi')
        axes[1, 2].set_ylabel('Tinggi Badan (cm)')
        axes[1, 2].set_title('Distribusi Tinggi per Status Gizi')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Convert to base64
        chart_url = plot_to_base64()
        app_state['charts_generated'] += 1
        
        # Calculate comprehensive statistics
        statistics = {
            'total_samples': len(df_global),
            'age_stats': {
                'mean': float(ages.mean()),
                'median': float(ages.median()),
                'std': float(ages.std()),
                'min': float(ages.min()),
                'max': float(ages.max())
            },
            'height_stats': {
                'mean': float(heights.mean()),
                'median': float(heights.median()),
                'std': float(heights.std()),
                'min': float(heights.min()),
                'max': float(heights.max())
            },
            'gender_distribution': gender_counts.to_dict(),
            'status_distribution': status_counts.to_dict()
        }
        
        logger.info("Distribution chart generated successfully")
        
        return jsonify({
            'success': True,
            'chart': chart_url,
            'statistics': statistics,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error generating distribution chart: {e}")
        return jsonify({'error': f'Failed to generate distribution chart: {str(e)}'}), 500

@app.route('/api/charts/correlation')
def generate_correlation_chart():
    """Generate correlation analysis heatmap"""
    if not DATA_SCIENCE_AVAILABLE or df_global is None:
        return jsonify({'error': 'Data science libraries or dataset not available'}), 500
    
    try:
        logger.info("Generating correlation analysis chart")
        setup_plot_style()
        
        # Prepare numeric data
        df_numeric = df_global.copy()
        
        # Encode categorical variables
        df_numeric['Jenis Kelamin'] = df_numeric['Jenis Kelamin'].map({
            'Laki-laki': 0, 'Perempuan': 1
        })
        
        # Map status gizi to numeric values
        status_mapping = {
            'Normal': 0,
            'Stunting': 1, 
            'Gizi Buruk': 2
        }
        df_numeric['Status Gizi'] = df_numeric['Status Gizi'].map(status_mapping)
        
        # Select numeric columns
        numeric_cols = ['Umur (bulan)', 'Tinggi Badan (cm)', 'Jenis Kelamin', 'Status Gizi']
        df_corr = df_numeric[numeric_cols].dropna()
        
        if len(df_corr) < 2:
            raise ValueError("Insufficient data for correlation analysis")
        
        # Calculate correlation matrix
        correlation_matrix = df_corr.corr()
        
        # Create enhanced heatmap
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Analisis Korelasi Variabel Stunting', fontsize=16, fontweight='bold')
        
        # 1. Correlation heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, 
                   annot=True, 
                   cmap='RdBu_r', 
                   center=0,
                   square=True, 
                   linewidths=0.5, 
                   cbar_kws={'shrink': 0.8},
                   mask=mask,
                   ax=axes[0])
        axes[0].set_title('Matriks Korelasi')
        axes[0].set_xlabel('')
        axes[0].set_ylabel('')
        
        # 2. Correlation strengths bar plot
        # Get correlation values with target variable (Status Gizi)
        target_corr = correlation_matrix['Status Gizi'].drop('Status Gizi').abs().sort_values(ascending=True)
        
        colors = ['green' if x < 0.3 else 'orange' if x < 0.7 else 'red' for x in target_corr.values]
        bars = axes[1].barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
        axes[1].set_yticks(range(len(target_corr)))
        axes[1].set_yticklabels(target_corr.index)
        axes[1].set_xlabel('Kekuatan Korelasi (|r|)')
        axes[1].set_title('Korelasi dengan Status Gizi')
        axes[1].grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, target_corr.values)):
            axes[1].text(value + 0.01, i, f'{value:.3f}', 
                        va='center', ha='left', fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to base64
        chart_url = plot_to_base64()
        app_state['charts_generated'] += 1
        
        # Prepare detailed correlation analysis
        correlation_insights = {
            'strongest_positive': {},
            'strongest_negative': {},
            'target_correlations': target_corr.to_dict()
        }
        
        # Find strongest correlations (excluding diagonal)
        corr_values = correlation_matrix.values
        np.fill_diagonal(corr_values, 0)  # Remove diagonal
        
        # Get position of max correlation
        max_pos = np.unravel_index(np.argmax(corr_values), corr_values.shape)
        min_pos = np.unravel_index(np.argmin(corr_values), corr_values.shape)
        
        correlation_insights['strongest_positive'] = {
            'variables': [correlation_matrix.index[max_pos[0]], correlation_matrix.columns[max_pos[1]]],
            'value': float(corr_values[max_pos])
        }
        
        correlation_insights['strongest_negative'] = {
            'variables': [correlation_matrix.index[min_pos[0]], correlation_matrix.columns[min_pos[1]]],
            'value': float(corr_values[min_pos])
        }
        
        logger.info("Correlation chart generated successfully")
        
        return jsonify({
            'success': True,
            'chart': chart_url,
            'correlation_matrix': correlation_matrix.to_dict(),
            'insights': correlation_insights,
            'interpretation': {
                'weak': "< 0.3 (korelasi lemah)",
                'moderate': "0.3-0.7 (korelasi sedang)", 
                'strong': "> 0.7 (korelasi kuat)"
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error generating correlation chart: {e}")
        return jsonify({'error': f'Failed to generate correlation chart: {str(e)}'}), 500

# ===============================================
# ERROR HANDLERS
# ===============================================

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    logger.warning(f"404 error: {request.url}")
    return create_error_response(
        "Halaman Tidak Ditemukan",
        "Halaman yang Anda cari tidak tersedia.",
        404
    )

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"500 error: {error}")
    return create_error_response(
        "Kesalahan Server Internal",
        "Terjadi kesalahan pada server. Silakan coba lagi nanti.",
        500
    )

@app.errorhandler(BadRequest)
def bad_request_error(error):
    """Handle 400 errors"""
    logger.warning(f"400 error: {error}")
    return jsonify({
        'success': False,
        'error': 'Bad Request',
        'message': str(error),
        'timestamp': datetime.now().isoformat()
    }), 400

# ===============================================
# APPLICATION INITIALIZATION
# ===============================================

def initialize_application():
    """Initialize the application with all components"""
    logger.info("🚀 Initializing Dashboard Stunting Application...")
    
    # Load dataset
    logger.info("📊 Loading dataset...")
    dataset_loaded = load_dataset()
    
    # Load AI models if available
    logger.info("🤖 Checking AI models...")
    model_exists = os.path.exists(Config.MODEL_PATH)
    scaler_exists = os.path.exists(Config.SCALER_PATH)
    
    if model_exists and scaler_exists and TENSORFLOW_AVAILABLE:
        logger.info("🔄 Loading AI models...")
        models_loaded = load_ai_models()
        if models_loaded:
            logger.info("✅ AI models loaded successfully")
        else:
            logger.warning("⚠️ AI models failed to load, using WHO Z-score fallback")
    else:
        missing_files = []
        if not model_exists:
            missing_files.append(Config.MODEL_PATH)
        if not scaler_exists:
            missing_files.append(Config.SCALER_PATH)
        if not TENSORFLOW_AVAILABLE:
            missing_files.append("TensorFlow library")
        
        logger.warning(f"⚠️ AI components not available: {missing_files}")
        logger.info("📊 Using WHO Z-score standards as fallback")
    
    # Print initialization summary
    print("\n" + "="*60)
    print("🌐 DASHBOARD STUNTING - INITIALIZATION COMPLETE")
    print("="*60)
    print(f"📍 Server URL: http://{Config.HOST}:{Config.PORT}")
    print(f"🗄️ Dataset: {'✅ Loaded' if dataset_loaded else '❌ Failed'}")
    print(f"🤖 AI Model: {'✅ Ready' if app_state['model_loaded'] else '❌ Not available'}")
    print(f"📊 Charts: {'✅ Available' if DATA_SCIENCE_AVAILABLE else '❌ Not available'}")
    print(f"🔧 Health Check: http://{Config.HOST}:{Config.PORT}/health")
    print("\n🔧 API Endpoints:")
    print("   - Prediction: /api/predict")
    print("   - Clustering Chart: /api/charts/clustering")
    print("   - Distribution Chart: /api/charts/distribution")
    print("   - Correlation Chart: /api/charts/correlation")
    print("="*60)
    
    return True

# ===============================================
# MAIN APPLICATION ENTRY POINT
# ===============================================

if __name__ == '__main__':
    try:
        # Initialize application
        initialization_successful = initialize_application()
        
        if not initialization_successful:
            logger.error("❌ Application initialization failed")
            sys.exit(1)
        
        # Start Flask server
        logger.info(f"🌐 Starting Flask server on {Config.HOST}:{Config.PORT}")
        app.run(
            debug=Config.DEBUG,
            host=Config.HOST,
            port=Config.PORT,
            threaded=True
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
        print("\n👋 Dashboard Stunting stopped. Thank you!")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        print(f"\n💥 Fatal error occurred: {e}")
        sys.exit(1)