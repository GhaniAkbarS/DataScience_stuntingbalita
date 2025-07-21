# Extended app.py dengan Chart Generation
from flask import Flask, render_template, request, send_from_directory, jsonify, send_file
import os
import sys
import io
import base64
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

print("🔄 Starting Dashboard Stunting App...")

# Import libraries dengan error handling
try:
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    print("✅ Data Science libraries imported successfully")
except ImportError as e:
    print(f"❌ Install required libraries: pip install numpy pandas scikit-learn matplotlib seaborn")
    print(f"Error: {e}")

try:
    import joblib
    print("✅ Joblib imported successfully")  
except ImportError:
    print("❌ Install Joblib: pip install joblib")

try:
    from tensorflow.keras.models import load_model
    print("✅ TensorFlow imported successfully")
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("⚠️ TensorFlow not available - AI predictions disabled")
    TENSORFLOW_AVAILABLE = False

# Create Flask app
app = Flask(__name__, template_folder='.')

# Global variables
model = None
scaler = None
df_global = None  # Store dataset globally for chart generation

# Load AI models (same as before)
def load_ai_models():
    global model, scaler
    if not TENSORFLOW_AVAILABLE:
        print("❌ TensorFlow not available, skipping model loading")
        return False
        
    try:
        print("🤖 Loading AI models...")
        
        if os.path.exists("model_status_gizi (1).h5"):
            model = load_model("model_status_gizi (1).h5")
            print("✅ Model loaded successfully")
        else:
            print("❌ Model file not found: model_status_gizi (1).h5")
            return False
            
        if os.path.exists("scaler_status_gizi.pkl"):
            scaler = joblib.load("scaler_status_gizi.pkl")
            print("✅ Scaler loaded successfully")
        else:
            print("❌ Scaler file not found: scaler_status_gizi.pkl")
            return False
            
        return True
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False

# Load dataset for charts
def load_dataset():
    global df_global
    try:
        if os.path.exists("data_balita.csv"):
            df_global = pd.read_csv("data_balita.csv")
            print(f"✅ Dataset loaded: {df_global.shape}")
            return True
        else:
            print("⚠️ data_balita.csv not found, generating sample data...")
            # Generate sample data
            np.random.seed(42)
            n_samples = 1000
            df_global = pd.DataFrame({
                'Umur (bulan)': np.random.randint(0, 60, n_samples),
                'Tinggi Badan (cm)': np.random.normal(80, 15, n_samples),
                'Jenis Kelamin': np.random.choice(['Laki-laki', 'Perempuan'], n_samples),
                'Status Gizi': np.random.choice(['Normal', 'Stunting', 'Gizi Buruk'], n_samples, p=[0.6, 0.3, 0.1])
            })
            df_global['Tinggi Badan (cm)'] = np.clip(df_global['Tinggi Badan (cm)'], 45, 120)
            print("✅ Sample dataset generated")
            return True
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return False

# Helper function to convert matplotlib plot to base64 string
def plot_to_base64():
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', dpi=150)
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    return plot_url

# Static file routes (same as before)
@app.route('/css/<path:filename>')
def css_files(filename):
    return send_from_directory('css', filename)

@app.route('/js/<path:filename>')
def js_files(filename):
    return send_from_directory('js', filename)

@app.route('/script.js')
def script_js():
    return send_from_directory('.', 'script.js')

@app.route('/styles.css')
def styles_css():
    return send_from_directory('.', 'styles.css')

@app.route('/')
def home():
    print("🏠 Home page accessed")
    possible_files = ['dashboard.html', 'index.html', 'main.html', 'stunting.html']
    
    for filename in possible_files:
        if os.path.exists(filename):
            print(f"✅ Found HTML file: {filename}")
            return render_template(filename)
    
    return """
    <div style="padding: 20px; font-family: Arial; max-width: 600px; margin: 50px auto; text-align: center;">
        <h1 style="color: #dc3545;">📄 File HTML Tidak Ditemukan</h1>
        <p>Berdasarkan struktur file Anda, pastikan file berikut ada:</p>
        <ul style="text-align: left; background: #f8f9fa; padding: 20px; border-radius: 8px;">
            <li><strong>dashboard.html</strong> ✅ (detected in your structure)</li>
            <li>index.html</li>
        </ul>
    </div>
    """

# =============================================================================
# CHART GENERATION ENDPOINTS
# =============================================================================

@app.route('/api/charts/clustering')
def generate_clustering_chart():
    """Generate clustering analysis chart (Elbow + Silhouette)"""
    try:
        if df_global is None:
            return jsonify({'error': 'Dataset not loaded'}), 500
        
        # Prepare data for clustering
        df_clean = df_global.dropna()
        df_clean = df_clean[(df_clean['Umur (bulan)'] >= 0) & (df_clean['Tinggi Badan (cm)'] > 30)]
        
        # Encode categorical variables
        df_encoded = df_clean.copy()
        df_encoded['Jenis Kelamin'] = df_encoded['Jenis Kelamin'].map({'Laki-laki': 0, 'Perempuan': 1})
        
        # Select features and scale
        X = df_encoded[['Umur (bulan)', 'Tinggi Badan (cm)', 'Jenis Kelamin']].values
        scaler_local = StandardScaler()
        X_scaled = scaler_local.fit_transform(X)
        
        # Clustering analysis
        range_k = range(2, 8)
        inertia = []
        sil_scores = []
        
        for k in range_k:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_scaled)
            inertia.append(kmeans.inertia_)
            sil_scores.append(silhouette_score(X_scaled, labels))
        
        # Create plot
        plt.figure(figsize=(12, 5))
        
        # Elbow Method
        plt.subplot(1, 2, 1)
        plt.plot(range_k, inertia, 'o-', linewidth=2, markersize=8)
        plt.xlabel("Jumlah Cluster (k)")
        plt.ylabel("Inertia")
        plt.title("Elbow Method")
        plt.grid(True, alpha=0.3)
        
        # Silhouette Score
        plt.subplot(1, 2, 2)
        plt.plot(range_k, sil_scores, 'o-', color='orange', linewidth=2, markersize=8)
        plt.xlabel("Jumlah Cluster (k)")
        plt.ylabel("Silhouette Score")
        plt.title("Silhouette Score vs k")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to base64
        chart_url = plot_to_base64()
        
        # Find optimal k
        optimal_k = range_k[sil_scores.index(max(sil_scores))]
        
        return jsonify({
            'success': True,
            'chart': chart_url,
            'optimal_k': optimal_k,
            'max_silhouette': max(sil_scores),
            'analysis': {
                'k_values': list(range_k),
                'inertia': inertia,
                'silhouette_scores': sil_scores
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/distribution')
def generate_distribution_chart():
    """Generate data distribution charts"""
    try:
        if df_global is None:
            return jsonify({'error': 'Dataset not loaded'}), 500
        
        # Create distribution plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Age distribution
        axes[0, 0].hist(df_global['Umur (bulan)'].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='navy')
        axes[0, 0].set_xlabel('Umur (bulan)')
        axes[0, 0].set_ylabel('Frekuensi')
        axes[0, 0].set_title('Distribusi Umur Balita')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Height distribution
        axes[0, 1].hist(df_global['Tinggi Badan (cm)'].dropna(), bins=20, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        axes[0, 1].set_xlabel('Tinggi Badan (cm)')
        axes[0, 1].set_ylabel('Frekuensi')
        axes[0, 1].set_title('Distribusi Tinggi Badan')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gender distribution
        gender_counts = df_global['Jenis Kelamin'].value_counts()
        axes[1, 0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('Distribusi Jenis Kelamin')
        
        # Status Gizi distribution
        status_counts = df_global['Status Gizi'].value_counts()
        axes[1, 1].bar(status_counts.index, status_counts.values, color=['green', 'orange', 'red'], alpha=0.7)
        axes[1, 1].set_xlabel('Status Gizi')
        axes[1, 1].set_ylabel('Jumlah')
        axes[1, 1].set_title('Distribusi Status Gizi')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        chart_url = plot_to_base64()
        
        return jsonify({
            'success': True,
            'chart': chart_url,
            'statistics': {
                'total_samples': len(df_global),
                'age_mean': df_global['Umur (bulan)'].mean(),
                'height_mean': df_global['Tinggi Badan (cm)'].mean(),
                'gender_distribution': gender_counts.to_dict(),
                'status_distribution': status_counts.to_dict()
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/charts/correlation')
def generate_correlation_chart():
    """Generate correlation heatmap"""
    try:
        if df_global is None:
            return jsonify({'error': 'Dataset not loaded'}), 500
        
        # Prepare numeric data
        df_numeric = df_global.copy()
        df_numeric['Jenis Kelamin'] = df_numeric['Jenis Kelamin'].map({'Laki-laki': 0, 'Perempuan': 1})
        df_numeric['Status Gizi'] = df_numeric['Status Gizi'].map({'Normal': 0, 'Stunting': 1, 'Gizi Buruk': 2})
        
        # Select numeric columns
        numeric_cols = ['Umur (bulan)', 'Tinggi Badan (cm)', 'Jenis Kelamin', 'Status Gizi']
        correlation_matrix = df_numeric[numeric_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
        plt.title('Correlation Matrix - Stunting Dataset')
        plt.tight_layout()
        
        chart_url = plot_to_base64()
        
        return jsonify({
            'success': True,
            'chart': chart_url,
            'correlation_data': correlation_matrix.to_dict()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# AI Prediction API (same as before)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    # Same implementation as before
    if model is None or scaler is None:
        return jsonify({
            'success': False,
            'error': 'Model tidak tersedia',
            'message': 'AI model belum dimuat. Menggunakan perhitungan WHO Z-score.'
        }), 503
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
        
        umur = float(data.get('age', 0))
        tinggi = float(data.get('height', 0))
        gender_str = data.get('gender', '')
        
        # Validasi data
        if not (0 <= umur <= 59):
            return jsonify({'success': False, 'error': f'Umur harus 0-59 bulan. Input: {umur}'}), 400
        
        if not (30 <= tinggi <= 120):
            return jsonify({'success': False, 'error': f'Tinggi harus 30-120 cm. Input: {tinggi}'}), 400
        
        if gender_str not in ['male', 'female', 'Laki-laki', 'Perempuan']:
            return jsonify({'success': False, 'error': 'Gender tidak valid'}), 400
        
        # Convert gender
        gender = 0 if gender_str in ['male', 'Laki-laki'] else 1
        
        # Predict
        input_data = np.array([[umur, tinggi, gender]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = float(np.max(prediction)) * 100
        
        # Map results
        label_map = {0: 'Normal', 1: 'Stunting', 2: 'Gizi Buruk'}
        result = label_map.get(predicted_class, 'Unknown')
        
        return jsonify({
            'success': True,
            'result': result,
            'confidence': round(confidence, 1),
            'predicted_class': int(predicted_class),
            'input_data': {'age': umur, 'height': tinggi, 'gender': gender_str}
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Health check endpoint
@app.route('/health')
def health():
    model_status = "✅ Ready" if model is not None else "❌ Not loaded"
    scaler_status = "✅ Ready" if scaler is not None else "❌ Not loaded"
    dataset_status = "✅ Loaded" if df_global is not None else "❌ Not loaded"
    
    return jsonify({
        'status': 'running',
        'model': model_status,
        'scaler': scaler_status,
        'dataset': dataset_status,
        'charts_available': True,
        'timestamp': str(np.datetime64('now'))
    })

if __name__ == '__main__':
    print("🚀 Initializing Flask application...")
    
    # Load dataset
    load_dataset()
    
    # Load AI models if available
    model_exists = os.path.exists("model_status_gizi (1).h5")
    scaler_exists = os.path.exists("scaler_status_gizi.pkl")
    
    if model_exists and scaler_exists:
        load_ai_models()
    
    print("\n🌐 Starting Flask server...")
    print("📍 Dashboard: http://127.0.0.1:5000")
    print("🔧 API Endpoints:")
    print("   - Prediction: /api/predict")
    print("   - Clustering Chart: /api/charts/clustering")
    print("   - Distribution Chart: /api/charts/distribution")
    print("   - Correlation Chart: /api/charts/correlation")
    print("   - Health Check: /health")
    print("="*50)
    
    app.run(debug=True, host='127.0.0.1', port=5000)