"""
Flask API for Cat and Dog Classification ML Pipeline
"""
import os
import json
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf
from src.preprocessing import prepare_dataset, preprocess_uploaded_image, load_images_from_directory
from src.model import create_model, train_model, evaluate_model, load_model, save_model_metadata, retrain_model
from src.prediction import CatDogPredictor
import threading
import time

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RETRAIN_FOLDER = 'retrain_data'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'models/cat_dog_model.h5'
METADATA_PATH = 'models/model_metadata.json'

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RETRAIN_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RETRAIN_FOLDER'] = RETRAIN_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables
predictor = None
model_status = {
    'status': 'ready',
    'uptime_start': datetime.now().isoformat(),
    'last_training': None,
    'total_predictions': 0
}
retraining_lock = threading.Lock()
retraining_status = {
    'in_progress': False,
    'progress': 0,
    'message': ''
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_model():
    """Initialize the predictor with the model"""
    global predictor
    try:
        if os.path.exists(MODEL_PATH):
            predictor = CatDogPredictor(MODEL_PATH)
            model_status['status'] = 'ready'
        else:
            model_status['status'] = 'not_trained'
            predictor = None
    except Exception as e:
        print(f"Error initializing model: {e}")
        model_status['status'] = 'error'
        predictor = None

# Initialize model on startup
initialize_model()

@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')

@app.route('/api/health')
def health():
    """Health check endpoint"""
    uptime = (datetime.now() - datetime.fromisoformat(model_status['uptime_start'])).total_seconds()
    return jsonify({
        'status': 'healthy',
        'model_status': model_status['status'],
        'uptime_seconds': uptime,
        'total_predictions': model_status['total_predictions']
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict endpoint for single image"""
    try:
        if predictor is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 503
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Make prediction
            result = predictor.predict(image_file=file)
            model_status['total_predictions'] += 1
            
            return jsonify(result), 200
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/status')
def model_status_endpoint():
    """Get model status and uptime"""
    uptime = (datetime.now() - datetime.fromisoformat(model_status['uptime_start'])).total_seconds()
    
    # Load metadata if available
    metadata = {}
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)
    
    return jsonify({
        'status': model_status['status'],
        'uptime_seconds': uptime,
        'uptime_hours': uptime / 3600,
        'total_predictions': model_status['total_predictions'],
        'last_training': model_status['last_training'],
        'metadata': metadata
    })

@app.route('/api/retrain/status')
def retrain_status_endpoint():
    """Get retraining status"""
    return jsonify(retraining_status)

@app.route('/api/retrain/upload', methods=['POST'])
def upload_retrain_data():
    """Upload data for retraining"""
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files[]')
        category = request.form.get('category', 'unknown')  # 'cats' or 'dogs'
        
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        saved_files = []
        category_folder = os.path.join(app.config['RETRAIN_FOLDER'], category)
        os.makedirs(category_folder, exist_ok=True)
        
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_')
                filename = timestamp + filename
                filepath = os.path.join(category_folder, filename)
                file.save(filepath)
                saved_files.append(filename)
        
        return jsonify({
            'message': f'Successfully uploaded {len(saved_files)} files',
            'files': saved_files,
            'category': category
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain/trigger', methods=['POST'])
def trigger_retraining():
    """Trigger model retraining"""
    global retraining_status
    
    if retraining_lock.locked():
        return jsonify({'error': 'Retraining already in progress'}), 400
    
    def retrain_worker():
        global retraining_status, predictor, model_status
        
        with retraining_lock:
            retraining_status['in_progress'] = True
            retraining_status['progress'] = 0
            retraining_status['message'] = 'Starting retraining...'
        
        try:
            # Load new data
            retraining_status['message'] = 'Loading new data...'
            retraining_status['progress'] = 10
            
            cats_dir = os.path.join(app.config['RETRAIN_FOLDER'], 'cats')
            dogs_dir = os.path.join(app.config['RETRAIN_FOLDER'], 'dogs')
            
            if not os.path.exists(cats_dir) or not os.path.exists(dogs_dir):
                raise ValueError("No retraining data found. Please upload data first.")
            
            from src.preprocessing import load_images_from_directory
            X_cats = load_images_from_directory(cats_dir, target_size=(224, 224))
            X_dogs = load_images_from_directory(dogs_dir, target_size=(224, 224))
            
            if len(X_cats) == 0 or len(X_dogs) == 0:
                raise ValueError("Insufficient data for retraining")
            
            X_new = np.concatenate([X_cats, X_dogs], axis=0)
            y_new = np.concatenate([
                np.zeros(len(X_cats)),
                np.ones(len(X_dogs))
            ])
            
            # Shuffle
            indices = np.random.permutation(len(X_new))
            X_new = X_new[indices]
            y_new = y_new[indices]
            
            retraining_status['message'] = 'Retraining model...'
            retraining_status['progress'] = 30
            
            # Retrain model
            model, history = retrain_model(
                X_new, y_new,
                base_model_path=MODEL_PATH,
                new_model_path=MODEL_PATH,
                epochs=5,
                batch_size=16
            )
            
            retraining_status['progress'] = 80
            retraining_status['message'] = 'Evaluating model...'
            
            # Evaluate
            # Split for validation
            split_idx = int(len(X_new) * 0.8)
            X_val = X_new[split_idx:]
            y_val = y_new[split_idx:]
            
            metrics = evaluate_model(model, X_val, y_val)
            save_model_metadata(metrics, METADATA_PATH)
            
            # Reload predictor
            predictor = CatDogPredictor(MODEL_PATH)
            model_status['last_training'] = datetime.now().isoformat()
            model_status['status'] = 'ready'
            
            retraining_status['progress'] = 100
            retraining_status['message'] = 'Retraining completed successfully!'
            retraining_status['in_progress'] = False
            
        except Exception as e:
            retraining_status['message'] = f'Error: {str(e)}'
            retraining_status['in_progress'] = False
            print(f"Retraining error: {e}")
    
    # Start retraining in background thread
    thread = threading.Thread(target=retrain_worker)
    thread.daemon = True
    thread.start()
    
    return jsonify({'message': 'Retraining started'}), 200

@app.route('/api/stats')
def get_stats():
    """Get statistics for visualizations"""
    try:
        # Load metadata
        metadata = {}
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                metadata = json.load(f)
        
        # Calculate dataset statistics
        train_cats_dir = 'data/train/cats'
        train_dogs_dir = 'data/train/dogs'
        test_cats_dir = 'data/test/cats'
        test_dogs_dir = 'data/test/dogs'
        
        train_cats_count = len([f for f in os.listdir(train_cats_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        train_dogs_count = len([f for f in os.listdir(train_dogs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        test_cats_count = len([f for f in os.listdir(test_cats_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        test_dogs_count = len([f for f in os.listdir(test_dogs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        retrain_cats_dir = os.path.join(app.config['RETRAIN_FOLDER'], 'cats')
        retrain_dogs_dir = os.path.join(app.config['RETRAIN_FOLDER'], 'dogs')
        retrain_cats_count = len([f for f in os.listdir(retrain_cats_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(retrain_cats_dir) else 0
        retrain_dogs_count = len([f for f in os.listdir(retrain_dogs_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(retrain_dogs_dir) else 0
        
        stats = {
            'dataset': {
                'train_cats': train_cats_count,
                'train_dogs': train_dogs_count,
                'test_cats': test_cats_count,
                'test_dogs': test_dogs_count,
                'retrain_cats': retrain_cats_count,
                'retrain_dogs': retrain_dogs_count
            },
            'model_metrics': metadata.get('metrics', {}),
            'predictions': {
                'total': model_status['total_predictions']
            }
        }
        
        return jsonify(stats), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)

