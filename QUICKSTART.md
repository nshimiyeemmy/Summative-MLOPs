# Quick Start Guide

## 🚀 Fastest Way to Get Started

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Train the Model (First Time Only)
```bash
python train_model.py
```
*This will take 10-30 minutes depending on your hardware*

### Step 3: Run the Application
```bash
python app.py
```

### Step 4: Open in Browser
Navigate to: `http://localhost:5000`

## 🐳 Using Docker (Alternative)

```bash
# Build and run
docker-compose up --build

# Access at http://localhost:5000
```

## 📝 What to Do Next

1. **Test Prediction**: Go to "Prediction" tab and upload a cat or dog image
2. **View Visualizations**: Check the "Visualizations" tab for dataset insights
3. **Retrain Model**: Upload new images in "Retrain" tab and trigger retraining
4. **Monitor Status**: View model uptime and metrics in "Model Status" tab

## 🧪 Load Testing

```bash
# Install locust if not already installed
pip install locust

# Run locust
locust -f locustfile.py --host=http://localhost:5000

# Open http://localhost:8089 in browser
# Set users and spawn rate, then start test
```

## ⚠️ Troubleshooting

**Model not found?**
- Run `python train_model.py` first
- Or use the Jupyter notebook: `jupyter notebook notebook/cat_dog_classification.ipynb`

**Port already in use?**
- Change port in `app.py`: `app.run(host='0.0.0.0', port=5001)`

**Import errors?**
- Make sure you're in the project root directory
- Activate virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)

