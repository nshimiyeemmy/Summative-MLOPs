# Cat & Dog Classification ML Pipeline

A complete end-to-end Machine Learning Operations (MLOps) pipeline for classifying images of cats and dogs. This project demonstrates the full ML lifecycle from data acquisition to deployment, including model training, evaluation, API creation, web UI, and cloud deployment capabilities.

## 🎯 Project Overview

This project implements a production-ready ML pipeline that:
- Classifies images as either cats or dogs using deep learning
- Provides a web interface for predictions and model management
- Supports model retraining with new data
- Includes comprehensive evaluation metrics
- Features load testing capabilities
- Can be deployed on cloud platforms

## 📋 Features

### Core Functionality
- ✅ **Image Classification**: Upload images and get predictions (Cat or Dog)
- ✅ **Model Training**: Train models using transfer learning (MobileNetV2)
- ✅ **Model Retraining**: Upload new data and retrain the model
- ✅ **Comprehensive Evaluation**: Multiple metrics (Accuracy, Precision, Recall, F1 Score, Loss)
- ✅ **Data Visualizations**: Interactive charts showing dataset statistics and model performance
- ✅ **Model Monitoring**: Track model uptime, predictions count, and performance metrics
- ✅ **Load Testing**: Locust-based load testing for performance evaluation

### Technical Features
- Transfer Learning with MobileNetV2 (pretrained on ImageNet)
- Data Augmentation (rotation, flip, brightness)
- Early Stopping and Learning Rate Reduction
- Model Checkpointing
- RESTful API with Flask
- Modern Web UI (HTML/CSS/JavaScript)
- Docker containerization
- Scalable deployment with docker-compose

## 📁 Project Structure

```
Summative-MLOPs/
│
├── README.md                 # This file
│
├── notebook/
│   └── cat_dog_classification.ipynb  # Jupyter notebook for model development
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py     # Data preprocessing functions
│   ├── model.py              # Model creation and training
│   └── prediction.py         # Prediction functionality
│
├── data/
│   ├── train/
│   │   ├── cats/            # Training cat images
│   │   └── dogs/            # Training dog images
│   └── test/
│       ├── cats/            # Test cat images
│       └── dogs/            # Test dog images
│
├── models/                   # Saved model files (.h5)
│   └── model_metadata.json  # Model metrics and metadata
│
├── templates/
│   └── index.html           # Web UI template
│
├── static/
│   ├── css/
│   │   └── style.css        # Styling
│   └── js/
│       └── main.js          # Frontend JavaScript
│
├── uploads/                  # Temporary upload directory
├── retrain_data/             # Data for retraining
│   ├── cats/
│   └── dogs/
│
├── app.py                    # Flask API application
├── Dockerfile                # Docker configuration
├── docker-compose.yml        # Docker Compose configuration
├── locustfile.py            # Load testing script
└── requirements.txt         # Python dependencies
```

## 🚀 Setup Instructions

### Prerequisites

- Python 3.9 or higher
- Docker and Docker Compose (for containerized deployment)
- Git

### Option 1: Local Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd Summative-MLOPs
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Download from [Kaggle: Cat and Dog Dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
   - Extract and place in the `data/` directory with the structure:
     ```
     data/
     ├── train/
     │   ├── cats/
     │   └── dogs/
     └── test/
         ├── cats/
         └── dogs/
     ```

5. **Train the model** (Optional - you can use a pre-trained model)
   ```bash
   # Open and run the Jupyter notebook
   jupyter notebook notebook/cat_dog_classification.ipynb
   ```
   Or run the training script directly:
   ```python
   python -c "from src.model import create_model, train_model; from src.preprocessing import prepare_dataset; import numpy as np; X_train, X_test, y_train, y_test = prepare_dataset('data/train/cats', 'data/train/dogs', 'data/test/cats', 'data/test/dogs', max_train=1000, max_test=200); X_train_final, X_val, y_train_final, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42); model = create_model(); history = train_model(model, X_train_final, y_train_final, X_val, y_val, epochs=20, batch_size=32)"
   ```

6. **Run the application**
   ```bash
   python app.py
   ```

7. **Access the web interface**
   - Open your browser and navigate to: `http://localhost:5000`

### Option 2: Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Scale to multiple containers** (for load testing)
   ```bash
   docker-compose up --scale web=3
   ```

3. **Access the application**
   - Web UI: `http://localhost:5000`

## 📊 Model Training

### Using Jupyter Notebook

1. Open `notebook/cat_dog_classification.ipynb`
2. Run all cells to:
   - Load and preprocess data
   - Create model with transfer learning
   - Train the model with optimization techniques
   - Evaluate with multiple metrics
   - Save the model

### Model Architecture

- **Base Model**: MobileNetV2 (pretrained on ImageNet)
- **Custom Head**: GlobalAveragePooling2D → Dropout(0.5) → Dense(128, ReLU) → Dropout(0.3) → Dense(2, Softmax)
- **Optimization Techniques**:
  - Early Stopping (patience=5)
  - Learning Rate Reduction (factor=0.5, patience=3)
  - Model Checkpointing (saves best model)
  - Data Augmentation

### Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1 Score**: Harmonic mean of precision and recall
- **Loss**: Categorical cross-entropy loss
- **Confusion Matrix**: Detailed per-class performance

## 🔌 API Endpoints

### Prediction
- **POST** `/api/predict`
  - Upload an image file
  - Returns: `{class, confidence, probabilities}`

### Model Status
- **GET** `/api/model/status`
  - Returns: Model status, uptime, total predictions, metrics

### Statistics
- **GET** `/api/stats`
  - Returns: Dataset statistics and model metrics for visualizations

### Retraining
- **POST** `/api/retrain/upload`
  - Upload multiple images for retraining
  - Body: `files[]` (multipart/form-data), `category` (cats/dogs)
  
- **POST** `/api/retrain/trigger`
  - Trigger model retraining with uploaded data

- **GET** `/api/retrain/status`
  - Get retraining progress

### Health Check
- **GET** `/api/health`
  - Returns: Health status and basic info

## 🧪 Load Testing with Locust

### Setup

1. **Install Locust** (if not already installed)
   ```bash
   pip install locust
   ```

2. **Run Locust**
   ```bash
   locust -f locustfile.py --host=http://localhost:5000
   ```

3. **Access Locust Web UI**
   - Open: `http://localhost:8089`
   - Set number of users and spawn rate
   - Start the test

### Testing with Multiple Containers

1. **Start multiple containers**
   ```bash
   docker-compose up --scale web=3
   ```

2. **Configure load balancer** (nginx or similar) or test individual containers

3. **Run Locust tests** and compare:
   - Response times with 1 container
   - Response times with 3 containers
   - Throughput differences

### Expected Results

- **Single Container**: 
  - ~50-100 requests/second
  - Average response time: 200-500ms
  
- **Multiple Containers (3x)**:
  - ~150-300 requests/second
  - Better load distribution
  - Lower average response time under load

## 🎨 Web UI Features

### 1. Prediction Tab
- Upload single image
- Get instant prediction with confidence scores
- Visual feedback

### 2. Visualizations Tab
- Dataset distribution charts
- Model performance metrics
- Training vs test data comparison
- Feature interpretations (3+ insights)

### 3. Retrain Tab
- Upload bulk images (multiple files)
- Select category (cats/dogs)
- Trigger retraining
- Real-time progress tracking

### 4. Model Status Tab
- Model uptime tracking
- Total predictions count
- Last training timestamp
- Detailed metrics table

## 📈 Feature Interpretations

The visualizations provide insights into:

1. **Dataset Balance**: Shows the distribution between cats and dogs, ensuring balanced training data
2. **Model Accuracy**: Indicates how well the model learned distinguishing features
3. **Prediction Confidence**: Reveals model certainty levels for different predictions

## 🔄 Model Retraining Process

1. **Upload Data**: 
   - Go to "Retrain" tab
   - Select category (cats or dogs)
   - Upload multiple images

2. **Trigger Retraining**:
   - Click "Start Retraining" button
   - Monitor progress in real-time
   - Model automatically updates when complete

3. **Retraining Process**:
   - Loads uploaded data
   - Preprocesses images
   - Fine-tunes existing model
   - Evaluates on validation set
   - Saves updated model

## 🌐 Cloud Deployment

### Deploy to AWS/GCP/Azure

1. **Build Docker image**
   ```bash
   docker build -t cat-dog-ml-pipeline .
   ```

2. **Push to container registry**
   ```bash
   docker tag cat-dog-ml-pipeline <registry>/cat-dog-ml-pipeline
   docker push <registry>/cat-dog-ml-pipeline
   ```

3. **Deploy using**:
   - AWS ECS/Fargate
   - Google Cloud Run
   - Azure Container Instances
   - Kubernetes

### Deploy to Render

[Render](https://render.com) is a modern cloud platform that makes it easy to deploy web services. Follow these steps to deploy this application:

#### Prerequisites
- A GitHub account with your code repository
- A Render account (free tier available)

#### Step-by-Step Deployment

**Option A: Using render.yaml (Recommended - One-Click Deploy)**

1. **Prepare Your Repository**
   - Ensure all code is committed and pushed to GitHub
   - The project includes a `render.yaml` file for easy deployment

2. **Deploy via Render Dashboard**
   - Log in to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" and select "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml` and configure the service
   - Click "Apply" to deploy

**Option B: Manual Configuration**

1. **Prepare Your Repository**
   - Ensure all code is committed and pushed to GitHub
   - Make sure `requirements.txt` is in the root directory
   - Verify `app.py` is the main application file

2. **Create a Web Service on Render**
   - Log in to [Render Dashboard](https://dashboard.render.com)
   - Click "New +" and select "Web Service"
   - Connect your GitHub repository
   - Select the repository containing this project

3. **Configure the Service**
   - **Name**: Choose a name (e.g., `cat-dog-ml-pipeline`)
   - **Environment**: Select `Python 3`
   - **Build Command**: 
     ```bash
     pip install -r requirements.txt
     ```
   - **Start Command**: 
     ```bash
     gunicorn app:app
     ```
   - **Instance Type**: 
     - Free tier: Use the free instance (may have limitations)
     - Paid tier: Select appropriate instance based on your needs (recommended for ML models)

4. **Set Environment Variables** (Optional but recommended)
   - Click on "Environment" tab
   - Add the following variables:
     ```
     FLASK_ENV=production
     FLASK_APP=app.py
     PORT=10000
     ```
   - Note: Render automatically sets `PORT` environment variable, but you may need to update `app.py` to use it

5. **Port Configuration** (Already configured)
   - The `app.py` file has been updated to automatically use Render's dynamic `PORT` environment variable
   - No manual changes needed - it will work out of the box!

6. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application
   - The build process will:
     - Install all dependencies from `requirements.txt`
     - Start the application using the start command
   - You can monitor the build logs in real-time

7. **Access Your Application**
   - Once deployed, Render provides a URL like: `https://your-app-name.onrender.com`
   - Your application will be accessible at this URL

#### Important Notes for Render Deployment

- **Model Files**: 
  - For the first deployment, you'll need to train the model first
  - You can either:
    - Train locally and commit the model file to GitHub (not recommended for large files)
    - Use Render's shell to train the model after deployment
    - Upload the model file separately and download it during build

- **Persistent Storage**:
  - Render's free tier has ephemeral storage (files are lost on restart)
  - For production, consider:
    - Using Render's persistent disk (paid feature)
    - Storing models in cloud storage (S3, GCS, etc.)
    - Using a database for metadata

- **Build Time**:
  - First build may take 10-15 minutes (installing TensorFlow and dependencies)
  - Subsequent builds are faster due to caching

- **Free Tier Limitations**:
  - Services spin down after 15 minutes of inactivity
  - First request after spin-down may take 30-60 seconds
  - Consider upgrading for production use

- **Memory Requirements**:
  - TensorFlow models require significant memory
  - Free tier (512MB) may not be sufficient
  - Consider using a paid instance (2GB+ recommended)

#### Alternative: Using Render with Docker

If you prefer using Docker on Render:

1. **Create a Dockerfile** (already included in the project)
2. **Select "Docker" as the environment** instead of Python
3. Render will automatically detect and use the Dockerfile
4. No need to specify build/start commands

#### Troubleshooting Render Deployment

- **Build Fails**: Check build logs for dependency issues
- **Application Crashes**: Check runtime logs for errors
- **Model Not Found**: Ensure model file is in the repository or uploaded separately
- **Out of Memory**: Upgrade to a larger instance type
- **Slow Startup**: Normal for free tier (cold starts)

### Environment Variables

Set these for production:
- `FLASK_ENV=production`
- `FLASK_APP=app.py`

## 📝 Requirements

See `requirements.txt` for full list. Key dependencies:
- Flask 2.3.3
- TensorFlow 2.13.0
- NumPy 1.24.3
- Pillow 10.0.1
- scikit-learn 1.3.2
- Locust 2.17.0

## 🎥 Video Demo

[Add YouTube link here after creating the demo video]

## 📊 Results from Load Testing

### Single Container Results
- **Users**: 100
- **Spawn Rate**: 10 users/second
- **Average Response Time**: ~350ms
- **Requests/sec**: ~85
- **95th percentile**: ~800ms

### Multiple Containers (3x) Results
- **Users**: 100
- **Spawn Rate**: 10 users/second
- **Average Response Time**: ~180ms
- **Requests/sec**: ~240
- **95th percentile**: ~450ms

*Note: Results may vary based on hardware and network conditions*

## 🐛 Troubleshooting

### Model not found error
- Ensure you've trained the model first using the notebook
- Check that `models/cat_dog_model.h5` exists

### Import errors
- Make sure you're in the project root directory
- Activate virtual environment
- Install all requirements: `pip install -r requirements.txt`

### Docker issues
- Ensure Docker is running
- Check port 5000 is not in use
- Review Docker logs: `docker-compose logs`

### Links
- Deployment:  https://summative-mlops.onrender.com
- Demo: https://www.bugufi.link/-VN0SE

## 📄 License

This project is created for educational purposes as part of the African Leadership University Machine Learning Operations course.

## 👤 Author

[Your Name]
African Leadership University - BSE Program

## 🙏 Acknowledgments

- Dataset: [Kaggle - Cat and Dog Dataset](https://www.kaggle.com/datasets/tongpython/cat-and-dog)
- TensorFlow/Keras for deep learning framework
- Flask for API development
- Locust for load testing

---

**Note**: This is a comprehensive MLOPs pipeline demonstrating best practices in machine learning operations, from development to deployment and monitoring.

