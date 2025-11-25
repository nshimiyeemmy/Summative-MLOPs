"""
Locust load testing script for Cat and Dog Classification API
"""
from locust import HttpUser, task, between
import random
import os
import io


class MLPipelineUser(HttpUser):
    """Simulates a user interacting with the ML Pipeline API"""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a simulated user starts"""
        # Health check
        self.client.get("/api/health")
    
    @task(3)
    def predict_image(self):
        """Test prediction endpoint - higher weight (3x)"""
        # Create a dummy image file for testing
        # In real scenario, you would use actual image files
        try:
            # Create a simple test image (1x1 pixel PNG)
            test_image = io.BytesIO()
            # For actual testing, you might want to use a real image file
            # For now, we'll simulate with a small request
            
            # Try to get a random image from test data if available
            test_images_dir = "data/test"
            if os.path.exists(test_images_dir):
                # Find a random image
                import glob
                images = []
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    images.extend(glob.glob(os.path.join(test_images_dir, '**', ext), recursive=True))
                
                if images:
                    image_path = random.choice(images)
                    with open(image_path, 'rb') as f:
                        files = {'image': f}
                        self.client.post("/api/predict", files=files, name="predict")
                else:
                    # Fallback: just test the endpoint (will fail but tests load)
                    self.client.post("/api/predict", files={'image': test_image}, name="predict")
            else:
                # Fallback: just test the endpoint
                self.client.post("/api/predict", files={'image': test_image}, name="predict")
        except Exception as e:
            print(f"Error in predict_image: {e}")
    
    @task(1)
    def get_model_status(self):
        """Get model status"""
        self.client.get("/api/model/status", name="model_status")
    
    @task(1)
    def get_stats(self):
        """Get statistics for visualizations"""
        self.client.get("/api/stats", name="stats")
    
    @task(1)
    def get_health(self):
        """Health check"""
        self.client.get("/api/health", name="health")
    
    @task(1)
    def get_retrain_status(self):
        """Get retraining status"""
        self.client.get("/api/retrain/status", name="retrain_status")
    
    @task(1)
    def get_homepage(self):
        """Access homepage"""
        self.client.get("/", name="homepage")

