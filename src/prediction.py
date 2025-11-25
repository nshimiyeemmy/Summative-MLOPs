"""
Prediction module for cat and dog classification
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
try:
    from .preprocessing import preprocess_single_image, preprocess_uploaded_image
except ImportError:
    from preprocessing import preprocess_single_image, preprocess_uploaded_image
import os


class CatDogPredictor:
    """Class for making predictions on cat and dog images"""
    
    def __init__(self, model_path='models/cat_dog_model.h5'):
        """
        Initialize predictor with model
        
        Args:
            model_path: Path to saved model
        """
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the model from file"""
        if os.path.exists(self.model_path):
            self.model = keras.models.load_model(self.model_path)
        else:
            raise FileNotFoundError(f"Model not found at {self.model_path}")
    
    def predict(self, image_path=None, image_file=None, return_proba=False):
        """
        Predict whether image is a cat or dog
        
        Args:
            image_path: Path to image file
            image_file: File object from upload
            return_proba: Whether to return probability scores
        
        Returns:
            Prediction dictionary with class and confidence
        """
        if self.model is None:
            self.load_model()
        
        # Preprocess image
        if image_path:
            preprocessed_img = preprocess_single_image(image_path)
        elif image_file:
            preprocessed_img = preprocess_uploaded_image(image_file)
        else:
            raise ValueError("Either image_path or image_file must be provided")
        
        # Make prediction
        predictions = self.model.predict(preprocessed_img, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        class_name = "Dog" if predicted_class == 1 else "Cat"
        
        result = {
            'class': class_name,
            'class_id': int(predicted_class),
            'confidence': confidence,
            'probabilities': {
                'cat': float(predictions[0][0]),
                'dog': float(predictions[0][1])
            }
        }
        
        if return_proba:
            result['raw_probabilities'] = predictions[0].tolist()
        
        return result
    
    def predict_batch(self, image_paths):
        """
        Predict on multiple images
        
        Args:
            image_paths: List of image paths
        
        Returns:
            List of prediction dictionaries
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.predict(image_path=img_path)
                result['image_path'] = img_path
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': img_path,
                    'error': str(e)
                })
        
        return results

