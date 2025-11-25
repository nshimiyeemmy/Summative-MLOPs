"""
Data preprocessing module for cat and dog image classification
"""
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import cv2


def load_images_from_directory(directory, target_size=(224, 224), max_images=None):
    """
    Load images from directory and preprocess them
    
    Args:
        directory: Path to directory containing images
        target_size: Target size for resizing images
        max_images: Maximum number of images to load (None for all)
    
    Returns:
        numpy array of preprocessed images
    """
    images = []
    image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if max_images:
        image_files = image_files[:max_images]
    
    for img_file in image_files:
        img_path = os.path.join(directory, img_file)
        try:
            # Load and resize image
            img = Image.open(img_path)
            img = img.convert('RGB')
            img = img.resize(target_size)
            
            # Convert to array and normalize
            img_array = np.array(img) / 255.0
            images.append(img_array)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
    
    return np.array(images)


def prepare_dataset(train_cats_dir, train_dogs_dir, test_cats_dir, test_dogs_dir, 
                   target_size=(224, 224), max_train=None, max_test=None):
    """
    Prepare dataset for training and testing
    
    Args:
        train_cats_dir: Directory with training cat images
        train_dogs_dir: Directory with training dog images
        test_cats_dir: Directory with test cat images
        test_dogs_dir: Directory with test dog images
        target_size: Target size for images
        max_train: Maximum training images per class
        max_test: Maximum test images per class
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    print("Loading training images...")
    train_cats = load_images_from_directory(train_cats_dir, target_size, max_train)
    train_dogs = load_images_from_directory(train_dogs_dir, target_size, max_train)
    
    print("Loading test images...")
    test_cats = load_images_from_directory(test_cats_dir, target_size, max_test)
    test_dogs = load_images_from_directory(test_dogs_dir, target_size, max_test)
    
    # Combine and create labels
    X_train = np.concatenate([train_cats, train_dogs], axis=0)
    y_train = np.concatenate([
        np.zeros(len(train_cats)),  # 0 for cats
        np.ones(len(train_dogs))    # 1 for dogs
    ])
    
    X_test = np.concatenate([test_cats, test_dogs], axis=0)
    y_test = np.concatenate([
        np.zeros(len(test_cats)),
        np.ones(len(test_dogs))
    ])
    
    # Shuffle training data
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    # Shuffle test data
    indices = np.random.permutation(len(X_test))
    X_test = X_test[indices]
    y_test = y_test[indices]
    
    print(f"Training set: {len(X_train)} images")
    print(f"Test set: {len(X_test)} images")
    
    return X_train, X_test, y_train, y_test


def augment_data(X, y, augment_factor=2):
    """
    Augment data using rotation, flip, and brightness adjustments
    
    Args:
        X: Image array
        y: Labels
        augment_factor: How many times to augment
    
    Returns:
        Augmented X and y
    """
    augmented_X = []
    augmented_y = []
    
    for i in range(len(X)):
        img = X[i]
        label = y[i]
        
        # Original
        augmented_X.append(img)
        augmented_y.append(label)
        
        # Augmentations
        for _ in range(augment_factor):
            # Random horizontal flip
            if np.random.random() > 0.5:
                img_aug = np.fliplr(img)
            else:
                img_aug = img.copy()
            
            # Random brightness
            brightness = np.random.uniform(0.8, 1.2)
            img_aug = np.clip(img_aug * brightness, 0, 1)
            
            augmented_X.append(img_aug)
            augmented_y.append(label)
    
    return np.array(augmented_X), np.array(augmented_y)


def preprocess_single_image(image_path, target_size=(224, 224)):
    """
    Preprocess a single image for prediction
    
    Args:
        image_path: Path to image file
        target_size: Target size for resizing
    
    Returns:
        Preprocessed image array
    """
    try:
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")


def preprocess_uploaded_image(image_file, target_size=(224, 224)):
    """
    Preprocess an uploaded image file
    
    Args:
        image_file: File object from upload
        target_size: Target size for resizing
    
    Returns:
        Preprocessed image array
    """
    try:
        img = Image.open(image_file)
        img = img.convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        raise ValueError(f"Error preprocessing uploaded image: {e}")

